import os
import json
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time

# ========= CONFIG =========
PDF_FOLDER = "data/raw_pdfs"
INDEX_FILE = "data/processed/index.faiss"
META_FILE = "data/processed/meta.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# OLLAMA CONFIG
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma:2b")   # DEFAULT MODEL

TOP_K = 3
CONTEXT_TRIM_CHARS = 1200

# ========= LOAD EMBEDDINGS =========
embedder = SentenceTransformer(EMBEDDING_MODEL)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# ========= PDF PROCESSOR =========
def build_faiss_index(pdf_folder=PDF_FOLDER):
    texts = []
    sources = []

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            import fitz  # pymupdf
            path = os.path.join(pdf_folder, filename)
            doc = fitz.open(path)
            full_text = ""
            for page in doc:
                full_text += page.get_text()

            chunks = chunk_text(full_text)
            texts.extend(chunks)
            sources.extend([filename] * len(chunks))

    embeddings = embedder.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "sources": sources}, f, indent=4)

    return index, {"texts": texts, "sources": sources}


def load_faiss_index():
    if not os.path.exists(INDEX_FILE):
        return None, None
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

# ========= OLLAMA REQUEST =========
def ollama_request(prompt, model=OLLAMA_MODEL):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt},
            timeout=60
        )
        if response.status_code != 200:
            return f"[Ollama Error] {response.text}"
        return response.text
    except Exception as e:
        return f"[Ollama Connection Error] {str(e)}\n" \
               f"🔥 FIX:\n" \
               f"1) Open terminal → run:  ollama serve\n" \
               f"2) Pull model →       ollama pull {OLLAMA_MODEL}\n" \
               f"3) Test: curl http://localhost:11434/api/generate -d '{{\"model\":\"{OLLAMA_MODEL}\",\"prompt\":\"hello\"}}'"

# ========= RETRIEVAL =========
def retrieve_context(query, index, metadata):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, TOP_K)

    ctx = ""
    for idx in I[0]:
        ctx += metadata["texts"][idx] + "\n---\n"

    return ctx[:CONTEXT_TRIM_CHARS]

# ========= CHATBOT RESPONSE =========
def chatbot_response(question, index=None, metadata=None, mode="default", teach_mode=False):
    context = ""
    if index is not None:
        context = retrieve_context(question, index, metadata)

    prompt = (
        f"You are SIGBOT. Answer clearly.\n\n"
        f"CONTEXT:\n{context}\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )

    result = ollama_request(prompt)
    return result

# ========= MAIN LOOP (Optional CLI) =========
def main(rebuild_index=False):
    if rebuild_index:
        print("Rebuilding PDF index...")
        index, metadata = build_faiss_index()
    else:
        index, metadata = load_faiss_index()

    print("SIGBOT ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        answer = chatbot_response(user_input, index, metadata)
        print("\nBot:", answer)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    main(rebuild_index=args.rebuild)
