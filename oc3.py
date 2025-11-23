import os
import json
import logging
import requests
import traceback
from typing import Optional, List

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    import fitz  # PyMuPDF
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False

# -----------------------------------------------------------
# Logging setup
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------------------------------------------
# File paths
# -----------------------------------------------------------
PDF_FOLDER = "data/raw_pdfs"
INDEX_FILE = "data/processed/index.faiss"
META_FILE = "data/processed/meta.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000

# -----------------------------------------------------------
# Ollama / OpenAI URLs
# -----------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# -----------------------------------------------------------
# Utility: text chunking
# -----------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# -----------------------------------------------------------
# Utility: extract PDF text
# -----------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with fitz.open(pdf_path) as doc:
            return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        logging.error(f"PDF extract failed for {pdf_path}: {e}")
        return ""


# -----------------------------------------------------------
# FAISS index building
# -----------------------------------------------------------
def build_index_from_pdfs(embedding_model: SentenceTransformer):
    texts = []
    metadata = {}

    if os.path.exists(PDF_FOLDER):
        for filename in os.listdir(PDF_FOLDER):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(PDF_FOLDER, filename)
                text = extract_text_from_pdf(pdf_path)
                if not text.strip():
                    continue
                for chunk in chunk_text(text):
                    texts.append(chunk)
                    metadata[len(metadata)] = {"source": filename}

    if not texts:
        raise ValueError("No PDFs found to index.")

    logging.info("Encoding %d chunks...", len(texts))
    embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs("data/processed", exist_ok=True)
    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    return index, metadata


# -----------------------------------------------------------
# FAISS + metadata loading
# -----------------------------------------------------------
def normalize_metadata(raw_meta):
    if isinstance(raw_meta, dict):
        return raw_meta
    if isinstance(raw_meta, list):
        return {str(i): item for i, item in enumerate(raw_meta)}
    raise ValueError("Invalid metadata format.")


def load_index_and_metadata():
    """
    This is the function the FRONTEND expects.
    """
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        try:
            index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, encoding="utf-8") as f:
                metadata_raw = json.load(f)
            metadata = normalize_metadata(metadata_raw)
            logging.info("Index + metadata loaded successfully.")
            return index, metadata
        except Exception:
            traceback.print_exc()

    logging.warning("Index not found — creating empty placeholder.")
    return None, None


# -----------------------------------------------------------
# OLLAMA raw request
# -----------------------------------------------------------
def _ollama_raw_request(prompt: str, model: str, timeout: int = 600) -> (bool, str):
    payload = {"model": model, "prompt": prompt, "stream": False}
    headers = {"Content-Type": "application/json"}

    try:
        logging.info("Calling Ollama at %s ...", OLLAMA_URL)
        resp = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=timeout)

        if resp.status_code >= 400:
            logging.error("Ollama Error %d: %s", resp.status_code, resp.text)
            return False, f"Ollama HTTP {resp.status_code}: {resp.text.strip()}"

        try:
            data = resp.json()
        except:
            return True, resp.text.strip()

        if isinstance(data, dict):
            if "response" in data and data["response"]:
                return True, data["response"].strip()
            if "text" in data and data["text"]:
                return True, data["text"].strip()
            if "error" in data:
                return False, f"Ollama error: {data['error']}"

        return True, json.dumps(data)

    except Exception as e:
        logging.exception("Ollama exception")
        return False, f"Ollama exception: {e}"


# -----------------------------------------------------------
# OpenAI fallback
# -----------------------------------------------------------
def _openai_fallback(prompt: str) -> (bool, str):
    if not OPENAI_API_KEY:
        return False, "OpenAI fallback error: OPENAI_API_KEY missing."

    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.2,
        }

        resp = requests.post(url, json=payload, headers=headers)

        if resp.status_code != 200:
            return False, f"OpenAI HTTP {resp.status_code}: {resp.text}"

        data = resp.json()
        return True, data["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return False, f"OpenAI error: {e}"


# -----------------------------------------------------------
# Main query handler
# -----------------------------------------------------------
def ollama_prompt(prompt: str, model: Optional[str] = None, timeout: int = 600) -> str:
    model = model or OLLAMA_MODEL

    ok, resp = _ollama_raw_request(prompt, model, timeout)
    if ok:
        return resp

    ok2, resp2 = _openai_fallback(prompt)
    if ok2:
        return resp2

    return (
        "ERROR:\n"
        f"Ollama failed: {resp}\n\n"
        f"OpenAI fallback failed: {resp2}\n"
    )


# -----------------------------------------------------------
# Function frontend calls
# -----------------------------------------------------------
def chatbot_response(user_input: str) -> str:
    return ollama_prompt(user_input)
