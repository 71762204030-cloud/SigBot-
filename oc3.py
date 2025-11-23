"""
oc3.py  -- SIGBOT backend (FAISS retrieval + Ollama)
- Ollama-only (no OpenAI fallback)
- Default model: gemma:2b (small enough for low-RAM machines)
- Exposes load_index_and_metadata() used by the frontend
- Provides chatbot_response(question, index, metadata, embedding_model, mode, teach_mode)
"""

import os
import json
import requests
import faiss
import numpy as np
import time
import re
from typing import Optional, Tuple

# optional: sentence-transformers may be heavy; if not needed, you can stub it.
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

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

# ========= GLOBALS =========
answer_cache = {}
session_mode = "detailed"

# Lazy-loaded embedder
_embedding_model_instance = None

def get_embedding_model():
    global _embedding_model_instance
    if _embedding_model_instance is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not available. Install it or pass an embedding model.")
        _embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model_instance

# ========= UTIL =========
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    L = len(text)
    if L == 0:
        return []
    while start < L:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(0, end - overlap)
    return chunks

# ========= INDEX BUILD / LOAD =========
def build_faiss_index(pdf_folder: str = PDF_FOLDER) -> Tuple[faiss.IndexFlatL2, dict]:
    if SentenceTransformer is None:
        raise RuntimeError("SentenceTransformer required to build index. Install sentence-transformers.")

    texts = []
    sources = []

    try:
        import fitz  # pymupdf
    except Exception:
        raise RuntimeError("pymupdf required to parse PDFs. Install pymupdf.")

    for filename in os.listdir(pdf_folder or "."):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_folder, filename)
        try:
            doc = fitz.open(path)
        except Exception:
            continue
        full_text = ""
        for page in doc:
            try:
                full_text += page.get_text()
            except Exception:
                continue
        if not full_text:
            continue
        chunks = chunk_text(full_text)
        texts.extend(chunks)
        sources.extend([filename] * len(chunks))

    if not texts:
        # empty index
        dim = 384  # fallback for all-MiniLM-L6-v2
        index = faiss.IndexFlatL2(dim)
        meta = {"texts": [], "sources": []}
        os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
        faiss.write_index(index, INDEX_FILE)
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return index, meta

    embedder = get_embedding_model()
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, 0)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))

    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    faiss.write_index(index, INDEX_FILE)

    meta = {"texts": texts, "sources": sources}
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return index, meta

def load_faiss_index() -> Tuple[Optional[faiss.IndexFlatL2], Optional[dict]]:
    if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
        return None, None
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta
    except Exception:
        return None, None

# Frontend expects this exact function name
def load_index_and_metadata(rebuild_if_missing: bool = False) -> Tuple[Optional[faiss.IndexFlatL2], Optional[dict]]:
    """
    Returns (index, metadata)
    If no index exists and rebuild_if_missing=True -> attempt to build index (may require dependencies)
    """
    idx, meta = load_faiss_index()
    if idx is None and rebuild_if_missing:
        try:
            idx, meta = build_faiss_index()
        except Exception as e:
            print(f"[WARN] Failed to build index: {e}")
            return None, None
    return idx, meta

# ========= OLLAMA CALLS =========
def _ollama_raw_request(prompt: str, model: str, timeout: int = 60) -> (bool, str):
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            if "response" in data:
                return True, data.get("response", "").strip()
            if "text" in data:
                return True, data.get("text", "").strip()
            # Some Ollama responses stream many JSON rows; fall back to string
            return True, json.dumps(data)
        return True, str(data)
    except requests.exceptions.RequestException as e:
        return False, f"Ollama request failed: {e}"
    except ValueError as e:
        return False, f"Ollama returned invalid JSON: {e}"
    except Exception as e:
        return False, f"Ollama unknown error: {e}"

def ollama_prompt(prompt: str, model: Optional[str] = None, timeout: int = 60) -> str:
    model = model or OLLAMA_MODEL
    ok, resp = _ollama_raw_request(prompt, model, timeout=timeout)
    if ok and resp and not str(resp).lower().startswith("ollama request failed:"):
        return resp

    # Ollama failed — return a friendly guidance message (no OpenAI fallback)
    guidance = (
        "⚠️ Ollama is not reachable or failed to generate a response.\n\n"
        "To fix this on the backend host:\n"
        "  1) Ensure Ollama is installed and running: `ollama serve`\n"
        f"  2) Pull the expected model: `ollama pull {OLLAMA_MODEL}`\n"
        f"  3) Test: curl {OLLAMA_URL} -d '{{\"model\":\"{OLLAMA_MODEL}\", \"prompt\":\"hello\"}}'\n\n"
        "This instance is configured to use Ollama only (no cloud fallback)."
    )
    if ok is False:
        return f"Ollama error: {resp}\n\n{guidance}"
    return guidance

# ========= RETRIEVAL HELPERS =========
def sanitize_text_for_plain(s: str) -> str:
    if not s:
        return ""
    # remove HTML-like tags if any
    s = re.sub(r"<[^>]+>", "", s)
    # normalize whitespace
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()

def retrieve_context(query: str, index, metadata) -> str:
    if index is None or metadata is None or not metadata.get("texts"):
        return ""
    # embed query
    try:
        embedder = get_embedding_model()
        qv = embedder.encode([query], convert_to_numpy=True)
    except Exception:
        # cannot embed: return empty context
        return ""
    D, I = index.search(np.array(qv, dtype="float32"), TOP_K)
    parts = []
    for idx in I[0]:
        try:
            txt = metadata["texts"][int(idx)]
            src = metadata["sources"][int(idx)] if "sources" in metadata else None
            if src:
                parts.append(f"(source: {src}) {txt}")
            else:
                parts.append(txt)
        except Exception:
            continue
    joined = "\n\n---\n\n".join(parts)
    return joined[:CONTEXT_TRIM_CHARS]

# ========= MAIN CHATBOT FUNCTION =========
def chatbot_response(user_query: str, index, metadata, embedding_model=None, mode: str = "quick", teach_mode: bool = False) -> str:
    """
    Produces a plain-text answer using retrieved context + Ollama model.
    Keeps outputs safe and returns a helpful message if model/backend is unavailable.
    """
    cache_key = (user_query.strip().lower(), mode, teach_mode)
    if cache_key in answer_cache:
        return answer_cache[cache_key]

    # Prepare context
    ctx = retrieve_context(user_query, index, metadata)

    # Compose system prompt
    prompt = (
        "You are SIGBOT, a helpful assistant that answers clearly and concisely.\n\n"
        "Use the CONTEXT where relevant. If the answer is not in the context, respond concisely.\n\n"
        "CONTEXT:\n"
        f"{ctx}\n\n"
        "QUESTION:\n"
        f"{user_query}\n\n"
        "FORMAT: Provide a clear plain-text answer. If a formula is needed, use ASCII only.\n\n"
        "ANSWER:"
    )

    raw = ollama_prompt(prompt, model=OLLAMA_MODEL, timeout=60)
    cleaned = sanitize_text_for_plain(str(raw))

    answer_cache[cache_key] = cleaned
    return cleaned

# ========= OPTIONAL CLI for testing =========
def main(rebuild_index: bool = False):
    idx, meta = load_index_and_metadata(rebuild_if_missing=rebuild_index)
    print("SIGBOT backend ready. (Ollama-only)")
    while True:
        try:
            q = input("\nYou: ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                break
            resp = chatbot_response(q, idx, meta)
            print("\nBot:", resp)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    main(rebuild_index=args.rebuild)
