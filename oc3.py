import os
import json
import logging
import requests
import traceback
from typing import Optional, List, Any

# Attempt to import heavy dependencies; code will behave gracefully if they are missing.
HAS_SENTENCE_TRANSFORMERS = False
HAS_FAISS = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None

try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except Exception:
    faiss = None
    np = None

# -----------------------------------------------------------
# Logging setup
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------------------------------------------
# File paths and defaults
# -----------------------------------------------------------
PDF_FOLDER = "data/raw_pdfs"
INDEX_FILE = "data/processed/index.faiss"
META_FILE = "data/processed/meta.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000

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
# Utility: extract PDF text (PyMuPDF / fitz)
# -----------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        import fitz  # PyMuPDF
        with fitz.open(pdf_path) as doc:
            return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        logging.error("PDF extract failed for %s: %s", pdf_path, e)
        return ""


# -----------------------------------------------------------
# FAISS index building
# -----------------------------------------------------------
def build_index_from_pdfs(embedding_model: Optional[Any] = None):
    """
    Build a FAISS index from PDFs found in PDF_FOLDER.
    Returns: (index, metadata)
    """
    if not HAS_SENTENCE_TRANSFORMERS or not HAS_FAISS:
        raise RuntimeError("FAISS and/or sentence-transformers not available in this environment.")

    if embedding_model is None:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)

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
                    metadata[len(metadata)] = {"source": filename, "text": chunk}

    if not texts:
        raise ValueError("No PDFs found to index.")

    logging.info("Encoding %d chunks...", len(texts))
    embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    faiss.write_index(index, INDEX_FILE)

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    logging.info("Index built and saved to %s", INDEX_FILE)
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
    Loads FAISS index and metadata if available. Returns (index, metadata) or (None, None).
    """
    if not HAS_FAISS:
        logging.warning("FAISS not available in this environment.")
        return None, None

    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        try:
            idx = faiss.read_index(INDEX_FILE)
            with open(META_FILE, encoding="utf-8") as f:
                metadata_raw = json.load(f)
            metadata = normalize_metadata(metadata_raw)
            logging.info("Index + metadata loaded successfully.")
            return idx, metadata
        except Exception:
            logging.exception("Failed to load index/metadata")
            return None, None

    logging.info("Index or metadata not found at expected paths.")
    return None, None


# -----------------------------------------------------------
# Retrieval: embed query and run FAISS search
# -----------------------------------------------------------
def retrieve_context(index, metadata, query: str, top_k: int = 4, embedding_model_name: Optional[str] = None) -> List[str]:
    """
    Retrieve top_k context strings from a FAISS index for the given query.
    Returns list of context text items (could be empty).
    """
    try:
        if index is None or metadata is None:
            logging.info("No index/metadata provided for retrieval.")
            return []

        if not HAS_SENTENCE_TRANSFORMERS:
            logging.warning("sentence-transformers not available; skipping retrieval.")
            return []

        # embed the query
        model_name = embedding_model_name or EMBEDDING_MODEL
        embedder = SentenceTransformer(model_name)
        q_emb = embedder.encode([query], convert_to_numpy=True)
        q_emb = np.array(q_emb, dtype="float32")

        if not hasattr(index, "search"):
            logging.warning("Provided index does not support search().")
            return []

        D, I = index.search(q_emb, top_k)
        contexts = []
        for idx in I[0]:
            key = str(idx)
            # metadata was stored keyed by numeric string (we used metadata[len(metadata)] = ...)
            item = metadata.get(key) or metadata.get(int(key)) if isinstance(metadata, dict) else None
            # if item contains 'text', use it; else try a best-effort representation
            if isinstance(item, dict):
                text_piece = item.get("text") or item.get("content") or item.get("source") or json.dumps(item)
            elif item is not None:
                text_piece = str(item)
            else:
                text_piece = f"<no metadata for id={key}>"
            contexts.append(f"Source {key}: {text_piece}")
        return contexts
    except Exception as e:
        logging.exception("retrieve_context error")
        return []


# -----------------------------------------------------------
# OLLAMA raw request
# -----------------------------------------------------------
def _ollama_raw_request(prompt: str, model: str, timeout: int = 600) -> (bool, str):
    payload = {"model": model, "prompt": prompt, "stream": False}
    headers = {"Content-Type": "application/json"}

    try:
        logging.info("Calling Ollama at %s ...", OLLAMA_URL)
        resp = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=timeout)

        logging.info("Ollama status: %s", resp.status_code)

        if resp.status_code >= 400:
            logging.error("Ollama Error %s: %s", resp.status_code, resp.text)
            return False, f"Ollama HTTP {resp.status_code}: {resp.text.strip()}"

        try:
            data = resp.json()
        except Exception:
            logging.warning("Ollama returned non-JSON response")
            return True, resp.text.strip()

        if isinstance(data, dict):
            if "response" in data and data["response"]:
                return True, data["response"].strip()
            if "text" in data and data["text"]:
                return True, data["text"].strip()
            if "error" in data:
                logging.error("Ollama error body: %s", data["error"])
                return False, f"Ollama returned error: {data['error']}"

            if "choices" in data and isinstance(data["choices"], (list, tuple)) and len(data["choices"]) > 0:
                c0 = data["choices"][0]
                if isinstance(c0, dict):
                    if "message" in c0 and isinstance(c0["message"], dict) and "content" in c0["message"]:
                        return True, c0["message"]["content"].strip()
                    if "text" in c0:
                        return True, c0["text"].strip()

        return True, json.dumps(data)

    except requests.exceptions.RequestException as e:
        logging.exception("Ollama request exception")
        return False, f"Ollama request failed: {e}"
    except Exception as e:
        logging.exception("Ollama unknown exception")
        return False, f"Ollama unknown error: {e}"


# -----------------------------------------------------------
# OpenAI fallback
# -----------------------------------------------------------
def _openai_fallback(prompt: str) -> (bool, str):
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY missing for OpenAI fallback.")
        return False, "OpenAI fallback error: OPENAI_API_KEY is not set."

    try:
        logging.info("Calling OpenAI fallback...")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.2
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        logging.info("OpenAI status: %s", resp.status_code)
        if resp.status_code != 200:
            logging.error("OpenAI HTTP %s: %s", resp.status_code, resp.text)
            return False, f"OpenAI HTTP {resp.status_code}: {resp.text}"

        data = resp.json()
        if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return True, choice["message"]["content"].strip()
            if "text" in choice:
                return True, choice["text"].strip()

        logging.warning("OpenAI returned unexpected shape: %s", data)
        return True, json.dumps(data)
    except requests.exceptions.RequestException as e:
        logging.exception("OpenAI request exception")
        return False, f"OpenAI request failed: {e}"
    except Exception as e:
        logging.exception("OpenAI unknown exception")
        return False, f"OpenAI unknown error: {e}"


# -----------------------------------------------------------
# Main query handler (Ollama -> OpenAI fallback)
# -----------------------------------------------------------
def ollama_prompt(prompt: str, model: Optional[str] = None, timeout: int = 600) -> str:
    model = model or OLLAMA_MODEL

    ok, resp = _ollama_raw_request(prompt, model, timeout)
    if ok:
        return resp

    logging.warning("Ollama failed: %s", resp)

    ok2, resp2 = _openai_fallback(prompt)
    if ok2:
        return resp2

    return (
        "ERROR:\n"
        f"Ollama failed: {resp}\n\n"
        f"OpenAI fallback failed: {resp2}\n\n"
        "Fix:\n"
        "- Ensure `ollama serve` is running\n"
        f"- Ensure model `{model}` is pulled for Ollama (e.g. `ollama pull {model}`)\n"
        "- Or set OPENAI_API_KEY for fallback\n"
    )


# -----------------------------------------------------------
# Frontend-facing function: accept flexible args and kwargs,
# perform retrieval when index+metadata are provided.
# -----------------------------------------------------------
def chatbot_response(user_input: str,
                     index: Optional[Any] = None,
                     metadata: Optional[Any] = None,
                     model: Optional[str] = None,
                     mode: Optional[str] = None,
                     teach_mode: bool = False,
                     top_k: int = 4,
                     **kwargs) -> str:
    """
    Backwards-compatible chatbot_response.

    Accepts extra keyword args used by different frontends (mode, teach_mode, index, metadata).
    If index and metadata are provided (or loadable), perform retrieval and include contexts
    in the prompt before querying the model.
    """
    logging.info("chatbot_response called (mode=%s, teach_mode=%s, model=%s)", mode, teach_mode, model)

    # Prefer provided model, else default
    use_model = model or OLLAMA_MODEL

    # If the frontend didn't provide index/metadata, attempt to load stored ones
    idx = index
    meta = metadata
    if idx is None or meta is None:
        try:
            loaded_idx, loaded_meta = load_index_and_metadata()
            if idx is None:
                idx = loaded_idx
            if meta is None:
                meta = loaded_meta
        except Exception:
            logging.debug("load_index_and_metadata failed or returned nothing.")

    # Attempt retrieval if we have index + metadata
    contexts = []
    if idx is not None and meta is not None:
        try:
            contexts = retrieve_context(idx, meta, user_input, top_k=top_k)
            logging.info("Retrieved %d contexts.", len(contexts))
        except Exception:
            logging.exception("Retrieval failed, continuing without context.")

    # Build prompt
    prompt = user_input
    if teach_mode:
        prompt = f"Teach mode enabled. Provide a clear, step-by-step explanation for: {user_input}"

    if contexts:
        # Prepend contexts to the prompt in a helpful format
        context_text = "\n\n".join(contexts)
        prompt = f"Use the following context to answer. If irrelevant, say so.\n\n{context_text}\n\nQuestion: {prompt}"

    # Call the main prompt function
    return ollama_prompt(prompt, model=use_model)


# Expose compatibility exports
__all__ = [
    "load_index_and_metadata",
    "build_index_from_pdfs",
    "chatbot_response",
    "ollama_prompt",
    "retrieve_context",
    "normalize_metadata",
]
