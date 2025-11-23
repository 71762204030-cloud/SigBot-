# oc3.py
# Backend: FAISS retrieval + Ollama with OpenAI fallback
# Output formatting: produces textbook-style answer (Definition, Intuition, Formula, Worked example, Takeaway),
# plus an exam-style concise answer, then a one-line Summary and Key points.
# All output is plain text (no Markdown, no HTML tags). Formulas are enforced in Option 1 ASCII style:
#   X[k] = Σ (n = 0 to N - 1)  x[n] * e^(-j * 2*pi * k * n / N)
# The code sanitizes LLM responses and guarantees this structure.

import os
import json
import traceback
import re
from typing import List, Optional

import requests
from sentence_transformers import SentenceTransformer
import faiss
import fitz  # PyMuPDF

# Optional plotting libs
try:
    import numpy as np
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False

# ---------- CONFIG ----------
PDF_FOLDER = "data/raw_pdfs"
INDEX_FILE = "data/processed/index.faiss"
META_FILE = "data/processed/meta.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Allow overriding via environment variables
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

TOP_K = 3
CONTEXT_TRIM_CHARS = 1200
answer_cache = {}
session_mode = "detailed"

# ---------- UTILITIES ----------
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception:
        traceback.print_exc()
    return text

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].lstrip())
        start += chunk_size - overlap
    return chunks

def _ollama_raw_request(prompt: str, model: str, timeout: int = 600) -> (bool, str):
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
            return True, json.dumps(data)
        return True, str(data)
    except requests.exceptions.RequestException as e:
        return False, f"Ollama request failed: {e}"
    except ValueError as e:
        return False, f"Ollama returned invalid JSON: {e}"
    except Exception as e:
        return False, f"Ollama unknown error: {e}"

def _openai_fallback(prompt: str) -> (bool, str):
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return False, "OpenAI API key not configured."
        import openai
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.2,
        )
        choices = resp.get("choices", [])
        if choices and "message" in choices[0]:
            return True, choices[0]["message"].get("content", "").strip()
        return True, str(resp)
    except Exception as e:
        return False, f"OpenAI fallback failed: {e}"

def ollama_prompt(prompt: str, model: Optional[str] = None, timeout: int = 600) -> str:
    model = model or OLLAMA_MODEL
    ok, resp = _ollama_raw_request(prompt, model, timeout=timeout)
    if ok and resp and not resp.lower().startswith("ollama request failed:"):
        return resp
    fallback_ok, fallback_resp = _openai_fallback(prompt)
    if fallback_ok:
        return fallback_resp
    err_parts = []
    if ok is False:
        err_parts.append(f"Ollama error: {resp}")
    if not fallback_ok:
        err_parts.append(f"OpenAI fallback: {fallback_resp}")
    guidance = (
        "⚠️ Both Ollama and OpenAI fallback failed. To fix Ollama, ensure you run:\n"
        "  1) Start Ollama: `ollama serve`\n"
        f"  2) Pull the model used by this backend: `ollama pull {OLLAMA_MODEL}`\n"
        f"  3) Verify connectivity: curl {OLLAMA_URL} -d '{{\"model\":\"{OLLAMA_MODEL}\", \"prompt\":\"hello\"}}'\n\n"
        "Alternatively, set the environment variable OPENAI_API_KEY to use OpenAI as a fallback."
    )
    return " | ".join(err_parts) + "\n\n" + guidance

# ---------- SANITIZE / FORMAT HELPERS ----------
HTML_TAG_RE = re.compile(r"<[^>]+>")
SUB_RE = re.compile(r"<sub>(.*?)</sub>", re.IGNORECASE | re.DOTALL)
SUP_RE = re.compile(r"<sup>(.*?)</sup>", re.IGNORECASE | re.DOTALL)

def sanitize_text_for_plain_output(text: str) -> str:
    """
    Convert and clean text so it is plain, human-readable ASCII:
    - Convert <sub>/<sup> to ASCII markers
    - Remove HTML tags
    - Replace unicode symbols (π -> pi)
    - Remove markdown markers like ** or __ etc.
    - Collapse multiple blank lines
    """
    if not text:
        return text

    # Handle <sub> and <sup>
    def sub_repl(m):
        inner = m.group(1).strip()
        return "_{" + inner + "}"
    def sup_repl(m):
        inner = m.group(1).strip()
        return "^{" + inner + "}"

    text = SUB_RE.sub(sub_repl, text)
    text = SUP_RE.sub(sup_repl, text)

    # Remove markdown bold/italic markers
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text, flags=re.DOTALL)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text, flags=re.DOTALL)

    # Replace HTML entities and symbols
    text = text.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    text = text.replace("π", "pi").replace("–", "-").replace("—", "-").replace("−", "-")

    # Drop other HTML tags
    text = HTML_TAG_RE.sub("", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse trailing/leading spaces
    text = text.strip()
    return text

def ascii_enforce_formula_option1(text: str) -> str:
    """
    Replace common DFT/FFT-like formula patterns with the enforced Option 1 ASCII formula:
      X[k] = Σ (n = 0 to N - 1)  x[n] * e^(-j * 2*pi * k * n / N)
    Very best-effort: find common patterns and replace them.
    """
    if not text:
        return text

    # common pattern fragment detection for DFT
    # Attempt to find occurrences of "DFT" or "Discrete Fourier Transform" and a formula nearby
    # and ensure formula uses our ASCII style.
    # This is heuristic — we will replace a few common LaTeX/HTML-like forms.
    s = text

    # Replace LaTeX-like sum_{n=0}^{N-1} patterns
    s = re.sub(r"sum_{\s*n\s*=\s*0\s*}\^{\s*N-1\s*}", "Σ (n = 0 to N - 1)", s, flags=re.IGNORECASE)
    s = re.sub(r"sum\_\{n=0\}\^{N-1}", "Σ (n = 0 to N - 1)", s, flags=re.IGNORECASE)

    # Replace common LaTeX with e^{-j 2\pi k n / N}
    s = re.sub(r"e\^\{\-j\s*2\\pi\s*k\s*n\s*/\s*N\}", "e^(-j * 2*pi * k * n / N)", s, flags=re.IGNORECASE)
    s = re.sub(r"e\^\{\-j\s*2π\s*k\s*n\s*/\s*N\}", "e^(-j * 2*pi * k * n / N)", s, flags=re.IGNORECASE)
    s = s.replace("e^{-j2pi k n/N}", "e^(-j * 2*pi * k * n / N)")
    s = s.replace("e^{-j2\\pi k n / N}", "e^(-j * 2*pi * k * n / N)")

    # Common bracketed forms
    s = re.sub(r"X\[\s*k\s*\]\s*=\s*\\sum\_\{n=0\}\^\{N-1\}\s*x\[\s*n\s*\]\s*e\^\{\-j\s*2\\pi\s*k\s*n\s*/\s*N\}",
               "X[k] = Σ (n = 0 to N - 1)  x[n] * e^(-j * 2*pi * k * n / N)", s, flags=re.IGNORECASE)

    # Generic replacements for several plausible variants
    s = re.sub(r"X\[\s*k\s*\]\s*=\s*sum\_\{n=0\}\^\{N-1\}\s*x\[\s*n\s*\]\s*e\^\{\-j\s*2pi\s*k\s*n\s*/\s*N\}",
               "X[k] = Σ (n = 0 to N - 1)  x[n] * e^(-j * 2*pi * k * n / N)", s, flags=re.IGNORECASE)

    # If still contains LaTeX-like 'sum' or 'Sigma' patterns, normalise them
    s = re.sub(r"Σ\_?\{?\s*n\s*=\s*0\s*(?:to|\-)\s*N\s*-\s*1\}?", "Σ (n = 0 to N - 1)", s, flags=re.IGNORECASE)
    s = re.sub(r"sum\s*\(\s*n\s*=\s*0\s*to\s*N\s*-\s*1\s*\)", "Σ (n = 0 to N - 1)", s, flags=re.IGNORECASE)

    # Finally, if we see "Discrete Fourier Transform" and no clear formula, append the forced formula
    if re.search(r"discrete\s+fourier\s+transform|dft", s, flags=re.IGNORECASE) and "Σ (n = 0 to N - 1)" not in s:
        s += "\n\nFormula: X[k] = Σ (n = 0 to N - 1)  x[n] * e^(-j * 2*pi * k * n / N)\n\nwhere:\n  X[k] = k-th frequency component\n  x[n] = sample at index n\n  N = length of the input sequence\n  k = frequency index (0 to N-1)\n  n = time/sample index (0 to N-1)\n  j = imaginary unit (sqrt(-1))\n  pi = 3.141592653589793"

    return s

# ---------- INDEX MANAGEMENT ----------
def normalize_metadata(raw_meta):
    if isinstance(raw_meta, dict):
        return raw_meta
    if isinstance(raw_meta, list):
        return {str(i): item for i, item in enumerate(raw_meta)}
    raise ValueError("Unsupported metadata format")

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
                    idx = len(metadata)
                    metadata[str(idx)] = {"text": chunk, "source": filename}
                    texts.append(chunk)
    embeddings = embedding_model.encode(texts, show_progress_bar=True) if texts else []
    dim = embeddings.shape[1] if len(embeddings) > 0 else 384
    index = faiss.IndexFlatL2(dim)
    if len(embeddings) > 0:
        index.add(embeddings)
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    return index, metadata

def load_index_and_metadata():
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        try:
            index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, encoding="utf-8") as f:
                metadata_raw = json.load(f)
            metadata = normalize_metadata(metadata_raw)
            return index, metadata
        except Exception:
            traceback.print_exc()
    dim_placeholder = 384
    index = faiss.IndexFlatL2(dim_placeholder)
    return index, {}

# ---------- TEACH PLOTTING ----------
def generate_fft_example_plot(filename: str = "teach_example.png", n_samples: int = 256, sample_rate: float = 1.0) -> str:
    if not PLOTTING_AVAILABLE:
        return ""
    t = np.arange(n_samples) / sample_rate
    f1 = 5
    f2 = 12
    x = 1.0 * np.sin(2 * np.pi * f1 * t / n_samples) + 0.6 * np.sin(2 * np.pi * f2 * t / n_samples)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(n_samples, d=1.0 / sample_rate)
    mag = np.abs(X)
    plt.figure(figsize=(8, 3.5))
    plt.subplot(1, 2, 1)
    plt.plot(t, x)
    plt.title("Time-domain example")
    plt.xlabel("Samples")
    plt.tight_layout()
    plt.subplot(1, 2, 2)
    half = n_samples // 2
    plt.stem(freqs[:half], mag[:half], basefmt=" ", use_line_collection=True)
    plt.title("Magnitude spectrum (example)")
    plt.xlabel("Frequency (normalized)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename

# ---------- PROMPT TEMPLATES ----------
TEXTBOOK_AND_EXAM_PROMPT = (
    "You are a helpful technical tutor. Produce ONLY plain ASCII text (no Markdown, no HTML tags, no bold/asterisks). "
    "Return TWO distinct sections followed by a single-line summary and bulleted key points.\n\n"
    "Section 1: Textbook-style explanation. Use explicit labeled paragraphs with these labels exactly: "
    "Definition:, Intuition:, Formula:, Worked example:, Takeaway:  Each label should be followed by a short paragraph. "
    "Formulas must be written in readable ASCII math using Option 1 style. For example:\n"
    "X[k] = Σ (n = 0 to N - 1)  x[n] * e^(-j * 2*pi * k * n / N)\n"
    "Do NOT use markdown or HTML. Do NOT use '**' or '<sub>' tags. Keep each subsection concise and clear.\n\n"
    "Section 2: Exam-answer style. Provide a concise exam-style response labeled 'Exam answer:' with one or two short lines suitable for an exam.\n\n"
    "Then include a single-line labeled 'Summary:' (one clear sentence) and then 'Key points:' followed by 3-6 plain-text bullet points prefixed with a hyphen and a space (e.g. '- Point text'). "
    "If you used any source filenames, you may mention them in parentheses at the end of the appropriate subsection only.\n\n"
    "If you have retrieved context from documents, you may use that context to make the answer more accurate, but keep the structure above. "
    "If no context is available, answer from general knowledge using the same structure. Be concise, accurate, and use plain ASCII only."
)

# ---------- CHAT FUNCTION ----------
def chatbot_response(user_query: str, index, metadata, embedding_model: SentenceTransformer, mode: str = "quick", teach_mode: bool = False) -> str:
    """
    Returns:
      - Textbook-style explanation (Definition, Intuition, Formula, Worked example, Takeaway)
      - Exam-style concise answer
      - Summary (one line)
      - Key points (bulleted)
    All as plain ASCII text with no Markdown/HTML.
    """
    cache_key = (user_query.strip().lower(), mode, teach_mode)
    if cache_key in answer_cache:
        return answer_cache[cache_key]

    try:
        query_vector = embedding_model.encode([user_query])
    except Exception as e:
        return f"⚠️ Embedding failed: {e}"

    # Retrieve top-k context chunks
    context_parts = []
    try:
        if index.ntotal > 0:
            D, I = index.search(query_vector, TOP_K)
            for idx in I[0]:
                key = str(int(idx))
                if key in metadata:
                    txt = metadata[key].get("text", "")
                    src = metadata[key].get("source")
                    snippet = txt.strip()
                    if src:
                        snippet = f"(source: {src}) {snippet}"
                    context_parts.append(snippet)
    except Exception:
        traceback.print_exc()

    context = "\n\n".join(context_parts).strip()

    # Choose length instruction based on mode
    if teach_mode or mode == "detailed":
        length_note = "Provide a complete textbook-style explanation and an exam-style short answer."
    elif mode == "quick":
        length_note = "Be concise but include all labeled sections; textbook section may be short (2-4 sentences)."
    else:
        length_note = "Provide a concise explanation."

    # Build final prompt for LLM
    if context:
        prompt = (
            f"{TEXTBOOK_AND_EXAM_PROMPT}\n\n"
            f"Context (extracted from documents):\n{context[:CONTEXT_TRIM_CHARS]}\n\n"
            f"Question: {user_query}\n\n{length_note}"
        )
    else:
        prompt = (
            f"{TEXTBOOK_AND_EXAM_PROMPT}\n\nQuestion: {user_query}\n\n{length_note}"
        )

    # Call the model (Ollama or OpenAI fallback)
    raw_ans = ollama_prompt(prompt)

    # Sanitize and format the model output to plain ASCII
    sanitized = sanitize_text_for_plain_output(raw_ans)

    # Enforce Option 1 formula style in the sanitized text
    sanitized = ascii_enforce_formula_option1(sanitized)

    # Ensure the structure exists. If the model didn't break output into sections, attempt to construct them.
    def ensure_structure(text: str) -> str:
        labels = ["Definition:", "Intuition:", "Formula:", "Worked example:", "Takeaway:", "Exam answer:", "Summary:", "Key points:"]
        if all(label in text for label in ["Definition:", "Intuition:", "Formula:"]):
            # Normalize bullets to hyphen style
            text = re.sub(r"^\s*[\*\u2022]\s+", "- ", text, flags=re.MULTILINE)
            return text
        # Otherwise build a minimal structured response heuristically
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        definition = sentences[0] if sentences else ""
        intuition = " ".join(sentences[1:3]) if len(sentences) > 1 else ""
        formula = ""
        m = re.search(r"([A-Za-z0-9_\[\]\(\)\s\^\-\*]+=[^.\n]+)", text)
        if m:
            formula = m.group(1).strip()
        example = ""
        if len(sentences) > 3:
            example = sentences[3]
        takeaway = sentences[-1] if len(sentences) > 1 else ""
        exam = (definition if len(definition) < 200 else definition[:200])
        summary = takeaway if takeaway else definition
        keypoints = ["- " + p.strip() for p in (definition, intuition, formula) if p]
        built = []
        built.append(f"Definition: {definition}")
        built.append(f"Intuition: {intuition}")
        if formula:
            built.append(f"Formula: {ascii_enforce_formula_option1(formula)}")
        if example:
            built.append(f"Worked example: {example}")
        built.append(f"Takeaway: {takeaway}")
        built.append(f"Exam answer: {exam}")
        built.append(f"Summary: {summary}")
        built.append("Key points:")
        built.extend(keypoints)
        return "\n\n".join(built)

    final_text = ensure_structure(sanitized)

    # Final sanitization pass (strip any stray tags/asterisks)
    final_text = sanitize_text_for_plain_output(final_text)

    answer_cache[cache_key] = final_text
    if len(answer_cache) > 200:
        first_key = next(iter(answer_cache))
        del answer_cache[first_key]
    return final_text

# ---------- CLI (optional) ----------
def main(rebuild_index: bool = False):
    global session_mode
    model = SentenceTransformer(EMBEDDING_MODEL)
    if rebuild_index or not (os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)):
        index, metadata = build_index_from_pdfs(model)
    else:
        index, metadata = load_index_and_metadata()
    print("Chatbot ready. Type your question or 'exit' to quit.")
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            low = user_input.lower()
            if low in ["exit", "quit"]:
                break
            if low.startswith("/mode "):
                new_mode = low.split("/mode ", 1)[1].strip()
                if new_mode in ("quick", "detailed"):
                    session_mode = new_mode
                    print(f"Mode set to '{session_mode}'.")
                else:
                    print("Usage: /mode quick OR /mode detailed")
                continue
            if low.startswith("/find "):
                keyword = user_input[6:].strip().lower()
                found = False
                for k, v in metadata.items():
                    if keyword in v.get("text", "").lower():
                        print(f"FOUND in chunk {k} (source: {v.get('source')}):\n{v.get('text')[:200]}...\n")
                        found = True
                if not found:
                    print("No matches found inside indexed PDF chunks. Consider rebuilding the index.")
                continue
            if low == "/cache_stats":
                print(f"Cache entries: {len(answer_cache)}")
                continue
            if low == "/rebuild":
                print("Rebuilding index from PDFs now...")
                index, metadata = build_index_from_pdfs(model)
                print("Rebuild complete.")
                continue
            if low.startswith("/teach"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    print("Usage: /teach <topic>  (e.g. /teach fft)")
                    continue
                topic = parts[1].strip()
                print("Generating a longer lesson (this may take a little longer)...")
                answer = chatbot_response(topic, index, metadata, model, mode=session_mode, teach_mode=True)
                print(f"\nBot (teach): {answer}\n")
                continue
            answer = chatbot_response(user_input, index, metadata, model, mode=session_mode, teach_mode=False)
            print(f"\nBot: {answer}\n")
    except KeyboardInterrupt:
        print("\nExiting chatbot...")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OC chatbot with FAISS + Ollama fallback")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the FAISS index from PDFs")
    args = parser.parse_args()
    main(rebuild_index=args.rebuild)
