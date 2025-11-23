import os
import json
import requests
import logging
from typing import Optional, List

# -----------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------------------------------------------
# Ollama and OpenAI config
# -----------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# -----------------------------------------------------------
# Improved Ollama Request Handler
# -----------------------------------------------------------
def _ollama_raw_request(prompt: str, model: str, timeout: int = 600) -> (bool, str):
    """
    Improved Ollama caller with logging:
    - Sends JSON with correct headers
    - Logs all responses
    - Returns (ok, text) or (False, error)
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    headers = {"Content-Type": "application/json"}

    try:
        logging.info("Calling Ollama at %s ...", OLLAMA_URL)
        resp = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=timeout)

        logging.info("Ollama status: %s", resp.status_code)

        if resp.status_code >= 400:
            logging.error("Ollama Error %s: %s", resp.status_code, resp.text)
            return False,
            f"Ollama HTTP {resp.status_code}: {resp.text.strip()}"

        # Parse JSON
        try:
            data = resp.json()
            logging.debug("Ollama JSON: %s", data)
        except:
            logging.warning("Ollama returned non-JSON response")
            return True, resp.text.strip()

        # Extract from known keys
        if "response" in data and data["response"]:
            return True, data["response"].strip()

        if "text" in data and data["text"]:
            return True, data["text"].strip()

        if "error" in data:
            logging.error("Ollama error body: %s", data["error"])
            return False, f"Ollama returned error: {data['error']}"

        # choices (rare)
        if "choices" in data:
            c = data["choices"][0]
            if "message" in c and "content" in c["message"]:
                return True, c["message"]["content"].strip()

        return True, json.dumps(data)

    except requests.exceptions.RequestException as e:
        logging.exception("Ollama connection error")
        return False, f"Ollama request failed: {e}"

    except Exception as e:
        logging.exception("Ollama unknown failure")
        return False, f"Ollama unknown error: {e}"


# -----------------------------------------------------------
# Improved OpenAI fallback handler
# -----------------------------------------------------------
def _openai_fallback(prompt: str) -> (bool, str):
    """
    OpenAI fallback with full logging.
    Returns (ok, answer) OR (False, error_message)
    """

    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY missing! Cannot use fallback.")
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
            logging.error("OpenAI error: %s", resp.text)
            return False, f"OpenAI HTTP {resp.status_code}: {resp.text}"

        data = resp.json()

        try:
            answer = data["choices"][0]["message"]["content"]
            return True, answer.strip()
        except:
            logging.warning("Unexpected OpenAI response: %s", data)
            return True, json.dumps(data)

    except Exception as e:
        logging.exception("OpenAI unknown exception")
        return False, f"OpenAI error: {e}"


# -----------------------------------------------------------
# Main function called by frontend
# -----------------------------------------------------------
def ollama_prompt(prompt: str, model: Optional[str] = None, timeout: int = 600) -> str:
    """
    1. Try Ollama
    2. If fails → OpenAI fallback
    3. If both fail → return full detailed error
    """
    model = model or OLLAMA_MODEL

    # ---- Try Ollama ----
    ok, resp = _ollama_raw_request(prompt, model, timeout)
    if ok:
        return resp

    logging.warning("Ollama failed: %s", resp)

    # ---- Try OpenAI fallback ----
    ok2, resp2 = _openai_fallback(prompt)
    if ok2:
        return resp2

    # ---- Both failed ----
    return (
        "ERROR:\n"
        f"Ollama failed: {resp}\n\n"
        f"OpenAI fallback failed: {resp2}\n\n"
        "Fix:\n"
        "- Ensure `ollama serve` is running\n"
        "- Run: ollama pull gemma3:4b\n"
        "- OR set OPENAI_API_KEY for fallback\n"
    )


# -----------------------------------------------------------
# Entry for Streamlit
# -----------------------------------------------------------
def chatbot_response(user_input: str) -> str:
    return ollama_prompt(user_input)
