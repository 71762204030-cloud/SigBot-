# /mnt/data/front_with_history.py
# Sidebar-style login UI (big logo + Login box) + Chats/Controls on left, conversation on right.
# - Logo is embedded from candidate paths (data URI) so it always shows
# - Sidebar contains big logo, "Login" header, helper text, email input, Sign in / Logout
# - No history/save buttons shown
# - Main area holds Chats + Controls (left column) and Conversation (right column)
# - Uses st.rerun() (Streamlit >= stable)
import base64
import streamlit as st
import os
import importlib.util
import time
from datetime import datetime
import html
import re
import json

# ---------------- CONFIG ----------------
LOGO_CANDIDATES = [
    "/mnt/data/cit_logo.png",                       # preferred app working dir
    r"D:\my_chatbot\sigbot\data\cit_logo.png",      # your Windows path (fallback)
]
BACKEND_FILE = "oc3.py"
ALLOWED_DOMAIN = "@cit.edu.in"
HISTORY_DIR = "data/history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# ---------------- UTIL: load image -> data URI ----------------
def file_to_data_uri(path: str) -> str:
    try:
        with open(path, "rb") as f:
            b = f.read()
        encoded = base64.b64encode(b).decode("ascii")
        mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return ""

def pick_logo(candidates):
    for p in candidates:
        if p and os.path.exists(p) and os.path.getsize(p) > 0:
            d = file_to_data_uri(p)
            if d:
                return d
    # fallback SVG
    return "data:image/svg+xml;utf8," + (
        "%3Csvg xmlns='http://www.w3.org/2000/svg' width='128' height='128'%3E"
        "%3Crect rx='16' width='100%25' height='100%25' fill='%230ea5a4'/%3E"
        "%3Ctext x='50%25' y='55%25' font-size='48' text-anchor='middle' fill='white' font-family='Arial' font-weight='bold'%3EC%3C/text%3E"
        "%3C/svg%3E"
    )

logo_data_uri = pick_logo(LOGO_CANDIDATES)

# ---------------- Page config ----------------
st.set_page_config(page_title="SIGBOT Chat", layout="wide", page_icon=logo_data_uri)

# ---------------- Backend import (optional) ----------------
backend_module = None
backend_load_error = None
if os.path.exists(BACKEND_FILE):
    try:
        spec = importlib.util.spec_from_file_location("backend_module", BACKEND_FILE)
        backend_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backend_module)
    except Exception as e:
        backend_module = None
        backend_load_error = str(e)
else:
    backend_load_error = f"Backend file not found at {BACKEND_FILE}"

# ---------------- History helpers (silent auto-save) ----------------
def history_path_for_email(email: str) -> str:
    safe = email.replace("@", "_at_").replace(".", "_")
    return os.path.join(HISTORY_DIR, f"{safe}.json")

def load_history(email: str):
    if not email:
        return []
    path = history_path_for_email(email)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []

def save_history(email: str, chats):
    if not email:
        return False
    path = history_path_for_email(email)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chats, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

# ---------------- Session defaults ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "chats" not in st.session_state:
    st.session_state.chats = []
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None
if "mode" not in st.session_state:
    st.session_state.mode = "quick"
if "backend_load_error" not in st.session_state:
    st.session_state.backend_load_error = backend_load_error

# ---------------- Sanitization ----------------
ALLOWED_TAGS = ["b", "i", "br", "code", "pre"]
def sanitize_backend_html(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    pattern = re.compile(r"</?(?!(" + "|".join(ALLOWED_TAGS) + r")\b)[^>]*>", flags=re.IGNORECASE)
    cleaned = pattern.sub("", text)
    return cleaned

# ---------------- Render helpers ----------------
def render_message_html_string(msg_html: str, sender: str = "bot"):
    allowed_tags = ALLOWED_TAGS
    escaped = html.escape(msg_html or "")
    for tag in allowed_tags:
        escaped = escaped.replace(f"&lt;{tag}&gt;", f"<{tag}>").replace(f"&lt;/{tag}&gt;", f"</{tag}>")
    escaped = escaped.replace("\n", "<br>")
    if sender == "bot":
        return (
            f'<div style="display:flex;gap:12px;align-items:flex-start;margin:8px 0;">'
            f'<div style="background:#0b2942;color:#dbeafe;padding:12px;border-radius:12px;max-width:85%;line-height:1.5;">'
            f'{escaped}</div></div>'
        )
    else:
        return (
            f'<div style="display:flex;gap:12px;align-items:flex-start;margin:8px 0;justify-content:flex-end;">'
            f'<div style="background:#0f172a;color:#e6fffa;padding:12px;border-radius:12px;max-width:85%;line-height:1.5;">{escaped}</div>'
            f'<div style="width:40px;flex-shrink:0;"><div style="width:40px;height:40px;border-radius:6px;background:#0ea5a4;display:flex;align-items:center;justify-content:center;color:#002b36;font-weight:bold;">U</div></div>'
            f'</div>'
        )

def email_domain_allowed(email: str) -> bool:
    return bool(email and email.strip().lower().endswith(ALLOWED_DOMAIN.lower()))

# ---------------- Page CSS ----------------
st.markdown(
    """
    <style>
    /* Sidebar login style to match screenshot */
    .sidebar-logo { display:flex; justify-content:center; padding-top:18px; padding-bottom:6px; }
    .sidebar-login { padding-left:12px; padding-right:12px; }
    .sidebar-login h3 { margin-top: 6px; margin-bottom: 4px; }
    .sidebar-helper { color:#9aa6b2; font-size:13px; margin-bottom:10px; }
    .sidebar-email { margin-bottom:10px; }
    .sidebar-buttons { display:flex; gap:8px; flex-direction:column; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR (big logo + Login UI) ----------------
with st.sidebar:
    # big logo centered
    st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
    # prefer physical file if exists for crisp rendering in sidebar
    shown = False
    for path in LOGO_CANDIDATES:
        if path and os.path.exists(path):
            try:
                st.image(path, width=110)
                shown = True
                break
            except Exception:
                pass
    if not shown:
        st.image(logo_data_uri, width=110)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-login">', unsafe_allow_html=True)
    st.markdown("<h3>Login</h3>", unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-helper">Only email addresses ending with <b>{ALLOWED_DOMAIN}</b> are allowed.</div>', unsafe_allow_html=True)

    # email input
    email_input = st.text_input("Email", value=st.session_state.user_email, placeholder=f"yourname{ALLOWED_DOMAIN}", key="sidebar_email")

    # sign in / logout
    st.markdown('<div class="sidebar-buttons">', unsafe_allow_html=True)
    if st.session_state.logged_in:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.session_state.chats = []
            st.session_state.active_chat = None
            st.rerun()
    else:
        if st.button("Sign in"):
            if email_input and email_input.strip().lower().endswith(ALLOWED_DOMAIN):
                st.session_state.logged_in = True
                st.session_state.user_email = email_input.strip()
                # load or create chat
                loaded = load_history(st.session_state.user_email) or []
                if loaded:
                    st.session_state.chats = loaded
                    st.session_state.active_chat = st.session_state.chats[0]["id"]
                else:
                    new_id = int(time.time() * 1000)
                    chat = {"id": new_id, "title": "New Chat", "messages": [], "created": datetime.utcnow().isoformat()}
                    st.session_state.chats = [chat]
                    st.session_state.active_chat = new_id
                st.success(f"Signed in as {st.session_state.user_email}")
                st.rerun()
            else:
                st.error(f"Email must end with {ALLOWED_DOMAIN}.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- require login before main UI ----------------
if not st.session_state.logged_in:
    st.stop()

# ---------------- Main layout: left (chats+controls) and right (conversation) ----------------
left_col, right_col = st.columns([0.32, 0.68], gap="large")

with left_col:
    st.markdown("### Chats")
    if st.button("➕ New chat"):
        new_id = int(time.time() * 1000)
        chat = {"id": new_id, "title": "New Chat", "messages": [], "created": datetime.utcnow().isoformat()}
        st.session_state.chats.insert(0, chat)
        st.session_state.active_chat = new_id
        save_history(st.session_state.user_email, st.session_state.chats)
        st.rerun()

    # list chats
    for chat in st.session_state.chats:
        cid = chat["id"]
        title = chat.get("title", "Chat")
        if st.button(title, key=f"open_{cid}"):
            st.session_state.active_chat = cid
            st.rerun()

    st.markdown("---")
    st.markdown("### Controls")
    st.radio("Mode", options=["detailed", "quick"], index=0 if st.session_state.mode == "detailed" else 1, key="mode")
    if st.button("Clear all chats (UI)"):
        st.session_state.chats = []
        st.session_state.active_chat = None
        save_history(st.session_state.user_email, st.session_state.chats)
        st.rerun()

with right_col:
    if st.session_state.active_chat is None:
        st.markdown("## Start a chat")
        st.stop()

    active_chat = next((c for c in st.session_state.chats if c["id"] == st.session_state.active_chat), None)
    if active_chat is None:
        st.session_state.active_chat = None
        st.rerun()

    # Chat title
    title = st.text_input("Chat title", value=active_chat.get("title", "Chat"), key=f"title_{active_chat['id']}")
    active_chat["title"] = title

    # Build messages (sanitized)
    messages_html = ""
    for msg in active_chat.get("messages", []):
        role = msg.get("role", "bot")
        raw = msg.get("text", "")
        safe = sanitize_backend_html(raw)
        messages_html += render_message_html_string(safe, sender="bot" if role == "bot" else "user")
        src = msg.get("source")
        if src:
            messages_html += f"<div style='color:#8aa0b3;font-size:12px;margin-left:8px;margin-bottom:8px;'>Source: {html.escape(src)}</div>"

    st.markdown(
        f"""
        <div style="height:calc(100vh - 220px); overflow-y:auto; padding:12px; border-radius:6px; background:#061425;">
            {messages_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input below
    with st.form(key="ask_form", clear_on_submit=False):
        user_q = st.text_input("", placeholder="Ask your question (e.g., 'define fft')", key=f"input_{active_chat['id']}")
        cols = st.columns([0.1, 0.1, 0.8])
        ask_btn = cols[0].form_submit_button("Ask")
        teach_btn = cols[1].form_submit_button("Teach (long)")
        if ask_btn or teach_btn:
            if not user_q.strip():
                st.warning("Please type a question.")
            else:
                um = {"role": "user", "text": user_q.strip(), "time": datetime.utcnow().isoformat()}
                active_chat["messages"].append(um)
                st.rerun()

    # Process last user message (call backend, sanitize, append)
    last_msg = active_chat["messages"][-1] if active_chat["messages"] else None
    if last_msg and last_msg["role"] == "user":
        qtext = last_msg["text"]
        teach_mode = False
        if qtext.lower().strip().startswith("teach "):
            teach_mode = True
            q = qtext[len("teach "):].strip()
        else:
            q = qtext

        raw_answer = ""
        if backend_module is None:
            raw_answer = f"⚠ Backend not found or failed to load. Ensure {BACKEND_FILE} is present."
            if st.session_state.backend_load_error:
                raw_answer += f"\n{st.session_state.backend_load_error}"
        else:
            try:
                idx, meta = backend_module.load_index_and_metadata()
                emb_model = None
                if hasattr(backend_module, "SentenceTransformer"):
                    try:
                        emb_model = backend_module.SentenceTransformer(backend_module.EMBEDDING_MODEL)
                    except Exception:
                        emb_model = None
                raw_answer = backend_module.chatbot_response(q, idx, meta, emb_model, mode=st.session_state.mode, teach_mode=teach_mode)
            except Exception as e:
                try:
                    raw_answer = backend_module.chatbot_response(q, backend_module.load_index_and_metadata()[0], backend_module.load_index_and_metadata()[1], None, mode=st.session_state.mode, teach_mode=teach_mode)
                except Exception as ee:
                    raw_answer = f"⚠ Error calling backend: {ee}"

        cleaned_answer = sanitize_backend_html(raw_answer)
        bot_msg = {"role": "bot", "text": cleaned_answer, "time": datetime.utcnow().isoformat()}
        m = re.search(r"\(book:\s*([^\)]+)\)", cleaned_answer or "")
        if m:
            bot_msg["source"] = m.group(1)
        active_chat["messages"].append(bot_msg)

        # silent auto-save
        save_history(st.session_state.user_email, st.session_state.chats)

        # update chat title (if default)
        if active_chat.get("title", "").lower().startswith("new chat") or active_chat.get("title", "") == "":
            t = q.strip()
            active_chat["title"] = (t[:40] + "...") if len(t) > 40 else t

        st.rerun()
