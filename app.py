import streamlit as st
import os
import uuid
import pandas as pd

from utils import save_uploaded_file, get_session_dir
from rag_pipeline import (
    build_vector_db,
    load_vector_db,
    stream_answer,
    generate_followup_suggestions,
    get_session_vector_db_path,
)

st.set_page_config(page_title="ChatDoc AI", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            max-width: 860px;
            padding-top: 1rem;
        }
        .stChatInput {
            max-width: 860px;
            margin: 0 auto;
        }
        .dm-subtle {
            color: #6b7280;
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = []

session_id = st.session_state.session_id
session_dir = get_session_dir(session_id)
vector_db_path = get_session_vector_db_path(session_id)


def format_bytes(size):
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.2f} MB"


def file_preview(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in [".txt", ".csv"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(600)
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path, nrows=5)
            return df.to_string(index=False)
        if ext == ".pdf":
            return "PDF preview is not shown here. It is indexed and searchable in the assistant."
    except Exception:
        return "Preview unavailable."
    return "Preview unavailable."


def collect_document_metadata(paths):
    meta = []
    for path in paths:
        if not os.path.exists(path):
            continue
        size = os.path.getsize(path)
        meta.append(
            {
                "name": os.path.basename(path),
                "path": path,
                "size": format_bytes(size),
                "preview": file_preview(path),
            }
        )
    return meta


def get_last_user_query(history):
    for msg in reversed(history):
        if msg.get("role") == "user":
            return msg.get("content")
    return None


PROMPT_PRESETS = {
    "None (Default)": "",
    "Concise Assistant": "Be concise. Use short paragraphs and avoid unnecessary detail.",
    "Technical Expert": "Answer with technical depth. Include assumptions, tradeoffs, and practical examples.",
    "Beginner Friendly": "Explain in simple language for beginners. Define key terms and use easy examples.",
    "Action-Oriented": "Give a direct answer first, then provide clear step-by-step actions.",
}

# Header
st.markdown("<h1 style='text-align: center; font-size: 1.9rem; margin-bottom: 0.2rem;'>🧠 ChatDoc AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='dm-subtle' style='text-align: center; margin-bottom: 0;'>Upload documents and ask questions with AI</p>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("Session Docs")
    if st.session_state.doc_metadata:
        for i, item in enumerate(st.session_state.doc_metadata):
            with st.expander(item["name"], expanded=False):
                st.caption(f"Size: {item['size']}")
                st.text(item["preview"])
                try:
                    with open(item["path"], "rb") as f:
                        st.download_button(
                            "Download",
                            data=f.read(),
                            file_name=item["name"],
                            key=f"download_{i}",
                        )
                except OSError:
                    st.caption("File not available for download.")
    else:
        st.caption("No uploaded documents yet.")

# =========================
# API Key Input
# =========================
with st.container(border=True):
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        api_key = st.text_input(
            "🔑 OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Your key is never stored. It is only used for this session.",
            key="api_key_input"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.link_button("Get a key ↗", "https://platform.openai.com/api-keys")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()

if not api_key:
    st.warning("⚠️ **On a shared computer?** Remember to click 'Logout' when done. Your key is only stored in your current browser session.")
    st.info("Enter your OpenAI API key above to get started.")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# Model Controls
# =========================
with st.expander("Advanced Settings", expanded=False):
    setting_col1, setting_col2 = st.columns(2)
    with setting_col1:
        model = st.selectbox("Model", options=["gpt-4o-mini", "gpt-4.1-mini"], index=0)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.2, step=0.1)
    with setting_col2:
        top_k = st.slider("Retrieved chunks (k)", min_value=1, max_value=8, value=3, step=1)
        safe_mode = st.toggle("🛡️ Hallucination Safe Mode", value=True)

    preset_name = st.selectbox(
        "Prompt Preset",
        options=list(PROMPT_PRESETS.keys()),
        index=0,
        help="Choose a ready-made style and then customize it if needed.",
    )

    st.caption("Examples:")
    st.caption("- Be concise and use bullet points.")
    st.caption("- Explain like I am new to this topic.")
    st.caption("- Prioritize practical steps over theory.")

    custom_system_prompt = st.text_area(
        "Custom System Prompt",
        value=PROMPT_PRESETS[preset_name],
        placeholder="Example: Be concise and use bullet points.",
        help="Optional: add behavior instructions for the assistant.",
    )

# =========================
# Upload Section
# =========================
uploaded_files = st.file_uploader(
    "Upload files",
    type=["txt", "pdf", "csv", "xlsx", "xls"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.container(border=True):
        up_col1, up_col2 = st.columns([3, 1])
        with up_col1:
            st.success(f"{len(uploaded_files)} file(s) ready for indexing.")
        with up_col2:
            build_now = st.button("Build Knowledge Base", use_container_width=True)

        if build_now:
            file_paths = [save_uploaded_file(f, session_id) for f in uploaded_files]
            with st.spinner("Building knowledge base..."):
                build_vector_db(file_paths, api_key, vector_db_path)
            st.success("Vector DB created!")
            st.session_state.chat_history = []
            st.session_state.doc_metadata = collect_document_metadata(file_paths)

# =========================
# Load DB
# =========================
if os.path.exists(vector_db_path):
    db = load_vector_db(api_key, vector_db_path)

    regenerate_col1, regenerate_col2 = st.columns([2, 3])
    with regenerate_col1:
        if st.button("Regenerate Last Response", disabled=get_last_user_query(st.session_state.chat_history) is None):
            st.session_state.regen_query = get_last_user_query(st.session_state.chat_history)
            st.rerun()
    with regenerate_col2:
        st.caption("Change model/temperature above, then click regenerate.")

    # Display full conversation history
    for msg_i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Sources"):
                    for s in message["sources"]:
                        st.write(s.get("text", ""))
                        if s.get("source_path"):
                            st.caption(f"File: {s['source_path']}")
            if message["role"] == "assistant" and message.get("usage"):
                usage = message["usage"]
                cost = message.get("estimated_cost_usd")
                usage_txt = (
                    f"Tokens - prompt: {usage.get('prompt_tokens', 0)}, "
                    f"completion: {usage.get('completion_tokens', 0)}, "
                    f"total: {usage.get('total_tokens', 0)}"
                )
                if cost is not None:
                    usage_txt += f" | Estimated cost: ${cost:.6f}"
                st.caption(usage_txt)

            if message["role"] == "assistant" and message.get("suggestions"):
                st.caption("Suggested follow-up questions:")
                sug_cols = st.columns(3)
                for j, suggestion in enumerate(message["suggestions"][:3]):
                    with sug_cols[j]:
                        if st.button(suggestion, key=f"sug_{msg_i}_{j}"):
                            st.session_state.pending_query = suggestion
                            st.rerun()

    # Chat input
    is_regen = False
    query = st.session_state.pop("pending_query", None)
    if not query:
        regen_query = st.session_state.pop("regen_query", None)
        if regen_query:
            query = regen_query
            is_regen = True
    if not query:
        query = st.chat_input("Ask a question about your documents...")

    if query:
        if not is_regen:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)
        else:
            st.caption("Regenerating answer for your last question with current model settings...")

        with st.chat_message("assistant"):
            result_holder = {}
            answer = st.write_stream(
                stream_answer(
                    db=db,
                    query=query,
                    api_key=api_key,
                    result_holder=result_holder,
                    safe_mode=safe_mode,
                    custom_system_prompt=custom_system_prompt,
                    model=model,
                    temperature=temperature,
                    k=top_k,
                )
            )

            sources = result_holder.get("sources", [])
            usage = result_holder.get("usage", {})
            estimated_cost = result_holder.get("estimated_cost_usd")

            with st.expander("📚 Sources"):
                for s in sources:
                    st.write(s.get("text", ""))
                    if s.get("source_path"):
                        st.caption(f"File: {s['source_path']}")

            if usage:
                usage_txt = (
                    f"Tokens - prompt: {usage.get('prompt_tokens', 0)}, "
                    f"completion: {usage.get('completion_tokens', 0)}, "
                    f"total: {usage.get('total_tokens', 0)}"
                )
                if estimated_cost is not None:
                    usage_txt += f" | Estimated cost: ${estimated_cost:.6f}"
                st.caption(usage_txt)

            suggestions = generate_followup_suggestions(
                api_key=api_key,
                query=query,
                answer=answer,
                model=model,
            )
            if suggestions:
                st.caption("Suggested follow-up questions:")
                sug_cols = st.columns(3)
                for j, suggestion in enumerate(suggestions[:3]):
                    with sug_cols[j]:
                        if st.button(suggestion, key=f"live_sug_{j}"):
                            st.session_state.pending_query = suggestion
                            st.rerun()

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "usage": result_holder.get("usage", {}),
            "estimated_cost_usd": result_holder.get("estimated_cost_usd"),
            "model": model,
            "temperature": temperature,
            "suggestions": suggestions,
        })