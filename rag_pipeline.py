import os
from dotenv import load_dotenv
load_dotenv()

import openai
import pandas as pd
from langchain_core.documents import Document

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_DB_PATH = "vector_store"

MODEL_PRICING_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}

def get_session_vector_db_path(session_id):
    """Return session-specific vector store path."""
    return os.path.join("session_data", session_id, "vector_store")


def _to_maybe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def estimate_cost_usd(model, usage):
    pricing = MODEL_PRICING_PER_1M.get(model)
    if not pricing:
        return None

    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0

    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def build_system_prompt(safe_mode=True, custom_system_prompt=""):
    base_prompt = "You are a helpful AI assistant."
    if safe_mode:
        base_prompt = (
            "Answer ONLY using the provided context. "
            "If the answer is not in context, say 'I don't know.' "
            "Include citations like [1], [2] when relevant."
        )

    if custom_system_prompt and custom_system_prompt.strip():
        return f"{base_prompt}\n\nAdditional instruction:\n{custom_system_prompt.strip()}"
    return base_prompt


def format_sources(retrieved_docs):
    sources = []
    for i, doc in enumerate(retrieved_docs):
        source_path = doc.metadata.get("source", "Unknown source")
        source_name = os.path.basename(source_path)
        snippet = doc.page_content[:180].replace("\n", " ")
        sources.append(
            {
                "index": i + 1,
                "source": source_name,
                "source_path": source_path,
                "preview": snippet,
                "text": f"[{i+1}] {source_name}: {snippet}...",
            }
        )
    return sources


# =========================
# 1. Build Vector DB
# =========================
def build_vector_db(file_paths, api_key, vector_db_path):
    """Accept a list of file paths and merge them into one vector DB."""
    print("📄 Loading documents...")
    all_documents = []

    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            all_documents.extend(loader.load())
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            text = df.to_string()
            doc = Document(page_content=text, metadata={"source": file_path})
            all_documents.append(doc)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
            text = df.to_string()
            doc = Document(page_content=text, metadata={"source": file_path})
            all_documents.append(doc)
        else:
            loader = TextLoader(file_path)
            all_documents.extend(loader.load())

    if not all_documents:
        print("⚠️  No documents loaded.")
        return

    print("✂️ Splitting...")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(all_documents)

    print("🧠 Embedding...")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    print("📦 Building DB...")
    db = FAISS.from_documents(docs, embeddings)

    db.save_local(vector_db_path)
    print("✅ DB saved")


# =========================
# 2. Load DB
# =========================
def load_vector_db(api_key, vector_db_path):
    if not os.path.exists(vector_db_path):
        raise ValueError("Vector DB not found for this session.")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    db = FAISS.load_local(
        vector_db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db


# =========================
# 3. Ask Question (Streaming)
# =========================
def stream_answer(
    db,
    query,
    api_key,
    result_holder,
    safe_mode=True,
    custom_system_prompt="",
    model="gpt-4o-mini",
    temperature=0.2,
    k=3,
):
    retrieved_docs = db.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    system_prompt = build_system_prompt(safe_mode=safe_mode, custom_system_prompt=custom_system_prompt)
    sources = format_sources(retrieved_docs)

    client = openai.OpenAI(api_key=api_key)
    stream = client.chat.completions.create(
        model=model,
        temperature=_to_maybe_float(temperature),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"},
        ],
        stream=True,
        stream_options={"include_usage": True},
    )

    answer_parts = []
    usage_data = {}

    for chunk in stream:
        if getattr(chunk, "usage", None):
            usage_data = {
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens,
                "total_tokens": chunk.usage.total_tokens,
            }

        if chunk.choices and chunk.choices[0].delta:
            piece = chunk.choices[0].delta.content
            if piece:
                answer_parts.append(piece)
                yield piece

    final_answer = "".join(answer_parts).strip() or "I don't know."
    result_holder["answer"] = final_answer
    result_holder["sources"] = sources
    result_holder["usage"] = usage_data
    result_holder["model"] = model
    result_holder["temperature"] = _to_maybe_float(temperature)
    result_holder["estimated_cost_usd"] = estimate_cost_usd(model, usage_data) if usage_data else None


def generate_followup_suggestions(api_key, query, answer, model="gpt-4o-mini"):
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate exactly 3 concise follow-up questions based on the user's question and the assistant answer. "
                    "Return each question on a new line and do not number them."
                ),
            },
            {
                "role": "user",
                "content": f"User question: {query}\nAssistant answer: {answer}",
            },
        ],
    )

    content = (response.choices[0].message.content or "").strip()
    lines = [line.strip("-• ").strip() for line in content.splitlines() if line.strip()]
    return lines[:3]