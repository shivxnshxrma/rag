import streamlit as st
import time
import json
import random
import tempfile
import os
import re
from datetime import datetime
from pathlib import Path
from html import escape

from dotenv import load_dotenv
from langchain_core.documents import Document
import streamlit.components.v1 as components
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind Study",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0d0d0d;
    color: #e8e4dc;
}
#MainMenu, footer, header 
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* Always show sidebar toggle arrow — fixes disappearing collapse button */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background-color: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 0 6px 6px 0 !important;
    color: #c9a96e !important;
}
[data-testid="collapsedControl"]:hover {
    background-color: #c9a96e !important;
    color: #0d0d0d !important;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1;
    background: linear-gradient(135deg, #f0e6d3 0%, #c9a96e 50%, #f0e6d3 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 300;
    color: #6b6560;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #1e1e1e;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
.sidebar-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #c9a96e;
    margin-bottom: 0.8rem;
    margin-top: 1.2rem;
}
[data-testid="stFileUploader"] {
    background: #161616;
    border: 1px dashed #2a2a2a;
    border-radius: 8px;
    padding: 0.5rem;
}
.stButton > button {
    background: transparent;
    border: 1px solid #c9a96e;
    color: #c9a96e;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.5rem 1.2rem;
    border-radius: 3px;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover { background: #c9a96e; color: #0d0d0d; }

.stTabs [data-baseweb="tab-list"] {
    background: #111111;
    border-bottom: 1px solid #1e1e1e;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #444;
    padding: 0.8rem 1.5rem;
    border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: #c9a96e !important;
    border-bottom: 2px solid #c9a96e !important;
    background: transparent !important;
}

.chat-bubble-user {
    background: #161616;
    border: 1px solid #1e1e1e;
    border-left: 3px solid #c9a96e;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
}
.chat-bubble-ai {
    background: #111111;
    border: 1px solid #1e1e1e;
    border-left: 3px solid #4a9eff;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    line-height: 1.7;
}
.chat-label { font-family: 'Syne', sans-serif; font-size: 0.6rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 0.4rem; }
.label-user { color: #c9a96e; }
.label-ai { color: #4a9eff; }

.source-card {
    background: #0a0a0a;
    border: 1px solid #1a1a1a;
    border-radius: 4px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.75rem;
    color: #6b6560;
}
.source-meta { color: #c9a96e; font-size: 0.68rem; letter-spacing: 1px; margin-bottom: 0.3rem; font-family: 'Syne', sans-serif; font-weight: 600; }

.quiz-card {
    background: #111111;
    border: 1px solid #1e1e1e;
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
}
.quiz-q { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 600; color: #e8e4dc; line-height: 1.5; }
.quiz-num { font-size: 0.62rem; letter-spacing: 3px; text-transform: uppercase; color: #c9a96e; margin-bottom: 0.4rem; font-family: 'Syne', sans-serif; }

.correct-ans { background: #0a1f0a; border: 1px solid #1a4a1a; border-radius: 4px; padding: 0.6rem 1rem; color: #4caf50; font-size: 0.82rem; margin-top: 0.5rem; }
.wrong-ans { background: #1f0a0a; border: 1px solid #4a1a1a; border-radius: 4px; padding: 0.6rem 1rem; color: #ef5350; font-size: 0.82rem; margin-top: 0.5rem; }

.flashcard-front {
    background: #161616;
    border: 1px solid #c9a96e;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    min-height: 320px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
    overflow-wrap: anywhere;
}
.flashcard-back {
    background: #0a0f1f;
    border: 1px solid #4a9eff;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    min-height: 320px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
    overflow-wrap: anywhere;
}
.flashcard-term { font-family: 'Syne', sans-serif; font-size: 1.35rem; font-weight: 700; color: #c9a96e; line-height: 1.45; }
.flashcard-def { font-size: 0.9rem; color: #b0c8f0; line-height: 1.75; }
.flashcard-label { font-size: 0.6rem; letter-spacing: 3px; text-transform: uppercase; color: #444; margin-bottom: 0.8rem; font-family: 'Syne', sans-serif; }

.score-box { background: #111111; border: 1px solid #1e1e1e; border-radius: 6px; padding: 1.5rem; text-align: center; margin-bottom: 1rem; }
.score-big { font-family: 'Syne', sans-serif; font-size: 3.5rem; font-weight: 800; color: #c9a96e; line-height: 1; }
.score-label { font-size: 0.65rem; letter-spacing: 3px; text-transform: uppercase; color: #444; margin-top: 0.3rem; }

.summary-box {
    background: #111111;
    border: 1px solid #1e1e1e;
    border-left: 3px solid #c9a96e;
    border-radius: 4px;
    padding: 1.4rem 1.6rem;
    font-size: 0.88rem;
    line-height: 1.9;
    color: #c8c4bc;
    white-space: pre-wrap;
}

.metric-box { background: #111111; border: 1px solid #1e1e1e; border-radius: 4px; padding: 0.8rem 1rem; text-align: center; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; color: #c9a96e; }
.metric-label { font-size: 0.62rem; letter-spacing: 2px; text-transform: uppercase; color: #444; }

.insight-box { background:#0f0f0f; border:1px solid #1f1f1f; border-left:3px solid #4a9eff; border-radius:4px; padding:0.9rem 1rem; margin:0.8rem 0; font-size:0.78rem; color:#9a958d; line-height:1.7; }
.gap-chip { display:inline-block; border:1px solid #3a2a10; color:#c9a96e; border-radius:18px; padding:0.22rem 0.65rem; margin:0.18rem; font-size:0.68rem; }
.map-node { display:inline-block; background:#141414; border:1px solid #292929; border-radius:4px; color:#e8e4dc; padding:0.45rem 0.7rem; margin:0.25rem; font-size:0.74rem; }

.status-badge { display: inline-block; background: #0d1f0d; border: 1px solid #1a3a1a; color: #4caf50; font-size: 0.65rem; letter-spacing: 2px; text-transform: uppercase; padding: 0.2rem 0.7rem; border-radius: 20px; }
.status-badge.inactive { background: #1a1108; border-color: #3a2a10; color: #c9a96e; }

.stChatInput > div { background: #111111 !important; border: 1px solid #2a2a2a !important; border-radius: 4px !important; }
.thin-divider { border: none; border-top: 1px solid #1a1a1a; margin: 1.5rem 0; }
.file-pill { display: inline-block; background: #161616; border: 1px solid #2a2a2a; border-radius: 20px; padding: 0.2rem 0.8rem; font-size: 0.68rem; color: #888; margin: 0.2rem; }

.section-header { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: #e8e4dc; margin-bottom: 0.3rem; }
.section-sub { font-size: 0.72rem; color: #555; letter-spacing: 1px; margin-bottom: 1.5rem; }
.progress-bar-bg { background: #1a1a1a; border-radius: 10px; height: 8px; margin: 0.5rem 0 1rem 0; }
.progress-bar-fill { background: linear-gradient(90deg, #c9a96e, #f0e6d3); border-radius: 10px; height: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────
STATE_FILE = Path(".documind_state.json")

defaults = {
    "chat_history": [],
    "indexed_files": [],
    "total_chunks": 0,
    "vector_store_ready": False,
    "quiz_questions": [],
    "quiz_answers": {},
    "quiz_submitted": False,
    "flashcards": [],
    "fc_index": 0,
    "fc_show_back": False,
    "summary_text": "",
    "summary_topic": "",
    "summary_sources": [],
    "quiz_history": [],
    "weak_topics": {},
    "last_quiz_topic": "",
    "last_quiz_difficulty": "Exam",
    "last_quiz_page_range": "",
    "study_plan": "",
    "concept_map": [],
    "last_missed_topics": {},
}

def load_saved_state():
    if not STATE_FILE.exists():
        return {}
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    allowed = {
        "indexed_files",
        "total_chunks",
        "vector_store_ready",
        "quiz_questions",
        "quiz_answers",
        "quiz_submitted",
        "flashcards",
        "fc_index",
        "fc_show_back",
        "summary_text",
        "summary_topic",
        "summary_sources",
        "quiz_history",
        "weak_topics",
        "last_quiz_topic",
        "last_quiz_difficulty",
        "last_quiz_page_range",
        "study_plan",
        "concept_map",
        "last_missed_topics",
    }
    cleaned = {k: v for k, v in saved.items() if k in allowed}
    if "summary_sources" in cleaned:
        cleaned["summary_sources"] = deserialize_docs(cleaned["summary_sources"])
    return cleaned


def serialize_docs(docs):
    return [
        {"page_content": doc.page_content, "metadata": doc.metadata or {}}
        for doc in docs
    ]


def deserialize_docs(records):
    docs = []
    for item in records or []:
        if isinstance(item, Document):
            docs.append(item)
        elif isinstance(item, dict):
            docs.append(Document(page_content=item.get("page_content", ""), metadata=item.get("metadata", {}) or {}))
    return docs

saved_state = load_saved_state()
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = saved_state.get(k, v)


# ─────────────────────────────────────────────
# Cached Resources
# ─────────────────────────────────────────────
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url="http://localhost:6333")

@st.cache_resource
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your-google-api-key-here":
        st.error("Missing GOOGLE_API_KEY. Add it to .env, then restart Streamlit.")
        st.stop()
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=api_key)


# ─────────────────────────────────────────────
# Core Helpers
# ─────────────────────────────────────────────
COLLECTION_NAME = "documind_rag"

def ensure_collection(client, dim=384):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(COLLECTION_NAME, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

def reset_collection(client, dim=384):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(COLLECTION_NAME, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

def save_progress():
    data = {
        "indexed_files": st.session_state.indexed_files,
        "total_chunks": st.session_state.total_chunks,
        "vector_store_ready": st.session_state.vector_store_ready,
        "quiz_questions": st.session_state.quiz_questions,
        "quiz_answers": st.session_state.quiz_answers,
        "quiz_submitted": st.session_state.quiz_submitted,
        "quiz_history": st.session_state.quiz_history,
        "weak_topics": st.session_state.weak_topics,
        "last_quiz_topic": st.session_state.last_quiz_topic,
        "last_quiz_difficulty": st.session_state.last_quiz_difficulty,
        "last_quiz_page_range": st.session_state.last_quiz_page_range,
        "flashcards": st.session_state.flashcards,
        "fc_index": st.session_state.fc_index,
        "fc_show_back": st.session_state.fc_show_back,
        "summary_text": st.session_state.summary_text,
        "summary_topic": st.session_state.summary_topic,
        "summary_sources": serialize_docs(st.session_state.summary_sources),
        "study_plan": st.session_state.study_plan,
        "concept_map": st.session_state.concept_map,
        "last_missed_topics": st.session_state.last_missed_topics,
    }
    try:
        with STATE_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass

def index_pdfs(uploaded_files):
    client = get_qdrant_client()
    reset_collection(client)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []
    progress = st.sidebar.progress(0, text="Loading PDFs...")
    for i, f in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        docs = PyPDFLoader(tmp_path).load()
        for doc in docs:
            doc.metadata["source_file"] = f.name
        all_chunks.extend(splitter.split_documents(docs))
        os.unlink(tmp_path)
        progress.progress((i + 1) / len(uploaded_files), text=f"Processed: {f.name}")
    progress.progress(0.9, text="Indexing into Qdrant...")
    QdrantVectorStore.from_documents(all_chunks, get_embedding_model(), url="http://localhost:6333", collection_name=COLLECTION_NAME)
    progress.progress(1.0, text="Done!")
    time.sleep(0.5)
    progress.empty()
    return len(all_chunks)

def parse_page_range(raw: str):
    if not raw:
        return None
    match = re.search(r"(\d+)\s*(?:-|to|–|—)\s*(\d+)", raw.lower())
    if match:
        start, end = sorted((int(match.group(1)), int(match.group(2))))
        return start, end
    match = re.search(r"(?:page|pages|p\.)\s*(\d+)", raw.lower())
    if match:
        page = int(match.group(1))
        return page, page
    match = re.fullmatch(r"\s*(\d+)\s*", raw)
    if match:
        page = int(match.group(1))
        return page, page
    return None

def doc_page(doc):
    page = doc.metadata.get("page_label") or doc.metadata.get("page")
    try:
        return int(page)
    except (TypeError, ValueError):
        return None

def in_page_range(doc, page_range):
    if not page_range:
        return True
    page = doc_page(doc)
    if page is None:
        return False
    return page_range[0] <= page <= page_range[1]

def filter_by_page_range(docs, page_range):
    return [doc for doc in docs if in_page_range(doc, page_range)]

def extract_page_range_from_text(text):
    match = re.search(r"\bpages?\s+\d+\s*(?:-|to|–|—)\s*\d+\b|\bp\.\s*\d+\b|\bpage\s+\d+\b", text.lower())
    return parse_page_range(match.group(0)) if match else None

def get_context(query: str, k: int = 6, page_range=None):
    vs = QdrantVectorStore(client=get_qdrant_client(), collection_name=COLLECTION_NAME, embedding=get_embedding_model())
    raw_docs = vs.as_retriever(search_kwargs={"k": max(k * 3, 12)}).invoke(query)
    if page_range:
        raw_docs = filter_by_page_range(raw_docs, page_range)
    docs = unique_docs(raw_docs)
    if page_range and len(docs) < k:
        docs = unique_docs(docs + keyword_docs(query, max_docs=k, page_range=page_range))
    docs = docs[:k]
    return docs, "\n\n---\n\n".join([d.page_content for d in docs])

def unique_docs(docs):
    unique = []
    seen = set()
    for doc in docs:
        key = (
            doc.metadata.get("source_file"),
            doc.metadata.get("page"),
            " ".join(doc.page_content.split())[:240],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique

def keyword_docs(topic: str, max_docs: int = 8, page_range=None):
    terms = [t for t in re.findall(r"[A-Za-z0-9_.-]+", topic.lower()) if len(t) >= 3]
    if not terms:
        return []

    matches = []
    for doc in scroll_indexed_docs():
        if not in_page_range(doc, page_range):
            continue
        text = doc.page_content.lower()
        score = sum(text.count(term) for term in terms)
        if score:
            matches.append((score, doc))

    matches.sort(key=lambda item: (
        -item[0],
        item[1].metadata.get("source_file", ""),
        item[1].metadata.get("page", 0) or 0,
    ))
    return unique_docs([doc for _, doc in matches])[:max_docs]

def get_topic_context(topic: str, fallback_query: str, k: int = 12, page_range=None):
    query = topic.strip() if topic.strip() else fallback_query
    retrieved, _ = get_context(query, k=k, page_range=page_range)
    exact = keyword_docs(topic, max_docs=max(4, k // 2), page_range=page_range) if topic.strip() else []
    docs = unique_docs(exact + retrieved)[:k]
    return docs, "\n\n---\n\n".join([d.page_content for d in docs])

def scroll_indexed_docs(limit: int = 2500, page_range=None):
    client = get_qdrant_client()
    docs = []
    offset = None
    while len(docs) < limit:
        points, offset = client.scroll(
            COLLECTION_NAME,
            limit=min(256, limit - len(docs)),
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for point in points:
            payload = point.payload or {}
            content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            doc = Document(page_content=content, metadata=metadata)
            if content and in_page_range(doc, page_range):
                docs.append(doc)
        if offset is None:
            break
    return docs

def is_front_matter_noise(doc):
    text = doc.page_content.lower()
    page = doc.metadata.get("page", 0) or 0
    noisy_terms = (
        "prerelease",
        "report any bugs",
        "us@fullstack.io",
        "follow @fullstackio",
        "discord channel",
        "testimonials",
        "revision",
    )
    return page <= 5 and any(term in text for term in noisy_terms)

def is_toc_like(text):
    return text.count(". . .") >= 3 or "table of contents" in text.lower() or "contents" in text.lower()

def select_representative_docs(max_docs: int = 18, page_range=None):
    docs = [doc for doc in scroll_indexed_docs(page_range=page_range) if not is_front_matter_noise(doc)]
    docs.sort(key=lambda d: (
        d.metadata.get("source_file", ""),
        d.metadata.get("page", 0) or 0,
        d.page_content[:80],
    ))

    grouped = {}
    for doc in docs:
        grouped.setdefault(doc.metadata.get("source_file", "uploaded document"), []).append(doc)

    selected = []
    seen = set()
    per_file = max(6, max_docs // max(1, len(grouped)))

    for file_docs in grouped.values():
        toc_docs = [d for d in file_docs if (d.metadata.get("page", 0) or 0) <= 8 and is_toc_like(d.page_content)]
        candidates = []
        candidates.extend(toc_docs[:3])
        added_for_file = 0

        useful = [d for d in file_docs if d not in candidates]
        if useful:
            positions = [0.05, 0.18, 0.32, 0.46, 0.60, 0.74, 0.88]
            for pos in positions:
                idx = min(len(useful) - 1, int(len(useful) * pos))
                candidates.append(useful[idx])

        for doc in candidates:
            key = (doc.metadata.get("source_file"), doc.metadata.get("page"), doc.page_content[:120])
            if key not in seen:
                selected.append(doc)
                seen.add(key)
                added_for_file += 1
            if added_for_file >= per_file:
                break

    return selected[:max_docs]

def is_summary_request(text: str):
    lower = text.lower()
    return any(word in lower for word in ("summary", "summarize", "summarise", "overview", "gist", "what is this pdf about", "what is this document about"))

def is_document_summary_request(text: str):
    lower = text.lower().strip()
    doc_words = ("pdf", "document", "file", "book", "notes", "uploaded", "whole", "entire")
    vague_topics = ("summary", "summarize", "summarise", "overview", "gist")
    return any(word in lower for word in doc_words) or lower in vague_topics

def extract_word_limit(text: str):
    match = re.search(r"\b(\d{2,4})\s*(?:word|words)\b", text.lower())
    if not match:
        return None
    return max(20, min(500, int(match.group(1))))

def run_llm(template: str, variables: dict) -> str:
    chain = ChatPromptTemplate.from_template(template) | get_llm() | StrOutputParser()
    return chain.invoke(variables)

def safe_parse_json(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

def source_stats(docs):
    if not docs:
        return {"confidence": "Low", "pages": [], "files": [], "count": 0}
    pages = []
    files = []
    for doc in docs:
        page = doc_page(doc)
        if page is not None:
            pages.append(page)
        src = doc.metadata.get("source_file")
        if src:
            files.append(src)
    unique_pages = sorted(set(pages))
    unique_files = sorted(set(files))
    confidence = "High" if len(docs) >= 4 and len(unique_pages) >= 2 else "Medium" if len(docs) >= 2 else "Low"
    return {"confidence": confidence, "pages": unique_pages, "files": unique_files, "count": len(docs)}

def render_source_insight(docs):
    stats = source_stats(docs)
    pages = ", ".join(str(p) for p in stats["pages"][:8]) or "unknown"
    files = ", ".join(stats["files"][:3]) or "unknown"
    st.markdown(
        f'<div class="insight-box">Confidence: <b>{stats["confidence"]}</b><br>'
        f'Source coverage: {stats["count"]} chunks · pages {pages} · files {files}</div>',
        unsafe_allow_html=True,
    )

def render_sources(docs, title="Sources Used"):
    if not docs:
        return
    with st.expander(f"📎 {len(docs)} {title}"):
        for doc in docs:
            src = doc.metadata.get("source_file", "Unknown")
            page = doc_page(doc) or "?"
            preview = escape(doc.page_content[:360].replace("\n", " "))
            st.markdown(
                f'<div class="source-card"><div class="source-meta">📄 {src} · Page {page}</div>{preview}…</div>',
                unsafe_allow_html=True,
            )

def update_weak_topics(questions, answers):
    missed = {}
    for i, q in enumerate(questions):
        if answers.get(i) == q.get("answer"):
            continue
        topic = (q.get("topic") or q.get("concept") or q.get("question", "General")[:42]).strip()
        missed[topic] = missed.get(topic, 0) + 1
        st.session_state.weak_topics[topic] = st.session_state.weak_topics.get(topic, 0) + 1
    return missed

def weakest_topics(limit=4):
    return [
        topic for topic, _ in sorted(
            st.session_state.weak_topics.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )[:limit]
    ]

def export_markdown():
    lines = [
        "# DocuMind Study Pack",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Indexed Files",
        *(f"- {name}" for name in st.session_state.indexed_files),
        "",
    ]
    if st.session_state.summary_text:
        lines.extend(["## Summary", st.session_state.summary_text, ""])
    if st.session_state.flashcards:
        lines.append("## Flashcards")
        for card in st.session_state.flashcards:
            lines.append(f"- **{card.get('term', '')}**: {card.get('definition', '')}")
        lines.append("")
    if st.session_state.quiz_questions:
        lines.append("## Quiz")
        for i, q in enumerate(st.session_state.quiz_questions, 1):
            lines.append(f"{i}. {q.get('question', '')}")
            for opt, text in q.get("options", {}).items():
                lines.append(f"   - {opt}. {text}")
            lines.append(f"   - Answer: {q.get('answer', '')}")
            if q.get("explanation"):
                lines.append(f"   - Explanation: {q.get('explanation')}")
        lines.append("")
    if st.session_state.weak_topics:
        lines.append("## Weak Topics")
        for topic, count in sorted(st.session_state.weak_topics.items(), key=lambda item: (-item[1], item[0].lower())):
            lines.append(f"- {topic}: {count}")
        lines.append("")
    if st.session_state.study_plan:
        lines.extend(["## Study Plan", st.session_state.study_plan, ""])
    if st.session_state.concept_map:
        lines.append("## Concept Map")
        for item in st.session_state.concept_map:
            lines.append(f"- {item.get('from', '')} -> {item.get('to', '')}: {item.get('relation', '')}")
    return "\n".join(lines)

def export_pdf_bytes(markdown_text):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        return None
    from io import BytesIO

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 48
    pdf.setFont("Helvetica", 10)
    for raw_line in markdown_text.splitlines():
        line = raw_line.replace("#", "").replace("*", "")
        wrapped = [line[i:i + 95] for i in range(0, len(line), 95)] or [""]
        for part in wrapped:
            if y < 48:
                pdf.showPage()
                pdf.setFont("Helvetica", 10)
                y = height - 48
            pdf.drawString(48, y, part)
            y -= 14
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

def render_export_buttons():
    md = export_markdown()
    st.download_button("⬇  Export Markdown", md, file_name="documind-study-pack.md", mime="text/markdown")
    pdf_bytes = export_pdf_bytes(md)
    if pdf_bytes:
        st.download_button("⬇  Export PDF", pdf_bytes, file_name="documind-study-pack.pdf", mime="application/pdf")

def render_concept_map(edges):
    if not edges:
        return
    parts = []
    for edge in edges:
        left = escape(str(edge.get("from", "")))
        right = escape(str(edge.get("to", "")))
        relation = escape(str(edge.get("relation", "")))
        parts.append(
            f'<div style="margin:10px 0;"><span class="map-node">{left}</span>'
            f'<span style="color:#c9a96e;margin:0 8px;">→ {relation} →</span>'
            f'<span class="map-node">{right}</span></div>'
        )
    components.html(
        f"""
        <style>
        body {{ background:#0d0d0d; color:#e8e4dc; font-family:monospace; }}
        .map-node {{ display:inline-block; background:#141414; border:1px solid #292929; border-radius:4px; color:#e8e4dc; padding:7px 11px; margin:3px; font-size:13px; }}
        </style>
        <div>{''.join(parts)}</div>
        """,
        height=min(520, 70 + len(edges) * 48),
    )

# ─────────────────────────────────────────────
# Feature Functions
# ─────────────────────────────────────────────
def ask_question(question, chat_history, page_range=None):
    page_range = page_range or extract_page_range_from_text(question)
    if is_summary_request(question):
        return generate_summary(question, page_range=page_range)

    retrieved_docs, context_text = get_context(question, k=6, page_range=page_range)
    history_text = "".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}\n"
        for m in chat_history[-6:]
    )
    answer = run_llm("""You are DocuMind, a precise document assistant.
Answer ONLY using the provided context. If not in context, say "I don't have enough information in the uploaded documents."
Be concise and structured.

Conversation History:
{history}

Context:
{context}

Question: {question}""",
        {"history": history_text, "context": context_text, "question": question})
    return answer, retrieved_docs

def generate_quiz(topic: str, num_q: int = 5, difficulty: str = "Exam", page_range=None, adaptive: bool = False):
    focus = topic.strip()
    if adaptive:
        weak = weakest_topics()
        focus = f"{focus}. Focus weak areas: {', '.join(weak)}" if focus and weak else ", ".join(weak) or focus
    _, context = get_topic_context(focus, "key ideas concepts definitions examples", k=10, page_range=page_range)
    raw = run_llm("""You are an expert quiz creator.
Generate exactly {num_q} multiple choice questions based on the context.

Rules:
- Each question has exactly 4 options: A, B, C, D
- Exactly one option is correct
- Difficulty: {difficulty}
- Beginner: test essential recall and definitions
- Exam: test understanding, comparison, and application
- Advanced: test edge cases, synthesis, and deeper reasoning
- Base every question strictly on the provided context
- Add a short "topic" label for each question so weak areas can be tracked

Return ONLY a valid JSON array, no markdown, no explanation:
[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "answer": "A",
    "explanation": "Brief explanation.",
    "topic": "Short topic label"
  }}
]

Context:
{context}

Topic: {topic}""",
        {"num_q": num_q, "context": context, "topic": focus or "general", "difficulty": difficulty})
    try:
        return safe_parse_json(raw)
    except Exception:
        return []

def generate_flashcards(topic: str, num: int = 8, page_range=None):
    docs, context = get_topic_context(topic, "main ideas key terms definitions examples processes", k=14, page_range=page_range)
    raw = run_llm("""You are a study assistant. Generate exactly {num} high-signal flashcards from the context.

Rules:
- If Topic is not blank, every flashcard must be about that topic or a directly related idea shown in the context.
- Prefer exact terms, people, places, formulas, dates, definitions, causes, effects, methods, steps, examples, or named concepts from the PDF.
- Avoid generic cards unless the definition explains their specific role in the requested topic.
- Definitions must be concrete and useful for studying, with 1-2 details from the context.
- Do not invent facts outside the context.

Return ONLY a valid JSON array, no markdown:
[
  {{"term": "Exact study term", "definition": "Specific explanation grounded in the PDF."}}
]

Context:
{context}

Topic: {topic}""",
        {"num": num, "context": context, "topic": topic or "key concepts"})
    try:
        return safe_parse_json(raw)
    except Exception:
        return []

def generate_summary(topic: str, page_range=None):
    page_range = page_range or extract_page_range_from_text(topic or "")
    word_limit = extract_word_limit(topic or "")
    if is_document_summary_request(topic or ""):
        docs = select_representative_docs(page_range=page_range)
        context = "\n\n---\n\n".join([d.page_content for d in docs])
        summary_target = "the whole uploaded PDF/document"
    else:
        docs, context = get_context(topic or "main topics overview key concepts", k=10, page_range=page_range)
        summary_target = topic

    if not docs:
        return "No indexed document content found. Upload and index a PDF first.", []

    summary = run_llm("""You are a study assistant summarising uploaded study PDFs.
Use ONLY the provided context.
If the user asks for the whole PDF/document/book, summarise the main subject, scope, and major concepts across the document.
Do not summarise only prerelease notes, revision notes, contact details, bug-report instructions, testimonials, or other front matter unless the user asks for metadata.
If a word limit is provided, stay under that limit.
Plain text only. No markdown headers.

Context:
{context}

Summary target: {summary_target}
Requested word limit: {word_limit}""",
        {"context": context, "summary_target": summary_target, "word_limit": word_limit or "none"})
    return summary, docs

def generate_study_plan(days: int, focus: str):
    weak = ", ".join(weakest_topics(6)) or "none yet"
    docs, context = get_topic_context(focus, "main ideas key concepts learning objectives", k=12)
    if not docs:
        return "No indexed document content found. Upload and index a PDF first."
    return run_llm("""Create a practical study plan from the uploaded document context.
Use ONLY the context and the weak topics. Make exactly {days} days.
Each day must include: reading focus, active recall task, quiz/revision task, and expected outcome.
Keep it concise and exam-oriented.

Context:
{context}

Requested focus: {focus}
Known weak topics: {weak}""",
        {"days": days, "context": context, "focus": focus or "whole document", "weak": weak})

def generate_concept_map(focus: str):
    docs, context = get_topic_context(focus, "main ideas key terms relationships", k=14)
    if not docs:
        return []
    raw = run_llm("""You are a document analysis assistant.
Use ONLY the provided context to extract concept relationships.
Do not invent concepts, examples, or domain knowledge that are not present in the context.
If the context does not support a valid relationship, return an empty JSON array.
Return ONLY a valid JSON array, without markdown or explanation.
Each edge must connect two concrete concepts from the document.
Use 6 to 12 edges.
[
  {{"from": "Concept A", "to": "Concept B", "relation": "short relationship"}}
]

Context:
{context}

Focus: {focus}""",
        {"context": context, "focus": focus or "whole document"})
    try:
        return safe_parse_json(raw)
    except Exception:
        return []


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;background:linear-gradient(135deg,#f0e6d3,#c9a96e,#f0e6d3);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">DocuMind Study</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Smart Study Assistant</div>', unsafe_allow_html=True)

    if st.session_state.vector_store_ready:
        st.markdown('<span class="status-badge">● READY</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge inactive">○ NO DOCS LOADED</span>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

    if uploaded_files:
        if st.button("⬆  Index Documents"):
            with st.spinner(""):
                chunk_count = index_pdfs(uploaded_files)
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.session_state.total_chunks = chunk_count
                st.session_state.vector_store_ready = True
                for k in ["chat_history", "quiz_questions", "quiz_answers", "flashcards", "summary_text"]:
                    st.session_state[k] = defaults[k]
                save_progress()
                st.success(f"Indexed {chunk_count} chunks!")

    if st.session_state.indexed_files:
        st.markdown('<div class="sidebar-label">Indexed Files</div>', unsafe_allow_html=True)
        for fname in st.session_state.indexed_files:
            st.markdown(f'<span class="file-pill">📄 {fname}</span>', unsafe_allow_html=True)

    if st.session_state.vector_store_ready:
        st.markdown('<div class="sidebar-label">Stats</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{len(st.session_state.indexed_files)}</div><div class="metric-label">Docs</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{st.session_state.total_chunks}</div><div class="metric-label">Chunks</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Actions</div>', unsafe_allow_html=True)
    if st.button("🗑  Clear Everything"):
        try:
            get_qdrant_client().delete_collection(COLLECTION_NAME)
        except:
            pass
        for k, v in defaults.items():
            st.session_state[k] = v
        try:
            STATE_FILE.unlink()
        except OSError:
            pass
        st.rerun()

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.62rem;color:#333;letter-spacing:1px;line-height:2;">QDRANT · LANGCHAIN<br>HUGGINGFACE · GEMINI<br>STREAMLIT</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">DocuMind Study</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Your AI-powered study assistant</div>', unsafe_allow_html=True)
st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

if not st.session_state.vector_store_ready:
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;">
        <div style="font-size:4rem;margin-bottom:1rem;">📚</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#333;letter-spacing:2px;">UPLOAD YOUR NOTES TO BEGIN</div>
        <div style="font-size:0.75rem;color:#222;margin-top:0.5rem;letter-spacing:1px;">Upload PDFs from the sidebar — textbooks, notes, slides</div>
    </div>""", unsafe_allow_html=True)

else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬  Chat", "📝  Quiz", "🗂  Flashcards", "📖  Summarise", "🧭  Study Lab"])

    # ── TAB 1: CHAT ──
    with tab1:
        st.markdown('<div class="section-header">Ask your documents anything</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Grounded answers from your uploaded notes only</div>', unsafe_allow_html=True)
        chat_page_text = st.text_input("Chat page range", placeholder="optional: pages 12-18", label_visibility="visible", key="chat_pages")
        chat_page_range = parse_page_range(chat_page_text)
        st.markdown('<div style="font-size:0.75rem;color:#777;margin-bottom:0.6rem;">Enter a page range to narrow chat answers to specific pages.</div>', unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user"><div class="chat-label label-user">You</div>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble-ai"><div class="chat-label label-ai">DocuMind</div>{msg["content"]}</div>', unsafe_allow_html=True)
                if msg.get("sources"):
                    render_source_insight(msg["sources"])
                    render_sources(msg["sources"], "Sources")

        if question := st.chat_input("Ask anything from your notes..."):
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Thinking..."):
                answer, sources = ask_question(question, st.session_state.chat_history, page_range=chat_page_range)
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🔄  Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    # ── TAB 2: QUIZ ──
    with tab2:
        st.markdown('<div class="section-header">Auto Quiz Generator</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">One-click questions generated from your uploaded material</div>', unsafe_allow_html=True)

        ca, cb, cc, cd, ce = st.columns([2.3, 1, 1, 1, 1])
        with ca:
            quiz_topic = st.text_input("Quiz topic or chapter", placeholder="e.g. Chapter 3, key terms, causes and effects... (blank = general)", label_visibility="visible")
        with cb:
            num_q = st.selectbox("Number of questions", [5, 8, 10], label_visibility="visible")
        with cc:
            quiz_difficulty = st.selectbox("Quiz difficulty", ["Beginner", "Exam", "Advanced"], index=1, label_visibility="visible")
        with cd:
            quiz_pages = st.text_input("Quiz page range", placeholder="e.g., 1-10", label_visibility="visible", key="quiz_pages")
        with ce:
            adaptive = st.checkbox("Adaptive quiz", value=False, help="Focus on weak topics from previous quizzes.")

        btn_a, btn_b, btn_c, btn_d, btn_e = st.columns([2.3, 1, 1, 1, 1])
        with btn_a:
            if st.button("⚡  Generate Quiz"):
                with st.spinner("Building your quiz..."):
                    qs = generate_quiz(
                        quiz_topic,
                        num_q,
                        difficulty=quiz_difficulty,
                        page_range=parse_page_range(quiz_pages),
                        adaptive=adaptive,
                    )
                if qs:
                    st.session_state.quiz_questions = qs
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.session_state.last_missed_topics = {}
                    st.session_state.last_quiz_topic = quiz_topic
                    st.session_state.last_quiz_difficulty = quiz_difficulty
                    st.session_state.last_quiz_page_range = quiz_pages
                    save_progress()
                else:
                    st.error("Quiz generation failed. Try a different topic.")

        if weakest_topics():
            st.markdown(
                '<div class="insight-box">Adaptive focus available: '
                + " ".join(f'<span class="gap-chip">{escape(t)}</span>' for t in weakest_topics())
                + "</div>",
                unsafe_allow_html=True,
            )

        if st.session_state.quiz_questions:
            qs = st.session_state.quiz_questions

            if not st.session_state.quiz_submitted:
                for i, q in enumerate(qs):
                    st.markdown(f'<div class="quiz-card"><div class="quiz-num">Question {i+1} of {len(qs)}</div><div class="quiz-q">{q["question"]}</div></div>', unsafe_allow_html=True)
                    choice = st.radio(
                        f"q{i}", list(q["options"].keys()),
                        format_func=lambda x, q=q: f"{x}.  {q['options'][x]}",
                        key=f"qr_{i}", label_visibility="collapsed"
                    )
                    st.session_state.quiz_answers[i] = choice
                    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

                if st.button("✅  Submit & See Results"):
                    missed = update_weak_topics(qs, st.session_state.quiz_answers)
                    correct = sum(1 for i, q in enumerate(qs) if st.session_state.quiz_answers.get(i) == q["answer"])
                    st.session_state.last_missed_topics = missed
                    st.session_state.quiz_history.append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "topic": st.session_state.last_quiz_topic or "general",
                        "difficulty": st.session_state.last_quiz_difficulty,
                        "score": correct,
                        "total": len(qs),
                        "missed": missed,
                    })
                    st.session_state.quiz_submitted = True
                    save_progress()
                    st.rerun()

            else:
                correct = sum(1 for i, q in enumerate(qs) if st.session_state.quiz_answers.get(i) == q["answer"])
                total = len(qs)
                pct = int((correct / total) * 100)
                grade = "Excellent!" if pct >= 80 else "Good effort!" if pct >= 60 else "Keep studying!"

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="score-box"><div class="score-big">{correct}/{total}</div><div class="score-label">Score</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="score-box"><div class="score-big">{pct}%</div><div class="score-label">Percentage</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="score-box"><div class="score-big" style="font-size:1.4rem;padding-top:0.8rem;">{grade}</div><div class="score-label">Result</div></div>', unsafe_allow_html=True)

                st.markdown(f'<div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{pct}%"></div></div>', unsafe_allow_html=True)
                if st.session_state.last_missed_topics:
                    st.markdown(
                        '<div class="insight-box">Knowledge gaps: '
                        + " ".join(f'<span class="gap-chip">{escape(t)} ×{c}</span>' for t, c in st.session_state.last_missed_topics.items())
                        + "<br>Next adaptive quiz will target these areas.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('<div class="insight-box">No weak topics detected in this quiz.</div>', unsafe_allow_html=True)
                st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

                for i, q in enumerate(qs):
                    user_ans = st.session_state.quiz_answers.get(i, "")
                    is_correct = user_ans == q["answer"]
                    st.markdown(f'<div class="quiz-num">Question {i+1}</div><div class="quiz-q">{q["question"]}</div>', unsafe_allow_html=True)
                    for opt, text in q["options"].items():
                        clr = "#4caf50" if opt == q["answer"] else "#ef5350" if opt == user_ans and not is_correct else "#555"
                        mark = " ✓" if opt == q["answer"] else " ✗" if opt == user_ans and not is_correct else ""
                        st.markdown(f'<div style="font-size:0.82rem;color:{clr};padding:0.15rem 0;">{opt}. {text}{mark}</div>', unsafe_allow_html=True)
                    if is_correct:
                        st.markdown(f'<div class="correct-ans">✓ Correct! {q.get("explanation","")}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="wrong-ans">✗ You chose {user_ans}. Correct: {q["answer"]}. {q.get("explanation","")}</div>', unsafe_allow_html=True)
                    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

                if st.button("🔁  New Quiz"):
                    st.session_state.quiz_questions = []
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.rerun()

    # ── TAB 3: FLASHCARDS ──
    with tab3:
        st.markdown('<div class="section-header">Flashcard Maker</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Study cards auto-generated from your uploaded material</div>', unsafe_allow_html=True)

        ca, cb, cc, cd = st.columns([2.5, 1, 1, 1])
        with ca:
            fc_topic = st.text_input("Flashcard topic or chapter", placeholder="e.g. main ideas, vocabulary, dates, formulas...", label_visibility="visible", key="fc_t")
        with cb:
            fc_num = st.selectbox("Number of cards", [8, 12, 15], label_visibility="visible")
        with cc:
            fc_pages = st.text_input("Flashcard page range", placeholder="e.g., 1-10", label_visibility="visible", key="fc_pages")
        with cd:
            if st.button("⚡  Generate Cards"):
                with st.spinner("Creating flashcards..."):
                    cards = generate_flashcards(fc_topic, fc_num, page_range=parse_page_range(fc_pages))
                    if cards:
                        st.session_state.flashcards = cards
                        st.session_state.fc_index = 0
                        st.session_state.fc_show_back = False
                        save_progress()
                    else:
                        st.error("Could not generate flashcards. Try a different topic.")

        if st.session_state.flashcards:
            cards = st.session_state.flashcards
            idx = st.session_state.fc_index
            card = cards[idx]
            total_fc = len(cards)

            pct = int(((idx + 1) / total_fc) * 100)
            st.markdown(f'<div style="font-size:0.65rem;letter-spacing:2px;color:#555;text-transform:uppercase;margin-bottom:0.3rem;">Card {idx+1} of {total_fc}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{pct}%"></div></div>', unsafe_allow_html=True)

            if not st.session_state.fc_show_back:
                st.markdown(f'<div class="flashcard-front"><div><div class="flashcard-label">Term — click Show Answer to reveal</div><div class="flashcard-term">{card["term"]}</div></div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="flashcard-back"><div><div class="flashcard-label">Definition</div><div class="flashcard-def">{card["definition"]}</div></div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("◀  Prev") and idx > 0:
                    st.session_state.fc_index -= 1
                    st.session_state.fc_show_back = False
                    st.rerun()
            with c2:
                lbl = "👁  Show Answer" if not st.session_state.fc_show_back else "🙈  Hide"
                if st.button(lbl):
                    st.session_state.fc_show_back = not st.session_state.fc_show_back
                    st.rerun()
            with c3:
                if st.button("Next  ▶") and idx < total_fc - 1:
                    st.session_state.fc_index += 1
                    st.session_state.fc_show_back = False
                    st.rerun()
            with c4:
                if st.button("🔀  Shuffle"):
                    random.shuffle(st.session_state.flashcards)
                    st.session_state.fc_index = 0
                    st.session_state.fc_show_back = False
                    save_progress()
                    st.rerun()

            st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.65rem;letter-spacing:2px;color:#333;text-transform:uppercase;margin-bottom:0.6rem;">All Terms</div>', unsafe_allow_html=True)
            for j, c in enumerate(cards):
                clr = "#c9a96e" if j == idx else "#444"
                st.markdown(f'<div style="font-size:0.78rem;color:{clr};padding:0.3rem 0;border-bottom:1px solid #1a1a1a;cursor:pointer;"><b>{j+1}.</b> {c["term"]}</div>', unsafe_allow_html=True)

    # ── TAB 4: SUMMARISE ──
    with tab4:
        st.markdown('<div class="section-header">Topic Summariser</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Exam-ready summaries pulled directly from your notes</div>', unsafe_allow_html=True)

        ca, cb, cc = st.columns([3, 1, 1])
        with ca:
            sum_topic = st.text_input("Summary topic or chapter", placeholder="e.g. whole document, Chapter 2, main argument, key concepts...", label_visibility="visible")
        with cb:
            sum_pages = st.text_input("Summary page range", placeholder="e.g., 1-10", label_visibility="visible", key="sum_pages")
        with cc:
            if st.button("📖  Summarise"):
                if sum_topic.strip():
                    with st.spinner(f"Summarising '{sum_topic}'..."):
                        summary, src_docs = generate_summary(sum_topic, page_range=parse_page_range(sum_pages))
                        st.session_state.summary_text = summary
                        st.session_state.summary_topic = sum_topic
                        st.session_state.summary_sources = src_docs
                        save_progress()
                else:
                    st.warning("Please enter a topic first.")

        if st.session_state.summary_text:
            st.markdown(f'<div style="font-size:0.62rem;letter-spacing:3px;text-transform:uppercase;color:#c9a96e;font-family:Syne,sans-serif;margin-bottom:0.6rem;">Summary: {st.session_state.summary_topic}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-box">{st.session_state.summary_text}</div>', unsafe_allow_html=True)

            if st.session_state.summary_sources:
                render_source_insight(st.session_state.summary_sources)
                render_sources(st.session_state.summary_sources)

    # ── TAB 5: STUDY LAB ──
    with tab5:
        st.markdown('<div class="section-header">Adaptive Study Lab</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Progress tracking, study plans, concept maps, and exportable packs</div>', unsafe_allow_html=True)

        if st.session_state.quiz_history:
            latest = st.session_state.quiz_history[-1]
            pct = int((latest["score"] / latest["total"]) * 100) if latest["total"] else 0
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="score-box"><div class="score-big">{pct}%</div><div class="score-label">Latest Quiz</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="score-box"><div class="score-big">{len(st.session_state.quiz_history)}</div><div class="score-label">Attempts</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="score-box"><div class="score-big">{len(st.session_state.weak_topics)}</div><div class="score-label">Weak Topics</div></div>', unsafe_allow_html=True)

        if st.session_state.weak_topics:
            st.markdown('<div class="section-header" style="font-size:0.95rem;">Knowledge Gaps</div>', unsafe_allow_html=True)
            st.markdown(
                " ".join(
                    f'<span class="gap-chip">{escape(topic)} ×{count}</span>'
                    for topic, count in sorted(st.session_state.weak_topics.items(), key=lambda item: (-item[1], item[0].lower()))
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
        pa, pb, pc = st.columns([3, 1, 1])
        with pa:
            plan_focus = st.text_input("Study plan focus", placeholder="e.g. whole document, missed topics, chapter 4...", label_visibility="visible", key="plan_focus")
        with pb:
            plan_days = st.selectbox("Plan duration (days)", [3, 5, 7, 14], index=2, label_visibility="visible")
        with pc:
            if st.button("🗓  Build Plan"):
                with st.spinner("Building study plan..."):
                    st.session_state.study_plan = generate_study_plan(plan_days, plan_focus)
                    save_progress()

        if st.session_state.study_plan:
            st.markdown('<div class="section-header" style="font-size:0.95rem;">Study Plan</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-box">{st.session_state.study_plan}</div>', unsafe_allow_html=True)

        st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
        ma, mb = st.columns([4, 1])
        with ma:
            map_focus = st.text_input("Concept map focus", placeholder="e.g. key concepts, chapter 2, missed topics...", label_visibility="visible", key="map_focus")
            st.markdown('<div style="font-size:0.75rem;color:#777;margin-top:0.4rem;">Enter a focus so the concept map stays grounded in your uploaded content.</div>', unsafe_allow_html=True)
        with mb:
            if st.button("🕸  Map"):
                if not map_focus.strip():
                    st.warning("Please enter a concept map focus before generating the map.")
                else:
                    with st.spinner("Mapping concepts..."):
                        st.session_state.concept_map = generate_concept_map(map_focus)
                        save_progress()

        if st.session_state.concept_map:
            st.markdown('<div class="section-header" style="font-size:0.95rem;">Concept Map</div>', unsafe_allow_html=True)
            render_concept_map(st.session_state.concept_map)

        st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="font-size:0.95rem;">Export Study Pack</div>', unsafe_allow_html=True)
        render_export_buttons()
