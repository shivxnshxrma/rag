import streamlit as st
import time
import json
import random
import tempfile
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    padding: 2.5rem 2rem;
    text-align: center;
    min-height: 180px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.flashcard-back {
    background: #0a0f1f;
    border: 1px solid #4a9eff;
    border-radius: 8px;
    padding: 2.5rem 2rem;
    text-align: center;
    min-height: 180px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.flashcard-term { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: #c9a96e; line-height: 1.4; }
.flashcard-def { font-size: 0.9rem; color: #b0c8f0; line-height: 1.8; }
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
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


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
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)


# ─────────────────────────────────────────────
# Core Helpers
# ─────────────────────────────────────────────
COLLECTION_NAME = "documind_rag"

def ensure_collection(client, dim=384):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(COLLECTION_NAME, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

def index_pdfs(uploaded_files):
    client = get_qdrant_client()
    ensure_collection(client)
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

def get_context(query: str, k: int = 6):
    vs = QdrantVectorStore(client=get_qdrant_client(), collection_name=COLLECTION_NAME, embedding=get_embedding_model())
    docs = vs.as_retriever(search_kwargs={"k": k}).invoke(query)
    return docs, "\n\n---\n\n".join([d.page_content for d in docs])

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


# ─────────────────────────────────────────────
# Feature Functions
# ─────────────────────────────────────────────
def ask_question(question, chat_history):
    retrieved_docs, context_text = get_context(question, k=4)
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

def generate_quiz(topic: str, num_q: int = 5):
    _, context = get_context(topic or "key concepts definitions", k=8)
    raw = run_llm("""You are an expert quiz creator.
Generate exactly {num_q} multiple choice questions based on the context.

Rules:
- Each question has exactly 4 options: A, B, C, D
- Exactly one option is correct
- Test genuine understanding, not trivial recall
- Base every question strictly on the provided context

Return ONLY a valid JSON array, no markdown, no explanation:
[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "answer": "A",
    "explanation": "Brief explanation."
  }}
]

Context:
{context}

Topic: {topic}""",
        {"num_q": num_q, "context": context, "topic": topic or "general"})
    try:
        return safe_parse_json(raw)
    except Exception:
        return []

def generate_flashcards(topic: str, num: int = 8):
    _, context = get_context(topic or "definitions terms concepts", k=8)
    raw = run_llm("""You are a study assistant. Generate exactly {num} flashcards from the context.
Return ONLY a valid JSON array, no markdown:
[
  {{"term": "Short term", "definition": "Clear explanation in 1-2 sentences."}}
]

Context:
{context}

Topic: {topic}""",
        {"num": num, "context": context, "topic": topic or "key concepts"})
    try:
        return safe_parse_json(raw)
    except Exception:
        return []

def generate_summary(topic: str):
    docs, context = get_context(topic, k=8)
    summary = run_llm("""You are a study assistant. Write a clear, structured, exam-ready summary.
Use ONLY the provided context. Short paragraphs. Plain text, no markdown headers.

Context:
{context}

Topic: {topic}""",
        {"context": context, "topic": topic})
    return summary, docs


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
    tab1, tab2, tab3, tab4 = st.tabs(["💬  Chat", "📝  Quiz", "🗂  Flashcards", "📖  Summarise"])

    # ── TAB 1: CHAT ──
    with tab1:
        st.markdown('<div class="section-header">Ask your documents anything</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Grounded answers from your uploaded notes only</div>', unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user"><div class="chat-label label-user">You</div>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble-ai"><div class="chat-label label-ai">DocuMind</div>{msg["content"]}</div>', unsafe_allow_html=True)
                if msg.get("sources"):
                    with st.expander(f"📎 {len(msg['sources'])} Sources"):
                        for doc in msg["sources"]:
                            src = doc.metadata.get("source_file", "Unknown")
                            page = doc.metadata.get("page", "?")
                            preview = doc.page_content[:200].replace("\n", " ")
                            st.markdown(f'<div class="source-card"><div class="source-meta">📄 {src} · Page {page}</div>{preview}…</div>', unsafe_allow_html=True)

        if question := st.chat_input("Ask anything from your notes..."):
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Thinking..."):
                answer, sources = ask_question(question, st.session_state.chat_history)
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🔄  Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    # ── TAB 2: QUIZ ──
    with tab2:
        st.markdown('<div class="section-header">Auto Quiz Generator</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">One-click MCQs generated from your own notes</div>', unsafe_allow_html=True)

        ca, cb, cc = st.columns([3, 1, 1])
        with ca:
            quiz_topic = st.text_input("Topic", placeholder="e.g. OSI Model, Sorting Algorithms... (blank = general)", label_visibility="collapsed")
        with cb:
            num_q = st.selectbox("No. of Questions", [5, 8, 10], label_visibility="collapsed")
        with cc:
            if st.button("⚡  Generate Quiz"):
                with st.spinner("Building your quiz..."):
                    qs = generate_quiz(quiz_topic, num_q)
                    if qs:
                        st.session_state.quiz_questions = qs
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = False
                    else:
                        st.error("Quiz generation failed. Try a different topic.")

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
                    st.session_state.quiz_submitted = True
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
        st.markdown('<div class="section-sub">Term → Definition cards auto-generated from your notes</div>', unsafe_allow_html=True)

        ca, cb, cc = st.columns([3, 1, 1])
        with ca:
            fc_topic = st.text_input("Topic", placeholder="e.g. Networking, Data Structures...", label_visibility="collapsed", key="fc_t")
        with cb:
            fc_num = st.selectbox("No. of Cards", [8, 12, 15], label_visibility="collapsed")
        with cc:
            if st.button("⚡  Generate Cards"):
                with st.spinner("Creating flashcards..."):
                    cards = generate_flashcards(fc_topic, fc_num)
                    if cards:
                        st.session_state.flashcards = cards
                        st.session_state.fc_index = 0
                        st.session_state.fc_show_back = False
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

        ca, cb = st.columns([4, 1])
        with ca:
            sum_topic = st.text_input("Topic", placeholder="e.g. TCP/IP model, Binary Trees, French Revolution...", label_visibility="collapsed")
        with cb:
            if st.button("📖  Summarise"):
                if sum_topic.strip():
                    with st.spinner(f"Summarising '{sum_topic}'..."):
                        summary, src_docs = generate_summary(sum_topic)
                        st.session_state.summary_text = summary
                        st.session_state.summary_topic = sum_topic
                        st.session_state.summary_sources = src_docs
                else:
                    st.warning("Please enter a topic first.")

        if st.session_state.summary_text:
            st.markdown(f'<div style="font-size:0.62rem;letter-spacing:3px;text-transform:uppercase;color:#c9a96e;font-family:Syne,sans-serif;margin-bottom:0.6rem;">Summary: {st.session_state.summary_topic}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="summary-box">{st.session_state.summary_text}</div>', unsafe_allow_html=True)

            if st.session_state.summary_sources:
                with st.expander(f"📎 {len(st.session_state.summary_sources)} Sources Used"):
                    for doc in st.session_state.summary_sources:
                        src = doc.metadata.get("source_file", "Unknown")
                        page = doc.metadata.get("page", "?")
                        preview = doc.page_content[:180].replace("\n", " ")
                        st.markdown(f'<div class="source-card"><div class="source-meta">📄 {src} · Page {page}</div>{preview}…</div>', unsafe_allow_html=True)