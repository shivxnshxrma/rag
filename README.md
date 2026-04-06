# DocuMind Study 🧠

> AI-powered study assistant — quiz, flashcards & summaries from your own PDFs.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?style=flat-square)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-purple?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-2.5Flash-orange?style=flat-square)

---

## What is DocuMind Study?

DocuMind Study is a Retrieval-Augmented Generation (RAG) application that turns your static PDF notes into a complete AI-powered study system. Upload your textbooks or lecture notes once — and instantly get:

- 💬 **Chat** — Ask anything, get answers grounded in your own documents with page citations
- 📝 **Auto Quiz** — One-click MCQ generation with scoring and per-question explanations
- 🗂 **Flashcards** — Auto-generated term → definition cards with flip, shuffle & navigation
- 📖 **Summariser** — Focused, exam-ready topic summaries pulled from your notes

Every output is traceable to the exact source document and page number.

---

## Architecture

```
PDF Upload → PyPDFLoader → Chunks → MiniLM Embeddings → Qdrant
                                                            ↓
User Query → MiniLM Embedding → Cosine Search → Top-k Chunks
                                                            ↓
              Prompt (chunks + history + query) → Gemini 2.5 Flash
                                                            ↓
                    Answer / Quiz / Flashcards / Summary → Streamlit UI
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, no API key) |
| Vector DB | Qdrant (Docker) |
| Orchestration | LangChain LCEL |
| Language | Python 3.12 |

---

## Getting Started

### Prerequisites
- Python 3.12+
- Docker
- Google Gemini API key → [Get one free at aistudio.google.com](https://aistudio.google.com)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/documind-study.git
cd documind-study

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Qdrant vector database
docker compose up -d

# 5. Set your Gemini API key
export GOOGLE_API_KEY="your-key-here"   # Mac/Linux
# set GOOGLE_API_KEY=your-key-here      # Windows

# 6. Run the app
streamlit run app.py
```

App opens at **http://localhost:8501**

---

## Project Structure

```
documind-study/
├── app.py               # Main Streamlit app (all 4 features)
├── docker-compose.yml   # Qdrant service
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Features in Detail

### 💬 Chat
- Ask natural language questions about your uploaded PDFs
- Answers are grounded **only** in your documents — no hallucination from internet data
- Every answer shows expandable source cards with filename and page number
- Conversation memory retains the last 3 turns for follow-up questions

### 📝 Quiz Generator
- Choose a topic (or leave blank for a general quiz)
- Select 5, 8, or 10 questions
- After submission: score, percentage, progress bar, and colour-coded per-question feedback with explanations

### 🗂 Flashcards
- Auto-generates term → definition pairs from your notes
- Flip cards to reveal definitions
- Navigate forward/backward, shuffle the deck, track progress

### 📖 Topic Summariser
- Type any topic covered in your notes
- Get a concise, exam-ready summary synthesised from your documents
- Source citations shown below every summary

---

## Requirements

```
streamlit
qdrant-client
langchain
langchain-qdrant
langchain-huggingface
langchain-google-genai
langchain-community
sentence-transformers
pypdf
google-generativeai
```

---

## Why Not Just Use ChatGPT?

| Feature | ChatGPT | DocuMind Study |
|---|---|---|
| Answer from your notes | ❌ Manual copy-paste | ✅ Auto from PDF |
| Quiz generation | ❌ Manual prompting | ✅ One click |
| Flashcard maker | ❌ Manual prompting | ✅ Auto with flip UI |
| Topic summary | ❌ Manual prompting | ✅ From your notes |
| Source citations | ❌ | ✅ Filename + page |
| Private / local embeddings | ❌ | ✅ Runs offline |

---

## License

MIT License — free to use, modify, and distribute.