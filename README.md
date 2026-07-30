# DocMind — Multi-Document RAG Research Assistant

A production-quality **Retrieval-Augmented Generation (RAG)** pipeline that lets you upload PDF documents and ask natural language questions. Answers are grounded in your documents with verifiable source citations.

Built from scratch as a learning project demonstrating every layer of a modern RAG system.

---

## Live Demo

> **[DocMind on Streamlit Cloud →](https://rag-pdf-assistant-rithul.streamlit.app)**

---

## Architecture

```
PDF Upload
    ↓  PyMuPDF — extract text + page numbers
    ↓  chunker.py — recursive character split (800 chars, 150 overlap)
    ↓  all-MiniLM-L6-v2 — 384-dim sentence embeddings → ChromaDB

User Question
    ↓  BM25 keyword search   ─┐
                               ├─ Reciprocal Rank Fusion (RRF)
    ↓  Dense cosine search   ─┘
    ↓  ms-marco-MiniLM-L-6-v2 — cross-encoder reranking
    ↓  Llama-3.3-70b via Groq — grounded answer generation
    ↓  Streamlit — citation-backed response with source badges
```

## Key Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Embeddings | `all-MiniLM-L6-v2` | Runs locally, no API cost, 384-dim |
| Vector store | ChromaDB | Persistent, no server required |
| Keyword search | BM25 (rank-bm25) | Complements dense retrieval for exact matches |
| Fusion | Reciprocal Rank Fusion | Parameter-free, outperforms linear combination |
| Reranker | `ms-marco-MiniLM-L-6-v2` | Cross-encoder reads query+passage together |
| LLM | Llama-3.3-70b on Groq | Free tier, ~1s latency via LPU hardware |
| Evaluation | LLM-as-Judge (RAGAS-inspired) | No human labels needed |

---

## Project Structure

```
rag-pdf-assistant/
├── app.py                    # Streamlit UI
├── ingestion/
│   ├── loader.py             # PyMuPDF PDF extraction
│   ├── chunker.py            # Pure-Python recursive text splitter
│   └── embedder.py           # Sentence-transformers + ChromaDB
├── retrieval/
│   ├── retriever.py          # Dense, BM25, Hybrid+RRF, Reranked
│   └── reranker.py           # Cross-encoder reranking
├── generation/
│   └── llm_chain.py          # GroundedAnswer dataclass + Groq call
├── evaluation/
│   ├── metrics.py            # Faithfulness, Answer Relevancy, Context Relevancy
│   ├── dataset.py            # Evaluation questions
│   └── run_eval.py           # Ablation study runner
├── utils/
│   ├── config.py             # Env-based configuration
│   ├── exceptions.py         # Custom exception hierarchy
│   └── logger.py             # Structured logging
└── tests/                    # 43 pytest tests
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Rithul-Raj/rag-pdf-assistant.git
cd rag-pdf-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

Create a `.env` file:
```bash
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 4. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Evaluation

Run the ablation study (requires documents already indexed):

```bash
python -m evaluation.run_eval
```

Sample results (your numbers will vary by document):

```
Method                       | Faithfulness | Ans.Relevancy | Ctx.Relevancy
Dense only                   |        0.72  |         0.78  |         0.61
Hybrid (BM25+Dense)          |        0.79  |         0.81  |         0.74
Hybrid + Reranked            |        0.85  |         0.86  |         0.82
```

---

## Deploying to Streamlit Cloud

### Step 1 — Push to GitHub

Your code is already on GitHub at [Rithul-Raj/rag-pdf-assistant](https://github.com/Rithul-Raj/rag-pdf-assistant).

### Step 2 — Sign up for Streamlit Cloud

Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.

### Step 3 — Create a new app

Click **"New app"** and fill in:

| Field | Value |
|-------|-------|
| Repository | `Rithul-Raj/rag-pdf-assistant` |
| Branch | `main` |
| Main file path | `app.py` |

### Step 4 — Add your API key as a secret

Before clicking Deploy, click **"Advanced settings"** → **"Secrets"** and paste:

```toml
GROQ_API_KEY = "gsk_your_actual_key_here"
```

This is stored encrypted and never exposed in logs or code.

### Step 5 — Deploy

Click **"Deploy"**. First deployment takes ~3-5 minutes:
- Installing Python packages (~500 MB)
- Downloading sentence-transformer model (~90 MB)
- Downloading cross-encoder model (~85 MB)

Subsequent deploys are much faster (models are cached).

> **Note on data persistence:** ChromaDB data resets when the app restarts on Streamlit Cloud. Users need to re-upload their PDFs per session. This is expected for a demo — production systems would use a persistent vector database like Pinecone or Weaviate.

### Step 6 — Share your URL

Copy the URL (format: `https://your-name-rag-pdf-assistant-app-xxxxx.streamlit.app`) and add it to:
- Your GitHub README
- Your LinkedIn profile
- Your resume under "Projects"

---

## Tests

```bash
pytest tests/ -v
```

43 tests covering ingestion, retrieval (BM25 + hybrid + reranked), reranking, and evaluation metrics. All tests use mocks for external services.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — web UI
- **PyMuPDF** — PDF text extraction
- **sentence-transformers** — local embeddings + cross-encoder reranking
- **ChromaDB** — local vector store
- **rank-bm25** — BM25 sparse retrieval
- **Groq** — free LLM API (Llama-3.3-70b)
- **pytest** — testing

---

## License

MIT
