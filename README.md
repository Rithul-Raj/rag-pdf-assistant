# Multi-Document RAG Research Assistant

A "chat with your PDFs" application built as a learning project demonstrating **Retrieval-Augmented Generation (RAG)** — the dominant architecture for grounding LLM answers in private documents.

## Architecture Overview

```
PDF Upload → Text Extraction → Chunking → Embedding → ChromaDB
                                                          ↓
User Query → BM25 + Dense Retrieval → Reranking → Context → Groq LLM → Answer (with citations)
```

### Key Design Decisions (Interview-Ready)

| Decision | Choice | Alternative Rejected | Reason |
|----------|--------|----------------------|--------|
| Embeddings | `all-MiniLM-L6-v2` | `all-mpnet-base-v2` | 5× faster, 80 MB vs 420 MB, small quality tradeoff |
| Vector DB | ChromaDB | FAISS | Persistent, metadata-aware, no server needed |
| Retrieval | Hybrid (BM25 + dense) | Pure dense | Keyword queries beat semantic search for exact terms |
| Reranking | Cross-encoder | Larger bi-encoder | Cross-encoders have full query-doc attention; bi-encoders are fast but less accurate |
| LLM | Groq (free tier) | OpenAI | Free, fast, no credit card required |
| Temperature | 0 | >0 | Factual RAG benefits from determinism over creativity |

## Setup

### 1. Clone and install
```bash
git clone <your-repo-url>
cd rag-pdf-assistant
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get a free key at https://console.groq.com
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Run tests
```bash
pytest tests/ -v
```

## Project Structure

```
rag-pdf-assistant/
├── utils/
│   ├── config.py        # Centralized config (fail-fast validation)
│   ├── exceptions.py    # Custom exception hierarchy
│   └── logger.py        # Shared logger factory
├── ingestion/
│   ├── loader.py        # PDF → (page_num, text) via PyMuPDF
│   ├── chunker.py       # Text → overlapping Chunk objects
│   └── embedder.py      # Chunks → ChromaDB vectors
├── retrieval/
│   ├── retriever.py     # Hybrid BM25 + dense search
│   └── reranker.py      # Cross-encoder reranking
├── generation/
│   └── llm_chain.py     # Prompt template + Groq LLM call
├── evaluation/
│   └── ragas_eval.py    # RAGAS metrics (faithfulness, relevance)
├── tests/
│   ├── test_ingestion.py
│   ├── test_embedder.py
│   └── test_retrieval.py
├── app.py               # Streamlit UI
├── requirements.txt
├── .env.example
└── .gitignore
```

## Pipeline Steps (What Each Stage Does)

1. **Ingestion** — PyMuPDF extracts text page-by-page. Each page gets a page number for citations.
2. **Chunking** — Recursive character splitting at 800 chars / 150 overlap. Metadata (source, page) travels with each chunk.
3. **Embedding** — `sentence-transformers` encodes each chunk into a 384-dimensional dense vector. ChromaDB stores them with metadata.
4. **Retrieval** — Hybrid: BM25 keyword search + dense cosine similarity. Scores fused with Reciprocal Rank Fusion (RRF).
5. **Reranking** — Cross-encoder scores each (query, chunk) pair and re-orders results.
6. **Generation** — Top-3 chunks + query → Groq LLM. Prompt instructs model to answer only from context and cite sources.

## Evaluation Results

| Metric | Dense Only | Hybrid + Reranked |
|--------|------------|-------------------|
| Faithfulness | TBD | TBD |
| Answer Relevance | TBD | TBD |
| Context Precision | TBD | TBD |

*(Results populated after running `python evaluation/ragas_eval.py`)*

## Technologies Used

- **PyMuPDF** — PDF parsing
- **LangChain text splitters** — Recursive character chunking
- **sentence-transformers** — Local embeddings (no API cost)
- **ChromaDB** — Local vector database with persistence
- **rank-bm25** — BM25 sparse retrieval
- **Groq** — Free-tier LLM inference (llama-3.3-70b-versatile)
- **RAGAS** — RAG evaluation framework
- **Streamlit** — Web UI
