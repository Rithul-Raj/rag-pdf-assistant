"""
app.py
------
Streamlit UI for the Multi-Document RAG Research Assistant.

PIPELINE (end-to-end):
  1. User uploads PDF(s) in the sidebar
  2. On "Process Documents": load_pdf → chunk_documents → embed_and_store
  3. User types a question in the chat input
  4. On submit: retrieve(query) → generate_answer(query, chunks) → display

STREAMLIT STATE MODEL:
  Streamlit re-runs the entire script on every user interaction.
  `st.session_state` is a dict that persists across re-runs.
  We use it to store:
    - messages:          Chat history [{role, content, sources}]
    - processed_files:   Set of filenames already embedded
    - total_chunks:      Running count of chunks in ChromaDB

  WHY NOT RE-EMBED ON EVERY RUN?
    Because embed_and_store uses `upsert`, re-embedding the same file
    is safe — but it wastes ~5-10 seconds per file. We track processed
    filenames in session_state to skip already-embedded files.

RUN THE APP:
  cd rag-pdf-assistant
  streamlit run app.py
  Then open http://localhost:8501 in your browser.
"""

import tempfile
from pathlib import Path

import streamlit as st

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_and_store, get_collection_count
from retrieval.retriever import retrieve_reranked
from generation.llm_chain import generate_answer
from utils.exceptions import (
    DocumentLoadError,
    EmptyDocumentError,
    EmbeddingError,
    VectorStoreError,
    RetrievalError,
    RerankingError,
    GenerationError,
)
from utils.logger import get_logger

log = get_logger(__name__)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — RAG Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Imports ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ──────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App Background ──────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0d1f2d 50%, #0d1117 100%);
    color: #e6edf3;
}

/* ── Hero Header ─────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #1a2744 0%, #0e3460 50%, #1a2744 100%);
    border: 1px solid rgba(88, 166, 255, 0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
}
.hero-header h1 {
    background: linear-gradient(135deg, #58a6ff, #bf91f3, #58a6ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-header p {
    color: #8b949e;
    margin: 0.5rem 0 0;
    font-size: 1rem;
    font-weight: 400;
}

/* ── Sidebar ─────────────────────────────── */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid rgba(88,166,255,0.1) !important;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #58a6ff;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Status Card ─────────────────────────── */
.status-card {
    background: rgba(22, 27, 34, 0.8);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 10px;
    padding: 0.8rem 1.1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #8b949e;
}
.status-card strong { color: #58a6ff; }

/* ── Chat Messages ───────────────────────── */
.chat-message {
    display: flex;
    gap: 0.75rem;
    margin: 1rem 0;
    align-items: flex-start;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.chat-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.user-avatar  { background: linear-gradient(135deg, #1f6feb, #388bfd); }
.assist-avatar { background: linear-gradient(135deg, #6e40c9, #bf91f3); }

.chat-bubble {
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    max-width: 85%;
    line-height: 1.65;
    font-size: 0.95rem;
}
.user-bubble {
    background: linear-gradient(135deg, rgba(31,111,235,0.15), rgba(56,139,253,0.1));
    border: 1px solid rgba(56,139,253,0.25);
    color: #e6edf3;
}
.assist-bubble {
    background: rgba(22, 27, 34, 0.9);
    border: 1px solid rgba(110,64,201,0.25);
    color: #e6edf3;
}

/* ── Source Citation Cards ───────────────── */
.sources-container {
    margin-top: 0.75rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.source-badge {
    background: rgba(22,27,34,0.9);
    border: 1px solid rgba(88,166,255,0.2);
    border-radius: 6px;
    padding: 0.3rem 0.6rem;
    font-size: 0.78rem;
    color: #58a6ff;
    cursor: default;
}
.source-score {
    color: #3fb950;
    font-weight: 600;
}

/* ── Welcome Box ─────────────────────────── */
.welcome-box {
    text-align: center;
    padding: 3rem 2rem;
    border-radius: 16px;
    border: 2px dashed rgba(88,166,255,0.2);
    color: #8b949e;
}
.welcome-box h3 { color: #58a6ff; font-size: 1.3rem; margin-bottom: 0.5rem; }

/* ── Hide default Streamlit elements ─────── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ──────────────────────────────────────────────
def _init_state() -> None:
    """Initialise session state keys that don't yet exist."""
    defaults = {
        "messages": [],           # [{role, content, sources}]
        "processed_files": set(), # filenames already embedded
        "total_chunks": 0,        # live count from ChromaDB
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ── Helper: refresh chunk count from ChromaDB ─────────────────────────────────
def _refresh_chunk_count() -> None:
    try:
        st.session_state.total_chunks = get_collection_count()
    except Exception:
        pass  # Don't crash the UI if ChromaDB isn't initialised yet


_refresh_chunk_count()


# ── Helper: Process a single uploaded PDF ─────────────────────────────────────
def _process_pdf(uploaded_file) -> str:
    """
    Run the full ingestion pipeline on one uploaded PDF file.

    Steps:
      1. Save the in-memory upload to a temp file (PyMuPDF needs a file path)
      2. load_pdf  → extract text per page
      3. chunk_documents → split into overlapping Chunk objects
      4. embed_and_store → encode + upsert into ChromaDB

    Returns:
        A status message string for display.

    Raises:
        DocumentLoadError, EmptyDocumentError, EmbeddingError, VectorStoreError
    """
    filename = uploaded_file.name

    # Streamlit's UploadedFile is an in-memory buffer — PyMuPDF needs a
    # real file path. We save to a temp file and pass its path.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        pages = load_pdf(tmp_path)
        chunks = chunk_documents(pages, source_file=filename)
        count = embed_and_store(chunks)
        st.session_state.processed_files.add(filename)
        _refresh_chunk_count()
        return f"✅ **{filename}**: {len(pages)} pages → {count} chunks embedded"
    finally:
        # Always delete the temp file, even if processing failed.
        Path(tmp_path).unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to add to the knowledge base.",
        label_visibility="collapsed",
    )

    if st.button("⚡ Process Documents", type="primary", use_container_width=True,
                 disabled=not uploaded_files):
        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state.processed_files
        ]
        if not new_files:
            st.info("All uploaded files have already been processed.")
        else:
            with st.spinner(f"Processing {len(new_files)} file(s)..."):
                for f in new_files:
                    try:
                        msg = _process_pdf(f)
                        st.success(msg)
                    except EmptyDocumentError as e:
                        st.error(f"📭 **{f.name}**: {e}")
                    except DocumentLoadError as e:
                        st.error(f"⚠️ **{f.name}**: {e}")
                    except (EmbeddingError, VectorStoreError) as e:
                        st.error(f"🗄️ **{f.name}**: {e}")

    st.divider()

    # ── Knowledge Base Status ─────────────────────────────────────────────────
    st.markdown("## 🗄️ Knowledge Base")
    n_docs = len(st.session_state.processed_files)
    n_chunks = st.session_state.total_chunks

    col1, col2 = st.columns(2)
    col1.metric("Documents", n_docs)
    col2.metric("Chunks", n_chunks)

    if st.session_state.processed_files:
        st.markdown("**Loaded files:**")
        for fname in sorted(st.session_state.processed_files):
            st.markdown(f"- 📄 {fname}")

    st.divider()

    # ── Retrieval Settings ────────────────────────────────────────────────────
    st.markdown("## ⚙️ Settings")
    top_k = st.slider(
        "Results to retrieve (top-k)",
        min_value=1, max_value=10, value=5,
        help="How many document chunks to retrieve per question. "
             "More = richer context but slower. Try 3–7 for best results.",
    )

    st.divider()

    # ── Clear chat button ─────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════════════════

# Hero header
st.markdown("""
<div class="hero-header">
    <h1>🧠 DocMind — Research Assistant</h1>
    <p>Upload PDFs and ask questions. Answers are grounded in your documents.</p>
</div>
""", unsafe_allow_html=True)


# ── Chat History Display ──────────────────────────────────────────────────────
def _render_sources(sources: list[dict]) -> str:
    """Render source badge HTML for one assistant message."""
    if not sources:
        return ""
    badges = "".join(
        f'<span class="source-badge">📄 {s["source_file"]} '
        f'p.{s["page_number"]} '
        # Show rerank_score if available (more accurate), else cosine score
        f'<span class="source-score">{s.get("rerank_score", s["score"]):.2f}</span></span>'
        for s in sources
    )
    return f'<div class="sources-container">{badges}</div>'


chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        # Show welcome placeholder
        st.markdown("""
        <div class="welcome-box">
            <h3>👋 Welcome to DocMind</h3>
            <p>Upload your PDFs using the sidebar, then ask a question below.<br>
            Your answers will be grounded in the documents you provide.</p>
            <p style="margin-top:1rem; font-size:0.85rem; opacity:0.7">
            💡 Try: "Summarise the key findings" or "What does the paper say about X?"
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message">
                    <div class="chat-avatar user-avatar">👤</div>
                    <div class="chat-bubble user-bubble">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                sources_html = _render_sources(msg.get("sources", []))
                st.markdown(f"""
                <div class="chat-message">
                    <div class="chat-avatar assist-avatar">🧠</div>
                    <div class="chat-bubble assist-bubble">
                        {msg["content"]}
                        {sources_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ── Chat Input ────────────────────────────────────────────────────────────────
question = st.chat_input(
    "Ask a question about your documents...",
    disabled=(st.session_state.total_chunks == 0),
)

if question:
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})

    # 2. Retrieve → Generate
    with st.spinner("🔍 Searching and ranking your documents..."):
        try:
            chunks = retrieve_reranked(question, top_n=top_k)
        except (RetrievalError, RerankingError) as e:
            st.error(f"Retrieval failed: {e}")
            chunks = []

    if chunks:
        with st.spinner("✍️ Generating answer..."):
            try:
                answer = generate_answer(question, chunks)
            except GenerationError as e:
                answer = f"⚠️ Generation failed: {e}"

        # Attach top sources to the assistant message
        sources = [
            {"source_file": c["source_file"],
             "page_number": c["page_number"],
             "score": c["score"]}
            for c in chunks[:3]  # show top 3 sources
        ]
    else:
        answer = (
            "I couldn't find relevant information in the knowledge base "
            "to answer that question."
        )
        sources = []

    # 3. Add assistant message and rerun to render it
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.rerun()
