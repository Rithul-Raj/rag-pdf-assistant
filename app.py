"""
app.py
------
Streamlit UI for the Multi-Document RAG Research Assistant.

PIPELINE (end-to-end):
  1. User uploads PDF(s) in the sidebar
  2. On "Process Documents": load_pdf → chunk_documents → embed_and_store
  3. User types a question in the chat input
  4. On submit: retrieve_reranked(query) → generate_answer(query, chunks) → display

STREAMLIT STATE MODEL:
  Streamlit re-runs the entire script on every user interaction.
  `st.session_state` is a dict that persists across re-runs.
  We use it to store:
    - messages:          Chat history [{role, content, sources, is_grounded}]
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
from generation.llm_chain import generate_answer, GroundedAnswer
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
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ─────────────────────────────────────────────────────────────
# Color palette (4 solid colors + neutrals — nothing more)
#   BG:       #111318  very dark charcoal, main canvas
#   SURFACE:  #1a1d24  card/panel backgrounds
#   BORDER:   #2a2d38  all borders
#   ACCENT:   #4361ee  one strong blue-indigo, used sparingly
#   TEXT:     #dde1ee  primary text
#   MUTED:    #6b7080  secondary / metadata text
#   OK:       #2e7d52  grounded indicator (solid green, not neon)
#   WARN:     #8b2e2e  not-found indicator (solid red, not neon)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ─────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 15px;
}

.stApp {
    background-color: #111318;
    color: #dde1ee;
}

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0e1015 !important;
    border-right: 1px solid #2a2d38 !important;
}

[data-testid="stSidebar"] .stMarkdown h2 {
    color: #6b7080;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* ── Sidebar buttons ──────────────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    background-color: #4361ee;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-weight: 500;
    font-size: 0.88rem;
    letter-spacing: 0.01em;
    padding: 0.55rem 1rem;
    transition: background-color 0.15s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #3451d1;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background-color: transparent;
    color: #6b7080;
    border: 1px solid #2a2d38;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    border-color: #4361ee;
    color: #dde1ee;
}

/* ── Metrics ──────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: #1a1d24;
    border: 1px solid #2a2d38;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
}
[data-testid="stMetricLabel"] { color: #6b7080; font-size: 0.78rem; }
[data-testid="stMetricValue"] { color: #dde1ee; font-size: 1.4rem; font-weight: 600; }

/* ── Slider ───────────────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div > div {
    background-color: #4361ee;
}

/* ── File uploader ────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1px dashed #2a2d38;
    border-radius: 8px;
    padding: 0.5rem;
}

/* ── Page header ──────────────────────────────────────────────── */
.page-header {
    padding: 1.4rem 0 1rem;
    border-bottom: 1px solid #2a2d38;
    margin-bottom: 1.4rem;
}
.page-header h1 {
    font-size: 1.35rem;
    font-weight: 600;
    color: #dde1ee;
    margin: 0;
    letter-spacing: -0.2px;
}
.page-header p {
    color: #6b7080;
    font-size: 0.85rem;
    margin: 0.3rem 0 0;
}

/* ── Chat message rows ────────────────────────────────────────── */
.chat-row {
    display: flex;
    gap: 0.65rem;
    margin: 0.85rem 0;
    align-items: flex-start;
}
.chat-row.user-row {
    flex-direction: row-reverse;
}

/* ── Sender label ─────────────────────────────────────────────── */
.sender-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7080;
    width: 38px;
    text-align: center;
    flex-shrink: 0;
    padding-top: 0.55rem;
    line-height: 1.2;
}
.user-row .sender-label  { color: #4361ee; }
.assist-row .sender-label { color: #6b7080; }

/* ── Chat bubbles ─────────────────────────────────────────────── */
.chat-bubble {
    border-radius: 8px;
    padding: 0.8rem 1.05rem;
    max-width: 82%;
    line-height: 1.7;
    font-size: 0.92rem;
}
.user-bubble {
    background-color: #1c2140;
    border: 1px solid #2d3260;
    color: #dde1ee;
}
.assist-bubble {
    background-color: #1a1d24;
    border: 1px solid #2a2d38;
    color: #dde1ee;
}

/* ── Source badges ────────────────────────────────────────────── */
.sources-row {
    margin-top: 0.6rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    align-items: center;
}
.tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-radius: 4px;
    padding: 0.18rem 0.5rem;
    border: 1px solid;
}
.tag-grounded {
    background-color: #152b1e;
    border-color: #2e7d52;
    color: #4caf77;
}
.tag-notfound {
    background-color: #2b1515;
    border-color: #8b2e2e;
    color: #e06060;
}
.source-chip {
    display: inline-block;
    background-color: #1a1d24;
    border: 1px solid #2a2d38;
    border-radius: 4px;
    padding: 0.18rem 0.55rem;
    font-size: 0.75rem;
    color: #8b95b5;
    font-family: 'JetBrains Mono', monospace;
}
.source-chip .score {
    color: #4361ee;
    font-weight: 600;
    margin-left: 0.35rem;
}

/* ── Welcome state ────────────────────────────────────────────── */
.welcome-state {
    margin: 3.5rem auto;
    max-width: 500px;
    text-align: center;
    color: #6b7080;
}
.welcome-state h2 {
    color: #dde1ee;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.welcome-state p {
    font-size: 0.88rem;
    line-height: 1.7;
    margin: 0;
}
.welcome-hint {
    display: inline-block;
    margin-top: 1.2rem;
    background-color: #1a1d24;
    border: 1px solid #2a2d38;
    border-radius: 6px;
    padding: 0.5rem 0.9rem;
    font-size: 0.82rem;
    color: #6b7080;
    font-style: italic;
}

/* ── Spinner / input ──────────────────────────────────────────── */
.stChatInput textarea {
    background-color: #1a1d24 !important;
    border: 1px solid #2a2d38 !important;
    color: #dde1ee !important;
    border-radius: 8px !important;
}
.stChatInput textarea:focus {
    border-color: #4361ee !important;
}

/* ── Expander (citation cards) ────────────────────────────────── */
.streamlit-expanderHeader {
    background-color: #1a1d24 !important;
    border: 1px solid #2a2d38 !important;
    border-radius: 6px !important;
    color: #6b7080 !important;
    font-size: 0.82rem !important;
}

/* ── Dividers ─────────────────────────────────────────────────── */
hr { border-color: #2a2d38 !important; }

/* ── Hide Streamlit chrome ────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; max-width: 860px; }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ──────────────────────────────────────────────
def _init_state() -> None:
    """Initialise session state keys that don't yet exist."""
    defaults = {
        "messages": [],           # [{role, content, sources, is_grounded}]
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
        return f"**{filename}** — {len(pages)} pages, {count} chunks indexed"
    finally:
        # Always delete the temp file, even if processing failed.
        Path(tmp_path).unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to add to the knowledge base.",
        label_visibility="collapsed",
    )

    if st.button("Process Documents", type="primary", use_container_width=True,
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
                        st.error(f"**{f.name}** — empty document: {e}")
                    except DocumentLoadError as e:
                        st.error(f"**{f.name}** — load error: {e}")
                    except (EmbeddingError, VectorStoreError) as e:
                        st.error(f"**{f.name}** — storage error: {e}")

    st.divider()

    # ── Knowledge Base Status ─────────────────────────────────────────────────
    st.markdown("## Knowledge Base")
    n_docs = len(st.session_state.processed_files)
    n_chunks = st.session_state.total_chunks

    col1, col2 = st.columns(2)
    col1.metric("Documents", n_docs)
    col2.metric("Chunks", n_chunks)

    if st.session_state.processed_files:
        st.markdown("**Indexed files**")
        for fname in sorted(st.session_state.processed_files):
            st.markdown(
                f'<div style="font-size:0.8rem;color:#6b7080;padding:0.15rem 0;'
                f'font-family:\'JetBrains Mono\',monospace;word-break:break-all;">'
                f'{fname}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Retrieval Settings ────────────────────────────────────────────────────
    st.markdown("## Settings")
    top_k = st.slider(
        "Results to retrieve",
        min_value=1, max_value=10, value=5,
        help="Number of document chunks retrieved per question. "
             "Higher = more context, slightly slower. Recommended: 3–5.",
    )

    st.divider()

    # ── Clear chat ────────────────────────────────────────────────────────────
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════════════════

# Page header — plain text, no decoration
st.markdown("""
<div class="page-header">
    <h1>DocMind</h1>
    <p>Ask questions about your documents. Answers are sourced and verifiable.</p>
</div>
""", unsafe_allow_html=True)


# ── Chat History Display ──────────────────────────────────────────────────────
def _render_sources(sources: list[dict], is_grounded: bool = True) -> str:
    """Render grounding tag + source chips. No emoji — text labels only."""
    parts = []

    if is_grounded and sources:
        parts.append('<span class="tag tag-grounded">verified</span>')
    elif not is_grounded:
        parts.append('<span class="tag tag-notfound">not in documents</span>')

    for s in sources:
        score = s.get("rerank_score", s.get("score", 0))
        parts.append(
            f'<span class="source-chip">'
            f'{s["source_file"]} &middot; p.{s["page_number"]}'
            f'<span class="score">{score:.2f}</span>'
            f'</span>'
        )

    if not parts:
        return ""
    return f'<div class="sources-row">{"".join(parts)}</div>'


chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-state">
            <h2>No conversation yet</h2>
            <p>Upload a PDF using the sidebar, click <strong>Process Documents</strong>,
            then type your question below.</p>
            <span class="welcome-hint">Try: "What is the main argument of this paper?"</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-row user-row">
                    <span class="sender-label">You</span>
                    <div class="chat-bubble user-bubble">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                msg_sources  = msg.get("sources", [])
                msg_grounded = msg.get("is_grounded", True)
                sources_html = _render_sources(msg_sources, msg_grounded)
                st.markdown(f"""
                <div class="chat-row assist-row">
                    <span class="sender-label">RAG</span>
                    <div class="chat-bubble assist-bubble">
                        {msg["content"]}
                        {sources_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Expandable citation cards — plain text, no emoji
                if msg_sources and msg_grounded:
                    with st.expander(
                        f"View {len(msg_sources)} source excerpt(s)", expanded=False
                    ):
                        for i, src in enumerate(msg_sources, 1):
                            score = src.get("rerank_score", src.get("score", 0))
                            st.markdown(
                                f"**Excerpt {i}** &nbsp;·&nbsp; "
                                f"`{src['source_file']}` &nbsp;·&nbsp; "
                                f"Page {src['page_number']} &nbsp;·&nbsp; "
                                f"score: `{score:.3f}`"
                            )
                            text = src.get("text", "")
                            st.caption(
                                text[:500] + ("..." if len(text) > 500 else "")
                            )
                            if i < len(msg_sources):
                                st.divider()


# ── Chat Input ────────────────────────────────────────────────────────────────
question = st.chat_input(
    "Ask a question about your documents...",
    disabled=(st.session_state.total_chunks == 0),
)

if question:
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})

    # 2. Retrieve → Generate
    with st.spinner("Searching documents..."):
        try:
            chunks = retrieve_reranked(question, top_n=top_k)
        except (RetrievalError, RerankingError) as e:
            st.error(f"Retrieval failed: {e}")
            chunks = []

    if chunks:
        with st.spinner("Generating answer..."):
            try:
                grounded: GroundedAnswer = generate_answer(question, chunks)
            except GenerationError as e:
                # Store error as assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Generation failed: {e}",
                    "sources": [],
                    "is_grounded": False,
                })
                st.rerun()

        # Build the sources list from GroundedAnswer.citations
        # Citations already have text, source_file, page_number, rerank_score
        sources = [
            {
                "source_file": c["source_file"],
                "page_number":  c["page_number"],
                "text":         c.get("text", ""),
                "rerank_score": c.get("rerank_score", c.get("score", 0)),
                "score":        c.get("score", 0),
            }
            for c in grounded.citations
        ]
        answer_text = grounded.answer
        is_grounded = grounded.is_grounded
    else:
        answer_text = (
            "I don't know based on the provided documents. "
            "No relevant excerpts were found."
        )
        sources     = []
        is_grounded = False

    # 3. Add assistant message and rerun to render it
    st.session_state.messages.append({
        "role":        "assistant",
        "content":     answer_text,
        "sources":     sources,
        "is_grounded": is_grounded,
    })
    st.rerun()
