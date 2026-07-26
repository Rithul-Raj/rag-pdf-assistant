"""
retrieval/retriever.py
-----------------------
Dense similarity search against ChromaDB.

WHERE THIS FITS:
  User Query
    → retriever.py (dense search)         ← THIS FILE (Step 4)
    → retriever.py (+ BM25 hybrid)        ← Step 5 adds this
    → reranker.py (cross-encoder)         ← Step 6 adds this
    → llm_chain.py (grounded generation)  ← Step 7 adds this

WHY NOT JUST CALL query_similar() DIRECTLY IN app.py?
  Two reasons:
  1. Separation of concerns — app.py should not know about ChromaDB
     internals. It just asks "retrieve for this query" and gets clean dicts.
  2. Step 5 will add BM25 here and Step 6 will add reranking. Having a
     dedicated retriever module means app.py never changes — only this
     file grows.

RESULT FORMAT:
  Each result is a plain dict:
  {
    "text":        str   — the chunk text
    "source_file": str   — e.g. "paper.pdf"
    "page_number": int   — 1-indexed
    "chunk_id":    str   — e.g. "paper.pdf::p3::c17"
    "score":       float — cosine similarity 0.0→1.0 (higher = more similar)
  }
  Using plain dicts (not dataclasses) because they serialise easily to JSON
  and can be passed directly to Streamlit components.
"""

from __future__ import annotations

from typing import List

from ingestion.embedder import query_similar
from utils.config import config
from utils.exceptions import RetrievalError
from utils.logger import get_logger

log = get_logger(__name__)


def retrieve(query: str, top_k: int | None = None) -> List[dict]:
    """
    Retrieve the top-k most semantically relevant chunks for a query.

    This is pure dense (cosine-similarity) retrieval against ChromaDB.
    Hybrid retrieval (dense + BM25) and reranking are added in Steps 5 & 6.

    Args:
        query:  The user's question or search string.
        top_k:  Number of results to return. Defaults to config.top_k (5).

    Returns:
        List of result dicts, sorted from most to least similar.
        Each dict has: text, source_file, page_number, chunk_id, score.

    Raises:
        RetrievalError: If ChromaDB is empty or the query itself fails.
    """
    k = top_k if top_k is not None else config.top_k
    log.info("Dense retrieval: query='%s...', top_k=%d", query[:60], k)

    try:
        raw = query_similar(query, top_k=k)
    except Exception as exc:
        # Wrap all embedder/ChromaDB exceptions in RetrievalError so
        # callers only need to catch one exception type.
        raise RetrievalError(f"Dense retrieval failed: {exc}") from exc

    results = _parse_chroma_results(raw)
    log.info("Retrieved %d chunks.", len(results))
    return results


def _parse_chroma_results(raw: dict) -> List[dict]:
    """
    Convert ChromaDB's nested-list result format into a flat list of dicts.

    ChromaDB returns results as:
      { 'documents': [[ chunk1, chunk2, ... ]],  ← outer list = per query
        'metadatas':  [[ {...}, {...}, ... ]],
        'distances':  [[ 0.12, 0.34, ... ]] }    ← cosine DISTANCE (0=same)

    We want: [{ text, source_file, page_number, chunk_id, score }, ...]

    WHY CONVERT DISTANCE TO SCORE?
      ChromaDB with hnsw:space=cosine returns *distance* (lower=better).
      For cosine: distance = 1 - similarity. So similarity = 1 - distance.
      We expose `score` (higher=better) for human readability in the UI.

    Args:
        raw: Raw ChromaDB query result dict.

    Returns:
        Flat list of result dicts, most similar first.
    """
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    results: List[dict] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        results.append(
            {
                "text": doc,
                "source_file": meta.get("source_file", "unknown"),
                "page_number": meta.get("page_number", 0),
                "chunk_id": meta.get("chunk_id", ""),
                # Convert distance → similarity score (0.0 to 1.0)
                "score": round(max(0.0, 1.0 - dist), 4),
            }
        )

    return results
