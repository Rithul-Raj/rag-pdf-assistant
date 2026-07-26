"""
ingestion/chunker.py
--------------------
Text splitting into overlapping chunks with metadata preservation.

WHY CHUNK AT ALL?
  A 200-page research paper might be 500,000 characters. The LLM's
  context window is 4,000–32,000 tokens. We cannot send the whole
  document. Instead, we:
    1. Split the document into small, manageable chunks.
    2. Embed each chunk as a vector.
    3. At query time, retrieve only the top-K most relevant chunks.
    4. Send only those chunks to the LLM.
  This is the fundamental trick that makes RAG scalable.

WHY RECURSIVE CHARACTER SPLITTING?
  LangChain's RecursiveCharacterTextSplitter tries a cascade of
  separators in order: ["\n\n", "\n", " ", ""] — i.e., it prefers to
  split at paragraph breaks, then at newlines, then at spaces, and only
  as a last resort splits mid-word. This preserves semantic coherence
  much better than naive fixed-width splitting.

  ALTERNATIVE CONSIDERED: Sentence-level splitting (e.g., NLTK's
  sent_tokenize). Rejected because sentence tokenizers are slow, add a
  dependency, and handle tables/bullet points poorly. Recursive character
  splitting is faster and good enough for our use case.

WHY CHUNK_SIZE=800 chars / CHUNK_OVERLAP=150 chars?
  800 chars ≈ 150–200 tokens (rough rule: 1 token ≈ 4 chars).
  - Large enough to capture a complete thought or paragraph.
  - Small enough that a chunk doesn't dominate the LLM context.
  - 150-char overlap ensures answers at chunk boundaries aren't lost.
  These are defaults from config.py and can be tuned via .env.

METADATA ATTACHED TO EVERY CHUNK:
  source_file : str   — original PDF filename
  page_number : int   — which page the chunk originated from
  chunk_id    : str   — globally unique ID: "{filename}::p{page}::c{n}"
  
  This metadata travels with the chunk through embedding and retrieval,
  so the final answer can say "Source: report.pdf, page 12".
"""

from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import config
from utils.exceptions import ChunkingError
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Chunk:
    """
    A single piece of text with its origin metadata.

    Using a dataclass (instead of a plain dict) gives us:
      - Type hints on every field
      - A readable __repr__ for debugging
      - Dot-attribute access (chunk.text, not chunk["text"])
    """
    text: str
    source_file: str
    page_number: int
    chunk_id: str


def chunk_documents(
    pages: List[tuple[int, str]],
    source_file: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Chunk]:
    """
    Split page texts into overlapping chunks and attach metadata.

    Args:
        pages:         List of (page_number, text) tuples from loader.py.
        source_file:   The original PDF filename (used in citations).
        chunk_size:    Override the default from config (useful for tests).
        chunk_overlap: Override the default from config.

    Returns:
        A flat list of Chunk objects, ordered by page then position.

    Raises:
        ChunkingError: If the splitter fails or `pages` is empty.
        ValueError:    If chunk_overlap >= chunk_size.
    """
    if not pages:
        raise ChunkingError(
            "Cannot chunk an empty document. "
            "Make sure load_pdf() returned at least one page with text."
        )

    # Use config defaults unless explicit overrides are provided.
    # This pattern (param or config.default) lets tests pass small values
    # without modifying the global config object.
    size = chunk_size if chunk_size is not None else config.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap

    if overlap >= size:
        raise ValueError(
            f"chunk_overlap ({overlap}) must be less than chunk_size ({size}). "
            "Otherwise chunks have no unique content — the overlap would cover "
            "the entire chunk."
        )

    log.info(
        "Chunking '%s': %d pages, chunk_size=%d, chunk_overlap=%d",
        source_file,
        len(pages),
        size,
        overlap,
    )

    # ── Initialize the splitter ───────────────────────────────────────────────
    # RecursiveCharacterTextSplitter cascades through separators:
    # ["\n\n", "\n", " ", ""] — always prefers the largest natural boundary.
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,  # measure chunk size in characters, not tokens
            # separators default is ["\n\n", "\n", " ", ""] which is exactly
            # what we want — no need to override.
        )
    except Exception as exc:
        raise ChunkingError(
            f"Failed to initialize text splitter: {exc}"
        ) from exc

    # ── Process each page ─────────────────────────────────────────────────────
    all_chunks: List[Chunk] = []
    chunk_counter = 0  # global counter across all pages for unique chunk IDs

    for page_number, page_text in pages:
        try:
            raw_chunks: List[str] = splitter.split_text(page_text)
        except Exception as exc:
            raise ChunkingError(
                f"Failed to split text on page {page_number} of '{source_file}': {exc}"
            ) from exc

        for raw_chunk in raw_chunks:
            stripped = raw_chunk.strip()
            if not stripped:
                # Skip chunks that are only whitespace — can happen when a
                # page has long runs of blank lines between sections.
                continue

            # Chunk ID format: "report.pdf::p3::c17"
            # Using "::" as separator avoids conflicts with typical filenames.
            chunk_id = f"{source_file}::p{page_number}::c{chunk_counter}"
            chunk_counter += 1

            all_chunks.append(
                Chunk(
                    text=stripped,
                    source_file=source_file,
                    page_number=page_number,
                    chunk_id=chunk_id,
                )
            )

    log.info(
        "Chunked '%s' into %d chunks across %d pages.",
        source_file,
        len(all_chunks),
        len(pages),
    )

    if not all_chunks:
        raise ChunkingError(
            f"Chunking produced zero chunks for '{source_file}'. "
            "The document may contain only whitespace."
        )

    return all_chunks
