"""
generation/llm_chain.py
------------------------
Basic LLM generation using the Groq API (free tier).

WHAT IS "GROUNDING"?
  A "grounded" LLM answer is one constrained to only use the provided
  context — it cannot hallucinate facts from its training data. We
  achieve this through prompt engineering: the system prompt explicitly
  tells the model to answer ONLY from the provided context and to say
  "I don't know" if the answer isn't there.

  THIS IS A BASIC v1. Step 7 will add:
    - Stricter "I don't know" instructions
    - Citation formatting (source file + page number in the answer)
    - Fallback to Ollama when Groq is unavailable

GROQ FREE TIER:
  As of 2025, Groq offers generous free-tier rate limits:
  - 6,000 tokens/minute on most models
  - 500 requests/day on free tier
  - No credit card required — just a free account at console.groq.com
  
  We use llama-3.3-70b-versatile: excellent quality, widely available
  on Groq's free tier. Check console.groq.com/docs/models for current
  availability since free models change.

TEMPERATURE = 0:
  We set temperature=0 (fully deterministic). For a RAG assistant,
  consistency matters more than creativity — you want the same question
  to always get the same answer from the same context.

  ALTERNATIVE: temperature=0.1 adds just enough variation to avoid
  robotic-sounding responses while remaining mostly deterministic.
  We'll expose this as a config option in Step 7.
"""

from __future__ import annotations

from typing import List

from groq import Groq

from utils.config import config
from utils.exceptions import GenerationError
from utils.logger import get_logger

log = get_logger(__name__)

# ── Client singleton ──────────────────────────────────────────────────────────
# Same pattern as embedder.py — create once, reuse across calls.
_client: Groq | None = None


def _get_client() -> Groq:
    """
    Lazily initialise the Groq client on first call.

    Raises:
        GenerationError: If GROQ_API_KEY is not set.
    """
    global _client
    if _client is None:
        if not config.groq_api_key:
            raise GenerationError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com and add it to your .env file."
            )
        _client = Groq(api_key=config.groq_api_key)
        log.info("Groq client initialised (model: %s).", config.groq_model)
    return _client


# ── Prompt Templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a research assistant that answers questions \
based ONLY on the provided document excerpts.

Rules:
1. Answer using ONLY the information in the excerpts below.
2. If the answer is not found in the excerpts, say exactly: \
"I'm sorry, I couldn't find the answer to that in the provided documents."
3. Be concise and factual.
4. When relevant, mention which document the information came from."""

_CONTEXT_TEMPLATE = """Document Excerpts:
{context}

---
Question: {question}

Answer:"""


def _build_context(context_chunks: List[dict]) -> str:
    """
    Format a list of retrieved chunk dicts into a single context string.

    Each chunk is formatted as:
      [Source: filename.pdf, Page 3]
      <chunk text>

    This format lets the LLM reference sources in its answer,
    which we'll use in Step 7 to extract and display proper citations.

    Args:
        context_chunks: List of dicts from retriever.retrieve().

    Returns:
        A single formatted string ready to insert into the prompt.
    """
    parts = []
    for i, chunk in enumerate(context_chunks, 1):
        header = (
            f"[Excerpt {i} — Source: {chunk['source_file']}, "
            f"Page {chunk['page_number']}]"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


def generate_answer(question: str, context_chunks: List[dict]) -> str:
    """
    Generate an answer grounded in the retrieved context chunks.

    FLOW:
      1. Format context chunks into a structured string.
      2. Build a prompt that instructs the model to only use that context.
      3. Call Groq API with temperature=0 for deterministic output.
      4. Return the model's answer text.

    Args:
        question:       The user's question.
        context_chunks: List of dicts from retriever.retrieve().
                        Each dict has: text, source_file, page_number.

    Returns:
        The model's answer as a plain string.

    Raises:
        GenerationError: If the Groq API call fails.
    """
    if not context_chunks:
        return (
            "I'm sorry, I couldn't find any relevant information in the "
            "uploaded documents to answer your question."
        )

    client = _get_client()
    context = _build_context(context_chunks)
    user_message = _CONTEXT_TEMPLATE.format(
        context=context,
        question=question,
    )

    log.info(
        "Calling Groq (%s) with %d context chunks...",
        config.groq_model,
        len(context_chunks),
    )

    try:
        response = client.chat.completions.create(
            model=config.groq_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=config.llm_temperature,
            max_tokens=config.max_tokens,
        )
        answer = response.choices[0].message.content.strip()
        log.info("Generation complete (%d tokens used).",
                 response.usage.total_tokens if response.usage else 0)
        return answer

    except Exception as exc:
        raise GenerationError(
            f"Groq API call failed: {exc}. "
            "Check your API key and network connection."
        ) from exc
