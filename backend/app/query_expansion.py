"""
Query Expansion via HyDE (Hypothetical Document Embeddings).

Instead of embedding the raw user query — which is short, ambiguous, and
stylistically different from document chunks — we ask the LLM to write a
hypothetical answer, then embed THAT. The embedding of a plausible answer
sits much closer in vector space to real answer chunks than a question does.

Flow:
    user query
        │
        ▼
    generate_hyde_query()
        │  calls Gemini to write a short hypothetical answer
        ▼
    expanded query text
        │
        ▼
    get_embedding()  ← embed the hypothesis, not the original question
        │
        ▼
    vector_search()

Fallback:   if LLM call fails for any reason, the original query is
            returned unchanged so retrieval still works.

Env vars:
    HYDE_ENABLED        "true" / "false"  (default: "true")
    HYDE_MAX_TOKENS     token budget for the hypothesis  (default: 150)
"""

from __future__ import annotations

import logging
import os

from .gemini_client import GEMINI_MODEL, _get_client

logger = logging.getLogger(__name__)

HYDE_ENABLED: bool = os.getenv("HYDE_ENABLED", "true").lower() == "true"
HYDE_MAX_TOKENS: int = int(os.getenv("HYDE_MAX_TOKENS", "150"))

_HYDE_PROMPT = (
    "Write a short, factual passage (2-4 sentences) that directly answers "
    "the following question. Do not add preamble, caveats, or formatting. "
    "Write as if you are a document that contains the answer.\n\n"
    "Question: {query}\n\n"
    "Passage:"
)


def generate_hyde_query(query: str) -> str:
    """
    Return a hypothetical answer passage for the query.

    If HYDE_ENABLED is False, or if the LLM call fails, returns the
    original query unchanged so the pipeline degrades gracefully.

    Args:
        query: Raw user question.

    Returns:
        Hypothetical answer text (for embedding) or original query on failure.
    """
    if not HYDE_ENABLED:
        return query

    if not query or not query.strip():
        return query

    try:
        from google.genai import types

        prompt = _HYDE_PROMPT.format(query=query.strip())
        response = _get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=HYDE_MAX_TOKENS,
            ),
        )
        hypothesis = (response.text or "").strip()
        if hypothesis:
            logger.debug(f"HyDE expansion: '{query[:40]}' → '{hypothesis[:60]}...'")
            return hypothesis

        logger.warning("HyDE returned empty response, using original query")
        return query

    except Exception as e:
        logger.warning(f"HyDE expansion failed, using original query: {e}")
        return query
