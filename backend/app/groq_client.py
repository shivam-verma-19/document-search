"""
Groq LLM Client supporting free, high-speed open models (e.g. llama-3.3-70b-versatile).

Configured via:
    LLM_BACKEND="groq"
    GROQ_API_KEY="gsk_..."
    GROQ_MODEL="llama-3.3-70b-versatile" (default)
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_client = None


def _get_client():
    global _client
    if _client is None:
        from groq import Groq

        from .secrets import get_secret

        api_key = get_secret("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not configured. Set GROQ_API_KEY in environment or secrets.")
        _client = Groq(api_key=api_key)
    return _client


def route_and_invoke(prompt: str, query: str = "", context: str = "") -> dict:
    """
    Invoke Groq model and return standardized response dictionary compatible with RouterResult.
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
        )

        answer = ""
        if response.choices and len(response.choices) > 0:
            answer = (response.choices[0].message.content or "").strip()

        # Classify complexity simply from query length / tokens
        is_simple = len((query or "").split()) < 8
        complexity = "simple" if is_simple else "complex"

        return {
            "answer": answer,
            "model_used": GROQ_MODEL,
            "complexity": complexity,
            "confidence": 0.90,
            "escalated": False,
            "attempted": [GROQ_MODEL],
            "errors": {},
        }
    except Exception as e:
        logger.error(f"Groq invocation failed: {e}")
        raise RuntimeError(f"Groq invocation failed: {e}") from e
