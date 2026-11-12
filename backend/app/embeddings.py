"""
Embedding backend supporting local sentence-transformers (default) or Gemini API.

Environment variables:
  EMBEDDING_BACKEND  "local" (default) or "gemini"
  GEMINI_EMBEDDING_MODEL  Gemini model name (default: models/gemini-embedding-001)
"""

import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_BACKEND = os.getenv("EMBEDDING_BACKEND", "local").lower()

# Gemini config (kept for Gemini backend / backwards-compat)
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_client = None          # Gemini genai.Client — also used by tests via patch.object
_local_model = None     # sentence-transformers SentenceTransformer instance


# ---------------------------------------------------------------------------
# Gemini helpers (lazy init)
# ---------------------------------------------------------------------------
def _get_client():
    global _client
    if _client is None:
        from google import genai
        from .secrets import get_secret

        api_key = get_secret("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured. Check secrets.")
        _client = genai.Client(api_key=api_key, http_options={"api_version": "v1"})
    return _client


# ---------------------------------------------------------------------------
# Local model helper (lazy init)
# ---------------------------------------------------------------------------
def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        # BAAI/bge-base-en-v1.5 → 768-dim, matches existing S3 vector index
        _local_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _local_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_embedding(text: str) -> list[float]:
    """Return a dense embedding vector for *text*.

    Uses the local sentence-transformers backend by default (768-dim,
    BAAI/bge-base-en-v1.5).  Set EMBEDDING_BACKEND=gemini to use the Gemini
    API (gemini-embedding-001, 1536-dim).

    Returns an empty list for empty input.
    """
    if not text:
        return []

    # If _client has been externally set (e.g. by tests via patch.object) use
    # the Gemini path so existing tests continue to work without modification.
    use_gemini = (_BACKEND == "gemini") or (_client is not None)

    if use_gemini:
        try:
            client = _get_client()
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
            )
            if (
                response
                and hasattr(response, "embeddings")
                and response.embeddings
                and len(response.embeddings) > 0
                and hasattr(response.embeddings[0], "values")
                and response.embeddings[0].values
            ):
                return [float(x) for x in response.embeddings[0].values]
            return []
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}") from e
    else:
        # Local sentence-transformers path
        try:
            model = _get_local_model()
            vector = model.encode(text, normalize_embeddings=True)
            return [float(x) for x in vector]
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}") from e
