import os

from google import genai

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

_client = None


def _get_client():
    global _client
    if _client is None:
        # FIX: call get_secret lazily inside _get_client so the module can be
        # imported without a network call, and key rotation is picked up after
        # a container restart.
        from .secrets import get_secret

        api_key = get_secret("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured. Check secrets.")
        _client = genai.Client(api_key=api_key)
    return _client


def get_embedding(text: str) -> list[float]:
    if not text:
        return []

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
        raise RuntimeError(f"Embedding failed: {e}")
