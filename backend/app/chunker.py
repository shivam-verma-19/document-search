"""
Text chunker with metadata preservation and semantic boundary detection.

Two strategies available via CHUNKING_STRATEGY env var:

  "recursive" (default) — RecursiveCharacterTextSplitter with paragraph/sentence
                           separators. Fast, no LLM calls, works on any text.

  "semantic"            — Splits on sentence boundaries and merges sentences into
                           chunks only when they are semantically similar (cosine
                           similarity above SEMANTIC_SIMILARITY_THRESHOLD).
                           Prevents splitting in the middle of a logical argument.
                           Slower (requires embedding each sentence), best for
                           dense technical or legal documents.

Both strategies return List[Chunk] with full source metadata so attribution
is preserved all the way to the LLM prompt.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNKING_STRATEGY = os.getenv(
    "CHUNKING_STRATEGY", "recursive"
)  # "recursive" | "semantic"
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
DEFAULT_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
SEMANTIC_SIMILARITY_THRESHOLD = float(
    os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.75")
)
SEMANTIC_MAX_CHUNK_SENTENCES = int(os.getenv("SEMANTIC_MAX_CHUNK_SENTENCES", "8"))


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


# ─── Recursive (fast default) ─────────────────────────────────────────────────


def _recursive_chunk(
    text: str,
    chunk_size: int,
    overlap: int,
    source_metadata: dict,
) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.create_documents([text])
    return [
        Chunk(
            text=doc.page_content,
            metadata={**source_metadata, "chunk_index": idx},
        )
        for idx, doc in enumerate(docs)
    ]


# ─── Semantic (boundary-aware) ────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Fast cosine similarity — no external deps."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex — no NLTK dependency."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _semantic_chunk(
    text: str,
    source_metadata: dict,
) -> List[Chunk]:
    """
    Merge sentences into chunks based on semantic similarity.

    Algorithm:
      1. Split text into sentences.
      2. Embed each sentence.
      3. Greedily merge consecutive sentences into a chunk while
         cosine(current_chunk_embedding, next_sentence_embedding) ≥ threshold
         AND chunk has fewer than SEMANTIC_MAX_CHUNK_SENTENCES sentences.
      4. When similarity drops or sentence limit reached, start a new chunk.

    Fallback: if embedding fails at any point, falls back to recursive chunking.
    """
    try:
        from .embeddings import get_embedding
    except Exception:
        logger.warning("Embeddings unavailable, falling back to recursive chunker")
        return _recursive_chunk(
            text, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, source_metadata
        )

    sentences = _split_sentences(text)
    if not sentences:
        return []

    # Embed all sentences upfront
    try:
        embeddings = [get_embedding(s) for s in sentences]
    except Exception as e:
        logger.warning(f"Sentence embedding failed, falling back to recursive: {e}")
        return _recursive_chunk(
            text, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, source_metadata
        )

    chunks: List[Chunk] = []
    chunk_sentences: list[str] = [sentences[0]]
    # Running average embedding for the current chunk
    chunk_embedding = list(embeddings[0])

    for i in range(1, len(sentences)):
        sim = _cosine_similarity(chunk_embedding, embeddings[i])
        at_sentence_limit = len(chunk_sentences) >= SEMANTIC_MAX_CHUNK_SENTENCES

        if sim >= SEMANTIC_SIMILARITY_THRESHOLD and not at_sentence_limit:
            # Similar enough — merge into current chunk and update running average
            chunk_sentences.append(sentences[i])
            n = len(chunk_sentences)
            chunk_embedding = [
                (chunk_embedding[j] * (n - 1) + embeddings[i][j]) / n
                for j in range(len(chunk_embedding))
            ]
        else:
            # Boundary detected — save current chunk, start new one
            chunks.append(
                Chunk(
                    text=" ".join(chunk_sentences),
                    metadata={**source_metadata, "chunk_index": len(chunks)},
                )
            )
            chunk_sentences = [sentences[i]]
            chunk_embedding = list(embeddings[i])

    # Flush final chunk
    if chunk_sentences:
        chunks.append(
            Chunk(
                text=" ".join(chunk_sentences),
                metadata={**source_metadata, "chunk_index": len(chunks)},
            )
        )

    logger.debug(
        f"Semantic chunker produced {len(chunks)} chunks from {len(sentences)} sentences"
    )
    return chunks


# ─── Public API ───────────────────────────────────────────────────────────────


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    source_metadata: dict | None = None,
    strategy: str | None = None,
) -> List[Chunk]:
    """
    Split text into chunks with source metadata.

    Args:
        text:            Raw document text.
        chunk_size:      Character budget per chunk (recursive strategy only).
        overlap:         Character overlap between chunks (recursive only).
        source_metadata: {filename, doc_base_id, ...} tagged on every chunk.
        strategy:        Override CHUNKING_STRATEGY env var for this call.
                         "recursive" or "semantic".

    Returns:
        List[Chunk] ordered by position, each with metadata["chunk_index"].
    """
    if not text or not text.strip():
        return []

    base_meta = source_metadata or {}
    active_strategy = strategy or CHUNKING_STRATEGY

    if active_strategy == "semantic":
        return _semantic_chunk(text, base_meta)

    return _recursive_chunk(text, chunk_size, overlap, base_meta)
