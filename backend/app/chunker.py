"""
Text chunker with metadata preservation, section header extraction, and
adaptive semantic boundary detection.

Strategies available via CHUNKING_STRATEGY env var:

  "adaptive" (default)  — Splits on document section boundaries and topic shifts.
                           Merges consecutive sentences while cosine similarity is
                           above an adaptive threshold. Preserves active section
                           headers in chunk metadata.
  "semantic"            — Alias for adaptive semantic chunking.
  "recursive"           — RecursiveCharacterTextSplitter with paragraph/sentence
                           separators. Fast, no LLM/embedding calls, works on any text.

All strategies return List[Chunk] with full source metadata and section headers
so attribution is preserved all the way to the LLM prompt.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNKING_STRATEGY = os.getenv(
    "CHUNKING_STRATEGY", "adaptive"
)  # "adaptive" | "semantic" | "recursive"
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
DEFAULT_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
SEMANTIC_SIMILARITY_THRESHOLD = float(
    os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.75")
)
SEMANTIC_MAX_CHUNK_SENTENCES = int(os.getenv("SEMANTIC_MAX_CHUNK_SENTENCES", "8"))
MIN_CHUNK_CHAR_LEN = int(os.getenv("MIN_CHUNK_CHAR_LEN", "50"))


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


# ─── Header & Boundary Detection ──────────────────────────────────────────────

_HEADER_PATTERNS = [
    re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE),  # Markdown # Header
    re.compile(r"^([A-Z0-9\s\-_]{3,50}):\s*$", re.MULTILINE),  # SECTION:
    re.compile(
        r"^(\d+\.?\s+[A-Z][A-Za-z0-9\s\-_]{2,50})$", re.MULTILINE
    ),  # 1. Introduction
]


def detect_section_header(line: str) -> Optional[str]:
    """Detect if a text line is a section header."""
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return None

    # Check regex patterns
    for pattern in _HEADER_PATTERNS:
        m = pattern.match(stripped)
        if m:
            header_text = m.group(1).strip()
            # Clean leading hashes if any
            header_text = re.sub(r"^#+\s*", "", header_text)
            return header_text

    # Check uppercase section line (e.g., "SCALING ARCHITECTURE")
    if stripped.isupper() and 3 <= len(stripped.split()) <= 10:
        return stripped.title()

    return None


def _find_section_for_text(text: str, full_doc: str, fallback: str = "General") -> str:
    """Find the most recent section header preceding this text segment."""
    pos = full_doc.find(text)
    if pos <= 0:
        return fallback

    doc_prefix = full_doc[:pos]
    lines = doc_prefix.split("\n")
    for line in reversed(lines):
        h = detect_section_header(line)
        if h:
            return h
    return fallback


# ─── Recursive (fast fallback) ────────────────────────────────────────────────


def _recursive_chunk(
    text: str,
    chunk_size: int,
    overlap: int,
    source_metadata: dict,
) -> List[Chunk]:
    # If document has distinct markdown headers, divide by section boundary
    header_indices = [m.start() for m in re.finditer(r"(?m)^#{1,6}\s+", text)]
    if len(header_indices) > 1:
        sections = []
        for i in range(len(header_indices)):
            start = header_indices[i]
            end = header_indices[i + 1] if i + 1 < len(header_indices) else len(text)
            sections.append(text[start:end].strip())
    else:
        sections = [text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: List[Chunk] = []
    current_header = source_metadata.get("section_header", "General")

    for section in sections:
        first_line = section.split("\n")[0]
        detected = detect_section_header(first_line)
        if detected:
            current_header = detected

        docs = splitter.create_documents([section])
        for doc in docs:
            chunks.append(
                Chunk(
                    text=doc.page_content,
                    metadata={
                        **source_metadata,
                        "chunk_index": len(chunks),
                        "section_header": current_header,
                    },
                )
            )
    return chunks


# ─── Adaptive / Semantic (boundary-aware) ────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Fast cosine similarity — no external deps."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences while respecting headings and punctuation."""
    lines = text.split("\n")
    sentences: list[str] = []
    current_para: list[str] = []

    def flush_para():
        nonlocal current_para
        if not current_para:
            return
        p = " ".join(current_para).strip()
        current_para = []
        if not p:
            return
        sents = re.split(r"(?<=[.!?])\s+", p)
        for s in sents:
            s_clean = s.strip()
            if not s_clean:
                continue
            if len(s_clean) > DEFAULT_CHUNK_SIZE:
                words = s_clean.split(" ")
                current_sub: list[str] = []
                current_len = 0
                for w in words:
                    if current_len + len(w) + 1 > DEFAULT_CHUNK_SIZE and current_sub:
                        sentences.append(" ".join(current_sub))
                        current_sub = [w]
                        current_len = len(w)
                    else:
                        current_sub.append(w)
                        current_len += len(w) + 1
                if current_sub:
                    sentences.append(" ".join(current_sub))
            else:
                sentences.append(s_clean)

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            flush_para()
            continue
        header = detect_section_header(line_clean)
        if header:
            flush_para()
            sentences.append(line_clean)
        else:
            current_para.append(line_clean)

    flush_para()
    return sentences


def _semantic_chunk(
    text: str,
    source_metadata: dict,
) -> List[Chunk]:
    """
    Merge sentences into chunks based on section headers and semantic similarity.

    Algorithm:
      1. Split text into sentences and track section headers.
      2. If a sentence is a section header, immediately flush the current chunk
         and update the active section header.
      3. Embed consecutive sentences.
      4. Merge sentences while cosine similarity >= threshold AND sentence count limit
         is not exceeded AND chunk character limit is not exceeded.
      5. When similarity drops or limit is reached, start a new chunk.

    Fallback: if embedding fails, falls back gracefully to recursive chunking.
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

    try:
        embeddings = [get_embedding(s) for s in sentences]
    except Exception as e:
        logger.warning(f"Sentence embedding failed, falling back to recursive: {e}")
        return _recursive_chunk(
            text, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, source_metadata
        )

    chunks: List[Chunk] = []
    chunk_sentences: list[str] = []
    chunk_embedding: list[float] | None = None
    current_header = source_metadata.get("section_header", "General")

    def _flush_chunk():
        nonlocal chunk_sentences, chunk_embedding
        if not chunk_sentences:
            return
        content = " ".join(chunk_sentences).strip()
        if content:
            chunks.append(
                Chunk(
                    text=content,
                    metadata={
                        **source_metadata,
                        "chunk_index": len(chunks),
                        "section_header": current_header,
                    },
                )
            )
        chunk_sentences = []
        chunk_embedding = None

    for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
        detected = detect_section_header(sent)
        if detected:
            # Header encountered: flush preceding chunk and switch header
            _flush_chunk()
            current_header = detected
            # Keep header as beginning of the new section
            chunk_sentences.append(sent)
            chunk_embedding = list(emb)
            continue

        if not chunk_sentences:
            chunk_sentences.append(sent)
            chunk_embedding = list(emb)
            continue

        sim = _cosine_similarity(chunk_embedding, emb)  # type: ignore
        at_sentence_limit = len(chunk_sentences) >= SEMANTIC_MAX_CHUNK_SENTENCES
        curr_len = sum(len(s) for s in chunk_sentences)
        at_char_limit = (curr_len + len(sent)) > DEFAULT_CHUNK_SIZE

        if (
            sim >= SEMANTIC_SIMILARITY_THRESHOLD
            and not at_sentence_limit
            and not at_char_limit
        ):
            # Merge into current chunk
            chunk_sentences.append(sent)
            n = len(chunk_sentences)
            chunk_embedding = [
                (chunk_embedding[j] * (n - 1) + emb[j]) / n  # type: ignore
                for j in range(len(chunk_embedding))  # type: ignore
            ]
        else:
            # Boundary detected (topic shift or limit reached)
            _flush_chunk()
            chunk_sentences = [sent]
            chunk_embedding = list(emb)

    _flush_chunk()

    logger.debug(
        f"Adaptive chunker produced {len(chunks)} chunks from {len(sentences)} sentences"
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
    Split text into chunks with source metadata and section headers.

    Args:
        text:            Raw document text.
        chunk_size:      Character budget per chunk (recursive strategy only).
        overlap:         Character overlap between chunks (recursive only).
        source_metadata: {filename, doc_base_id, ...} tagged on every chunk.
        strategy:        Override CHUNKING_STRATEGY env var for this call.
                         "adaptive", "semantic", or "recursive".

    Returns:
        List[Chunk] ordered by position, each with chunk_index and section_header.
    """
    if not text or not text.strip():
        return []

    base_meta = source_metadata or {}
    active_strategy = strategy or CHUNKING_STRATEGY

    if active_strategy in ("semantic", "adaptive"):
        return _semantic_chunk(text, base_meta)

    return _recursive_chunk(text, chunk_size, overlap, base_meta)
