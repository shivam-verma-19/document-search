"""Search service: HyDE query expansion → hybrid RRF fusion → reranking."""

import logging
import time
from typing import List

from . import embeddings
from .document_repository import DocumentRepository, SearchDocument, get_repository
from .query_expansion import generate_hyde_query
from .utils import normalize_text

logger = logging.getLogger(__name__)


_embedding_blocked_until: float = 0.0


def _is_embedding_blocked() -> bool:
    return time.time() < _embedding_blocked_until


def _block_embedding(seconds: int = 60) -> None:
    global _embedding_blocked_until
    _embedding_blocked_until = time.time() + seconds


# ─── RRF fusion ───────────────────────────────────────────────────────────────


def _reciprocal_rank_fusion(
    list1: List[SearchDocument],
    list2: List[SearchDocument],
    k: int = 5,
    rrf_k: int = 60,
) -> List[SearchDocument]:
    """
    Combine two ranked lists using Reciprocal Rank Fusion.
    Deduplication keys on doc_id (not page_content) so identical text from
    different logical documents is preserved.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, SearchDocument] = {}

    for rank, doc in enumerate(list1, start=1):
        key = doc.doc_id if doc.doc_id else doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        doc_map[key] = doc

    for rank, doc in enumerate(list2, start=1):
        key = doc.doc_id if doc.doc_id else doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked[:k]]


# ─── Hybrid search with HyDE ──────────────────────────────────────────────────


def hybrid_search(
    query: str,
    k: int = 5,
    repository: DocumentRepository | None = None,
) -> List[SearchDocument]:
    """
    Full retrieval pipeline:
      1. Normalize query.
      2. HyDE expansion — generate hypothetical answer, embed that instead
         of the raw question. Falls back to original query if LLM fails.
      3. Vector search using HyDE embedding.
      4. BM25 keyword search using normalized original query.
      5. RRF fusion of both result lists.

    Gracefully degrades if either branch fails.
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to hybrid_search")
        return []

    if repository is None:
        repository = get_repository()

    normalized_query = normalize_text(query)
    start_time = time.time()

    vector_docs: List[SearchDocument] = []
    bm25_docs: List[SearchDocument] = []

    # ── Vector branch (HyDE-expanded) ─────────────────────────────────────────
    try:
        if _is_embedding_blocked():
            raise Exception("Embedding skipped: rate-limited")
        hyde_text = generate_hyde_query(query)
        embedding = embeddings.get_embedding(hyde_text)
        vector_docs = repository.vector_search(embedding, k=k)
        logger.debug(
            f"Vector branch: {len(vector_docs)} docs (HyDE={'yes' if hyde_text != query else 'no'})"
        )
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            _block_embedding(60)
        logger.warning(f"Vector search failed (continuing with keyword only): {e}")

    # ── BM25 branch (original normalized query) ────────────────────────────────
    try:
        bm25_docs = repository.keyword_search(normalized_query, k=k)
        logger.debug(f"BM25 branch: {len(bm25_docs)} docs")
    except Exception as e:
        logger.warning(f"BM25 search failed (continuing with vector only): {e}")

    retrieval_depth = max(k * 4, 20)

    vector_docs = repository.vector_search(
        embedding,
        k=retrieval_depth,
    )
    
    bm25_docs = repository.keyword_search(
        normalized_query,
        k=retrieval_depth,
    )
    
    fused = _reciprocal_rank_fusion(
        vector_docs,
        bm25_docs,
        k=retrieval_depth,
    )
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        f"Hybrid search (RRF) completed in {elapsed_ms:.0f}ms, {len(fused)} docs"
    )
    return fused


# ─── Reranker ─────────────────────────────────────────────────────────────────


def rerank_documents(
    query: str,
    docs: List[SearchDocument],
    top_k: int | None = None,
) -> List[SearchDocument]:

    from . import reranker

    if not docs:
        return docs

    try:

        start_time = time.time()

        reranked = reranker.rerank(
            query=query,
            docs=docs,
        )

        if top_k:
            reranked = reranked[:top_k]

        logger.info(
            "BGE reranking completed in %.0f ms",
            (time.time() - start_time) * 1000,
        )

        return reranked

    except Exception as e:

        logger.warning(
            "Reranking failed, using original ranking: %s",
            e,
            exc_info=True,
        )

        return docs
