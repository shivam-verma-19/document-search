"""Search service using hybrid RRF fusion with injected document repository.

Provides high-level search operations combining vector similarity and
keyword matching, with dependency injection for testability.
"""

import logging
import time
from typing import List

from .document_repository import DocumentRepository, SearchDocument, get_repository
from .utils import normalize_text

logger = logging.getLogger(__name__)


# ─── RRF fusion ───────────────────────────────────────────────────────────────


def _reciprocal_rank_fusion(
    list1: List[SearchDocument],
    list2: List[SearchDocument],
    k: int = 5,
    rrf_k: int = 60,
) -> List[SearchDocument]:
    """Combine two ranked lists using Reciprocal Rank Fusion.

    RRF gives equal weight to both rankers and handles missing documents.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, SearchDocument] = {}
    for rank, doc in enumerate(list1, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        doc_map[key] = doc
    for rank, doc in enumerate(list2, start=1):
        key = doc.page_content
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        doc_map[key] = doc
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked[:k]]


# ─── Hybrid search ────────────────────────────────────────────────────────────


def hybrid_search(
    query: str,
    k: int = 5,
    repository: DocumentRepository | None = None,
) -> List[SearchDocument]:
    """Search documents combining vector similarity and keyword matching.

    Args:
        query: User search query
        k: Number of results to return
        repository: DocumentRepository instance (uses default if None)

    Returns:
        List of SearchDocument results ranked by RRF

    Gracefully degrades if either branch fails (vector OR keyword search).
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to hybrid_search")
        return []

    if repository is None:
        repository = get_repository()

    # Normalize query for better semantic matching
    normalized_query = normalize_text(query)

    start_time = time.time()
    vector_docs: List[SearchDocument] = []
    bm25_docs: List[SearchDocument] = []

    # Vector branch
    try:
        from . import embeddings

        embedding = embeddings.get_embedding(normalized_query)
        vector_docs = repository.vector_search(embedding, k=k)
    except Exception as e:
        logger.warning(f"Vector search failed (continuing with keyword only): {e}")

    # BM25 branch — corpus from cache, NOT from paginating S3 Vectors
    try:
        bm25_docs = repository.keyword_search(normalized_query, k=k)
    except Exception as e:
        logger.warning(f"BM25 search failed (continuing with vector only): {e}")

    fused = _reciprocal_rank_fusion(vector_docs, bm25_docs, k=k)
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        f"Hybrid search (RRF) completed in {elapsed_ms:.0f}ms, {len(fused)} docs"
    )
    return fused


# ─── Reranker ─────────────────────────────────────────────────────────────────


def rerank_documents(query: str, docs: List[SearchDocument]) -> List[SearchDocument]:
    """Re-rank documents using cross-encoder for better relevance.

    Args:
        query: User search query
        docs: Documents to rerank

    Returns:
        Reranked documents (same list if reranking fails)
    """
    from . import reranker

    if not docs:
        return docs
    try:
        start_time = time.time()
        reranked = reranker.rerank(query, docs)
        logger.debug(f"Reranking completed in {(time.time()-start_time)*1000:.0f}ms")
        return reranked
    except Exception as e:
        logger.warning(f"Reranking failed, using original order: {e}", exc_info=True)
        return docs
