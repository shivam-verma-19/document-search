"""Document repository abstraction for dependency injection and testability.

This module defines the DocumentRepository interface that decouples
search_service.py from concrete storage implementations, enabling:
  - Easy unit testing with mock repositories
  - Swappable implementations (vector store, BM25, etc.)
  - Clean separation of concerns
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class SearchDocument:
    """Represents a document returned from search."""

    page_content: str


class DocumentRepository(ABC):
    """Abstract interface for document search operations."""

    @abstractmethod
    def vector_search(self, embedding: list[float], k: int) -> List[SearchDocument]:
        pass

    @abstractmethod
    def keyword_search(self, query: str, k: int) -> List[SearchDocument]:
        pass


class HybridDocumentRepository(DocumentRepository):
    """Concrete implementation combining vector and keyword search.

    Uses:
    - s3_vectors_client for semantic search
    - bm25_cache for keyword search
    - RRF (Reciprocal Rank Fusion) to combine results
    """

    def __init__(self):
        # FIX: cache the BM25Retriever instance keyed by corpus version so we
        # don't rebuild the inverted index on every search request.
        self._bm25_retriever = None
        self._bm25_corpus_version: int = -1

    def vector_search(self, embedding: list[float], k: int) -> List[SearchDocument]:
        """Vector search via S3 Vectors."""
        from . import s3_vectors_client as vector_store
        from .retry import retry_with_backoff

        try:
            results = retry_with_backoff(
                lambda: vector_store.search_similar(embedding, k=k)
            )
            seen: set[str] = set()
            docs = []
            for text in results:
                if text and text not in seen:
                    seen.add(text)
                    docs.append(SearchDocument(page_content=text))
            logger.debug(f"Vector search returned {len(docs)} docs")
            return docs
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    def keyword_search(self, query: str, k: int) -> List[SearchDocument]:
        """Keyword search via BM25 cache.

        The BM25Retriever is rebuilt only when the corpus version changes,
        avoiding an O(n) index rebuild on every query.
        """
        from .bm25_cache import _warm_version, get_corpus
        from .hybrid import BM25Retriever

        try:
            corpus_texts = get_corpus()
            if not corpus_texts:
                logger.debug("BM25 corpus empty, skipping keyword search")
                return []

            # Rebuild retriever only if the corpus version changed.
            if (
                self._bm25_retriever is None
                or self._bm25_corpus_version != _warm_version
            ):
                all_docs = [SearchDocument(page_content=t) for t in corpus_texts]
                self._bm25_retriever = BM25Retriever(all_docs)
                self._bm25_corpus_version = _warm_version
                logger.debug(
                    f"BM25 retriever rebuilt for corpus v{_warm_version} ({len(corpus_texts)} docs)"
                )

            results = self._bm25_retriever.search(query, k=k)
            logger.debug(f"BM25 search returned {len(results)} docs")
            return results
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []


# Default singleton instance (can be replaced for testing)
_default_repository: DocumentRepository | None = None


def get_repository() -> DocumentRepository:
    """Get the default DocumentRepository instance."""
    global _default_repository
    if _default_repository is None:
        _default_repository = HybridDocumentRepository()
    return _default_repository


def set_repository(repo: DocumentRepository) -> None:
    """Replace the default repository (primarily for testing)."""
    global _default_repository
    _default_repository = repo
