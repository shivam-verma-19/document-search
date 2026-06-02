"""Document repository abstraction for dependency injection and testability."""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Minimum similarity score to admit a vector result into retrieval.
# Chunks below this are too distant from the query to be useful context.
# Set to 0.0 to disable filtering. Typical useful range: 0.3–0.6.
VECTOR_SCORE_THRESHOLD = float(os.getenv("VECTOR_SCORE_THRESHOLD", "0.35"))


@dataclass
class SearchDocument:
    """Represents a document returned from search."""

    page_content: str
    doc_id: str = ""
    metadata: dict = field(default_factory=dict)
    score: float = 1.0  # similarity score from vector search; 1.0 default for BM25


class DocumentRepository(ABC):
    """Abstract interface for document search operations."""

    @abstractmethod
    def vector_search(self, embedding: list[float], k: int) -> List[SearchDocument]:
        pass

    @abstractmethod
    def keyword_search(self, query: str, k: int) -> List[SearchDocument]:
        pass


class HybridDocumentRepository(DocumentRepository):
    """Concrete implementation combining vector and keyword search."""

    def __init__(self):
        self._bm25_retriever = None
        self._bm25_corpus_version: int = -1

    def vector_search(self, embedding: list[float], k: int) -> List[SearchDocument]:
        """
        Vector search via S3 Vectors with similarity score threshold.

        Chunks below VECTOR_SCORE_THRESHOLD are discarded before RRF fusion,
        preventing low-relevance content from polluting the prompt context.
        """
        from . import s3_vectors_client as vector_store
        from .retry import retry_with_backoff

        try:
            results = retry_with_backoff(
                lambda: vector_store.search_documents(
                    embedding, k=k, score_threshold=VECTOR_SCORE_THRESHOLD
                )
            )
            seen: set[str] = set()
            docs = []
            for item in results:
                doc_id = item.get("_id", "")
                source = item.get("_source", {})
                text = source.get("text", "")
                meta = source.get("metadata", {})
                score = float(item.get("score", 1.0))
                if text and doc_id not in seen:
                    seen.add(doc_id)
                    docs.append(
                        SearchDocument(
                            page_content=text,
                            doc_id=doc_id,
                            metadata=meta,
                            score=score,
                        )
                    )
            logger.debug(
                f"Vector search returned {len(docs)} docs "
                f"(threshold={VECTOR_SCORE_THRESHOLD})"
            )
            return docs
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    def keyword_search(self, query: str, k: int) -> List[SearchDocument]:
        """Keyword search via BM25 cache — retriever rebuilt only on corpus version change."""
        from .bm25_cache import _warm_version, get_corpus
        from .hybrid import BM25Retriever

        try:
            corpus_texts = get_corpus()
            if not corpus_texts:
                logger.debug("BM25 corpus empty, skipping keyword search")
                return []

            if (
                self._bm25_retriever is None
                or self._bm25_corpus_version != _warm_version
            ):
                all_docs = [
                    SearchDocument(page_content=t, doc_id=f"bm25:{i}")
                    for i, t in enumerate(corpus_texts)
                ]
                self._bm25_retriever = BM25Retriever(all_docs)
                self._bm25_corpus_version = _warm_version
                logger.debug(
                    f"BM25 retriever rebuilt for corpus v{_warm_version} "
                    f"({len(corpus_texts)} docs)"
                )

            results = self._bm25_retriever.search(query, k=k)
            logger.debug(f"BM25 search returned {len(results)} docs")
            return results
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []


_default_repository: Optional[DocumentRepository] = None


def get_repository() -> DocumentRepository:
    global _default_repository
    if _default_repository is None:
        _default_repository = HybridDocumentRepository()
    return _default_repository


def set_repository(repo: DocumentRepository) -> None:
    global _default_repository
    _default_repository = repo
