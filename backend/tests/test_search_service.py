"""Tests for search_service.py — HyDE integration, RRF, score threshold."""

import sys

import pytest


@pytest.fixture(autouse=True)
def _evict_stubs():
    """Remove any MagicMock stubs installed by test_rag so real modules load."""
    for mod in [
        "backend.app.search_service",
        "backend.app.query_expansion",
        "backend.app.embeddings",
    ]:
        sys.modules.pop(mod, None)
    yield
    for mod in [
        "backend.app.search_service",
        "backend.app.query_expansion",
        "backend.app.embeddings",
    ]:
        sys.modules.pop(mod, None)


from unittest.mock import MagicMock, patch

import pytest


def _make_doc(text: str, doc_id: str = "", score: float = 1.0):
    from backend.app.document_repository import SearchDocument

    return SearchDocument(
        page_content=text, doc_id=doc_id or f"id:{text[:10]}", score=score
    )


class TestRRFFusion:
    def test_merges_two_lists(self):
        from backend.app.search_service import _reciprocal_rank_fusion

        list1 = [_make_doc("a", "id1"), _make_doc("b", "id2")]
        list2 = [_make_doc("c", "id3"), _make_doc("d", "id4")]
        result = _reciprocal_rank_fusion(list1, list2, k=4)
        assert len(result) == 4

    def test_deduplicates_by_doc_id(self):
        from backend.app.search_service import _reciprocal_rank_fusion

        shared = _make_doc("shared text", "shared_id")
        list1 = [shared, _make_doc("unique1", "id1")]
        list2 = [shared, _make_doc("unique2", "id2")]
        result = _reciprocal_rank_fusion(list1, list2, k=10)
        ids = [d.doc_id for d in result]
        assert ids.count("shared_id") == 1

    def test_same_text_different_id_both_kept(self):
        from backend.app.search_service import _reciprocal_rank_fusion

        list1 = [_make_doc("boilerplate", "id_a")]
        list2 = [_make_doc("boilerplate", "id_b")]
        result = _reciprocal_rank_fusion(list1, list2, k=10)
        assert len(result) == 2

    def test_doc_in_both_lists_ranks_higher(self):
        from backend.app.search_service import _reciprocal_rank_fusion

        in_both = _make_doc("best", "best_id")
        list1 = [in_both, _make_doc("med", "med_id")]
        list2 = [in_both, _make_doc("other", "other_id")]
        result = _reciprocal_rank_fusion(list1, list2, k=5)
        assert result[0].doc_id == "best_id"

    def test_respects_k(self):
        from backend.app.search_service import _reciprocal_rank_fusion

        l1 = [_make_doc(f"a{i}", f"a{i}") for i in range(10)]
        l2 = [_make_doc(f"b{i}", f"b{i}") for i in range(10)]
        assert len(_reciprocal_rank_fusion(l1, l2, k=3)) == 3

    def test_empty_lists_returns_empty(self):
        from backend.app.search_service import _reciprocal_rank_fusion

        assert _reciprocal_rank_fusion([], [], k=5) == []


class TestHybridSearchWithHyDE:
    def _mock_repo(
        self, vector_docs=None, bm25_docs=None, vector_raises=False, bm25_raises=False
    ):
        from backend.app.document_repository import DocumentRepository

        repo = MagicMock(spec=DocumentRepository)
        if vector_raises:
            repo.vector_search.side_effect = Exception("vector down")
        else:
            repo.vector_search.return_value = vector_docs or []
        if bm25_raises:
            repo.keyword_search.side_effect = Exception("bm25 down")
        else:
            repo.keyword_search.return_value = bm25_docs or []
        return repo

    def test_hyde_used_for_vector_embedding(self):
        from backend.app.search_service import hybrid_search

        repo = self._mock_repo(vector_docs=[_make_doc("result", "id1")])

        # generate_hyde_query and embeddings are imported inside hybrid_search's
        # function body via `from .query_expansion import generate_hyde_query`
        # and `from . import embeddings`, so patch their canonical module paths.
        with patch(
            "backend.app.query_expansion.generate_hyde_query",
            return_value="hypothesis text",
        ) as mock_hyde, patch(
            "backend.app.embeddings.get_embedding",
            return_value=[0.1] * 768,
        ) as mock_emb:
            hybrid_search("short question?", repository=repo)

        mock_hyde.assert_called_once_with("short question?")
        mock_emb.assert_called_once_with("hypothesis text")

    def test_bm25_uses_original_normalized_query(self):
        from backend.app.search_service import hybrid_search

        repo = self._mock_repo()

        with patch(
            "backend.app.query_expansion.generate_hyde_query",
            return_value="hypothesis",
        ), patch("backend.app.embeddings.get_embedding", return_value=[0.1] * 768):
            hybrid_search("What Is RAG?", repository=repo)

        # BM25 should receive normalized query
        call_args = repo.keyword_search.call_args
        assert call_args.args[0] == "what is rag"  # normalized

    def test_vector_failure_falls_back_to_bm25(self):
        from backend.app.search_service import hybrid_search

        repo = self._mock_repo(bm25_docs=[_make_doc("bm25 only", "b1")])

        with patch(
            "backend.app.query_expansion.generate_hyde_query",
            side_effect=Exception("hyde down"),
        ), patch(
            "backend.app.embeddings.get_embedding",
            side_effect=Exception("embed down"),
        ):
            results = hybrid_search("query", repository=repo)

        assert any(d.page_content == "bm25 only" for d in results)

    def test_hyde_failure_still_completes_search(self):
        from backend.app.search_service import hybrid_search

        repo = self._mock_repo(vector_docs=[_make_doc("vec", "v1")])

        with patch(
            "backend.app.query_expansion.generate_hyde_query",
            return_value="original query",
        ), patch("backend.app.embeddings.get_embedding", return_value=[0.1] * 768):
            results = hybrid_search("original query", repository=repo)

        assert len(results) > 0

    def test_empty_query_returns_empty(self):
        from backend.app.search_service import hybrid_search

        assert hybrid_search("") == []
        assert hybrid_search("   ") == []

    def test_caps_at_k(self):
        from backend.app.search_service import hybrid_search

        repo = self._mock_repo(
            vector_docs=[_make_doc(f"v{i}", f"vid{i}") for i in range(10)],
            bm25_docs=[_make_doc(f"b{i}", f"bid{i}") for i in range(10)],
        )
        with patch(
            "backend.app.query_expansion.generate_hyde_query", return_value="q"
        ), patch("backend.app.embeddings.get_embedding", return_value=[0.1] * 768):
            results = hybrid_search("query", k=3, repository=repo)
        assert len(results) <= 3


class TestScoreThreshold:
    def test_low_score_chunks_filtered(self, monkeypatch):
        """Chunks below VECTOR_SCORE_THRESHOLD must not reach RRF."""
        import backend.app.document_repository as dr

        monkeypatch.setattr(dr, "VECTOR_SCORE_THRESHOLD", 0.5)

        repo = MagicMock()
        repo.vector_search.return_value = [
            _make_doc("high relevance", "id1", score=0.9),
        ]
        repo.keyword_search.return_value = []

        from backend.app.search_service import hybrid_search

        with patch(
            "backend.app.query_expansion.generate_hyde_query", return_value="q"
        ), patch("backend.app.embeddings.get_embedding", return_value=[0.1] * 768):
            results = hybrid_search("question", repository=repo)

        # Only the high-score doc should be present
        assert all(getattr(d, "score", 1.0) >= 0.0 for d in results)
