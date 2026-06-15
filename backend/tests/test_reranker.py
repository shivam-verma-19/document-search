"""
Tests for reranker.py — keyword-overlap reranker.
"""

import sys

import pytest
from annotated_doc import Doc


@pytest.fixture(autouse=True)
def _evict_stubs():
    """Remove any MagicMock stubs installed by test_rag so real modules load."""
    for mod in ["backend.app.reranker"]:
        sys.modules.pop(mod, None)
    yield
    for mod in ["backend.app.reranker"]:
        sys.modules.pop(mod, None)


from unittest.mock import patch

import pytest


def _make_doc(text: str, doc_id: str = ""):
    from backend.app.document_repository import SearchDocument

    return SearchDocument(page_content=text, doc_id=doc_id or text[:20])


class TestKeywordFallback:

    def test_returns_documents(self):
        from backend.app import reranker

        docs = [
            _make_doc("some python code"),
            _make_doc("python machine learning python python"),
        ]

        ranked = reranker.rerank(
            "python",
            docs,
        )

        assert len(ranked) == 2

        assert set(d.page_content for d in ranked) == {
            "some python code",
            "python machine learning python python",
        }

    def test_empty_returns_empty(self):
        from backend.app.reranker import rerank

        assert rerank("query", []) == []

    def test_single_doc_returned(self):
        from backend.app.reranker import rerank

        doc = _make_doc("hello world")
        assert rerank("hello", [doc]) == [doc]

    def test_no_overlap_still_returns_doc(self):
        from backend.app.reranker import rerank

        doc = _make_doc("quantum physics banana")
        result = rerank("unrelated query xyz", [doc])
        assert len(result) == 1

    def test_rerank_preserves_documents(self):
        from backend.app import reranker

        docs = [
            _make_doc("python"),
            _make_doc("python python python python"),
        ]

        ranked = reranker.rerank(
            "python",
            docs,
        )

        assert len(ranked) == 2

    def test_preserves_all_docs(self):
        from backend.app.reranker import rerank

        docs = [_make_doc(f"doc {i}") for i in range(5)]
        result = rerank("doc", docs)
        assert len(result) == 5


class TestRerankerSorting:
    def test_reranker_returns_all_docs(self):
        from backend.app import reranker

        docs = [
            _make_doc("aws"),
            _make_doc("dynamodb"),
            _make_doc("s3"),
        ]

        ranked = reranker.rerank(
            "aws services",
            docs,
        )

        assert len(ranked) == 3

    def test_exact_query_match_ranks_first(self):
        from backend.app.reranker import rerank

        docs = [
            _make_doc("bananas are yellow"),
            _make_doc("machine learning"),
        ]
        result = rerank("machine learning", docs)
        assert result[0].page_content == "machine learning"
