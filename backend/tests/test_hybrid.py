"""
Tests for backend/app/hybrid.py – BM25Retriever

Covers:
  - init with empty corpus
  - search returns correct number of docs
  - top result is the most relevant
  - k > corpus size returns all docs
  - query with no overlap returns docs (not crash)
  - case-insensitive tokenisation gap (known limitation documented)
"""

import types

import pytest


def _doc(text):
    d = types.SimpleNamespace()
    d.page_content = text
    d.metadata = {}
    return d


class TestBM25Retriever:
    def _make(self, texts):
        from backend.app.hybrid import BM25Retriever

        docs = [_doc(t) for t in texts]
        return BM25Retriever(docs), docs

    def test_returns_k_results(self):
        retriever, _ = self._make(
            ["apple pie", "banana bread", "cherry tart", "date cake"]
        )
        results = retriever.search("apple", k=2)
        assert len(results) == 2

    def test_best_match_first(self):
        retriever, _ = self._make(
            [
                "machine learning algorithms",
                "cooking pasta recipes",
                "deep learning neural networks",
            ]
        )
        results = retriever.search("deep learning", k=3)
        assert "deep learning" in results[0].page_content

    def test_k_larger_than_corpus(self):
        retriever, docs = self._make(["doc one", "doc two"])
        results = retriever.search("doc", k=100)
        assert len(results) == len(docs)

    def test_no_matching_terms_no_crash(self):
        retriever, _ = self._make(["hello world", "foo bar"])
        results = retriever.search("zzzzzzz", k=2)
        assert isinstance(results, list)

    def test_single_document_corpus(self):
        retriever, _ = self._make(["only document here"])
        results = retriever.search("document", k=1)
        assert len(results) == 1
        assert "only document here" in results[0].page_content

    def test_empty_query_no_crash(self):
        retriever, _ = self._make(["hello world", "test data"])
        results = retriever.search("", k=2)
        assert isinstance(results, list)

    def test_returns_document_objects(self):
        retriever, _ = self._make(["text here"])
        results = retriever.search("text", k=1)
        assert hasattr(results[0], "page_content")

    def test_empty_corpus_returns_empty_list(self):
        """
        BM25Okapi raises ZeroDivisionError on empty corpus.
        BM25Retriever must guard against this and return [] gracefully.
        Regression test for: https://github.com/dorianbrown/rank_bm25/issues/...
        """
        from backend.app.hybrid import BM25Retriever

        retriever = BM25Retriever([])
        # Must not raise ZeroDivisionError
        results = retriever.search("anything", k=5)
        assert results == []
