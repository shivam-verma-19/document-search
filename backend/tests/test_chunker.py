"""Tests for chunker.py — recursive and semantic chunking strategies."""

import sys

import pytest


@pytest.fixture(autouse=True)
def _evict_stubs():
    """Remove any MagicMock stubs installed by test_rag so real modules load."""
    for mod in ["backend.app.chunker", "backend.app.embeddings"]:
        sys.modules.pop(mod, None)
    yield
    for mod in ["backend.app.chunker", "backend.app.embeddings"]:
        sys.modules.pop(mod, None)


from unittest.mock import patch

import pytest


class TestRecursiveChunking:
    def test_returns_chunk_objects(self):
        from backend.app.chunker import chunk_text

        result = chunk_text("Hello world. " * 100, strategy="recursive")
        assert len(result) > 0
        assert hasattr(result[0], "text")
        assert hasattr(result[0], "metadata")

    def test_chunk_index_sequential(self):
        from backend.app.chunker import chunk_text

        result = chunk_text("word " * 500, strategy="recursive")
        for i, chunk in enumerate(result):
            assert chunk.metadata["chunk_index"] == i

    def test_source_metadata_propagated(self):
        from backend.app.chunker import chunk_text

        meta = {"filename": "test.pdf", "doc_base_id": "abc123"}
        result = chunk_text("word " * 200, source_metadata=meta, strategy="recursive")
        for chunk in result:
            assert chunk.metadata["filename"] == "test.pdf"
            assert chunk.metadata["doc_base_id"] == "abc123"

    def test_empty_text_returns_empty(self):
        from backend.app.chunker import chunk_text

        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_multiple_chunks_for_long_text(self):
        from backend.app.chunker import chunk_text

        result = chunk_text(
            "word " * 500, chunk_size=200, overlap=20, strategy="recursive"
        )
        assert len(result) > 1

    def test_respects_chunk_size(self):
        from backend.app.chunker import chunk_text

        result = chunk_text(
            "word " * 500, chunk_size=200, overlap=20, strategy="recursive"
        )
        for chunk in result:
            assert len(chunk.text) <= 300  # splitter tolerance

    def test_overlap_produces_shared_words(self):
        from backend.app.chunker import chunk_text

        text = " ".join([f"word{i}" for i in range(300)])
        result = chunk_text(text, chunk_size=200, overlap=50, strategy="recursive")
        if len(result) > 1:
            words0 = set(result[0].text.split())
            words1 = set(result[1].text.split())
            assert len(words0 & words1) > 0

    def test_no_source_metadata_still_has_chunk_index(self):
        from backend.app.chunker import chunk_text

        result = chunk_text("word " * 100, strategy="recursive")
        assert "chunk_index" in result[0].metadata


class TestSemanticChunking:
    def _mock_embedding(self, text: str) -> list[float]:
        """Returns a vector that changes based on first word — simulates topic shifts."""
        first_word = text.split()[0] if text.split() else "x"
        seed = sum(ord(c) for c in first_word)
        import math

        return [math.sin(seed + i * 0.1) for i in range(16)]

    def test_returns_chunks(self):
        from backend.app.chunker import chunk_text

        text = "The sky is blue. The sun is bright. " * 5
        # get_embedding is imported inside _semantic_chunk via
        # `from .embeddings import get_embedding`, so patch the embeddings module.
        with patch(
            "backend.app.embeddings.get_embedding", side_effect=self._mock_embedding
        ):
            result = chunk_text(text, strategy="semantic")
        assert len(result) > 0
        assert all(hasattr(c, "text") for c in result)

    def test_chunk_index_sequential(self):
        from backend.app.chunker import chunk_text

        text = "Sentence one here. Sentence two here. Sentence three here. " * 3
        with patch(
            "backend.app.embeddings.get_embedding", side_effect=self._mock_embedding
        ):
            result = chunk_text(text, strategy="semantic")
        for i, chunk in enumerate(result):
            assert chunk.metadata["chunk_index"] == i

    def test_source_metadata_propagated(self):
        from backend.app.chunker import chunk_text

        meta = {"filename": "report.pdf"}
        text = "This is a sentence. This is another sentence. " * 4
        with patch(
            "backend.app.embeddings.get_embedding", side_effect=self._mock_embedding
        ):
            result = chunk_text(text, source_metadata=meta, strategy="semantic")
        for chunk in result:
            assert chunk.metadata["filename"] == "report.pdf"

    def test_falls_back_to_recursive_on_embedding_failure(self):
        from backend.app.chunker import chunk_text

        with patch(
            "backend.app.embeddings.get_embedding", side_effect=Exception("embed down")
        ):
            result = chunk_text("word " * 200, strategy="semantic")
        # Should still produce chunks via recursive fallback
        assert len(result) > 0

    def test_empty_text_returns_empty(self):
        from backend.app.chunker import chunk_text

        with patch("backend.app.embeddings.get_embedding", return_value=[0.1] * 16):
            assert chunk_text("", strategy="semantic") == []

    def test_similar_sentences_merged_into_one_chunk(self):
        """All sentences highly similar → should merge into fewer chunks."""
        from backend.app.chunker import SEMANTIC_SIMILARITY_THRESHOLD, chunk_text

        # Same embedding for all sentences → cosine sim = 1.0 → always merge
        with patch(
            "backend.app.embeddings.get_embedding", return_value=[1.0] + [0.0] * 15
        ):
            text = (
                ". ".join([f"Sentence about topic A number {i}" for i in range(5)])
                + "."
            )
            result = chunk_text(text, strategy="semantic")
        # With identical embeddings all sentences merge (up to SEMANTIC_MAX_CHUNK_SENTENCES)
        assert len(result) <= 2

    def test_strategy_env_default(self, monkeypatch):
        monkeypatch.setenv("CHUNKING_STRATEGY", "recursive")
        import importlib

        import backend.app.chunker as ch

        importlib.reload(ch)
        assert ch.CHUNKING_STRATEGY == "recursive"
