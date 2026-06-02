"""
Tests for utils.py — normalize_text, clean_text, build_prompt.
"""

import pytest


class TestNormalizeText:
    def test_lowercases(self):
        from backend.app.utils import normalize_text

        assert normalize_text("What IS This?") == "what is this"

    def test_strips_whitespace(self):
        from backend.app.utils import normalize_text

        assert normalize_text("  hello world  ") == "hello world"

    def test_collapses_internal_spaces(self):
        from backend.app.utils import normalize_text

        assert normalize_text("hello   world") == "hello world"

    def test_strips_trailing_punctuation(self):
        from backend.app.utils import normalize_text

        assert normalize_text("what is ai?") == "what is ai"
        assert normalize_text("hello!") == "hello"
        assert normalize_text("end.") == "end"

    def test_same_query_different_case_equal(self):
        from backend.app.utils import normalize_text

        assert normalize_text("What Is RAG?") == normalize_text("what is rag")

    def test_empty_string(self):
        from backend.app.utils import normalize_text

        assert normalize_text("") == ""


class TestBuildPrompt:
    def test_returns_string(self):
        from backend.app.utils import build_prompt

        result = build_prompt("some context", "my question")
        assert isinstance(result, str)

    def test_contains_query(self):
        from backend.app.utils import build_prompt

        result = build_prompt("context here", "what is rag?")
        assert "what is rag?" in result

    def test_contains_context(self):
        from backend.app.utils import build_prompt

        result = build_prompt("unique context block", "question")
        assert "unique context block" in result

    def test_source_labels_when_sources_provided(self):
        from backend.app.utils import build_prompt

        sources = [
            {"filename": "report.pdf", "chunk_index": 0, "text": "chunk content"},
        ]
        result = build_prompt("", "question", sources=sources)
        assert "report.pdf" in result
        assert "chunk content" in result
        assert "Source 1" in result

    def test_multiple_sources_labeled(self):
        from backend.app.utils import build_prompt

        sources = [
            {"filename": "a.pdf", "chunk_index": 0, "text": "text a"},
            {"filename": "b.pdf", "chunk_index": 1, "text": "text b"},
        ]
        result = build_prompt("", "question", sources=sources)
        assert "Source 1" in result
        assert "Source 2" in result
        assert "a.pdf" in result
        assert "b.pdf" in result

    def test_separator_between_chunks(self):
        from backend.app.utils import build_prompt

        sources = [
            {"filename": "f.pdf", "chunk_index": 0, "text": "chunk one"},
            {"filename": "f.pdf", "chunk_index": 1, "text": "chunk two"},
        ]
        result = build_prompt("", "q", sources=sources)
        assert "---" in result

    def test_no_sources_uses_fallback(self):
        from backend.app.utils import build_prompt

        result = build_prompt("line one\nline two", "question", sources=None)
        assert "Source" in result or "line one" in result


class TestCleanText:
    def test_removes_control_chars(self):
        from backend.app.utils import clean_text

        assert "\x00" not in clean_text("hello\x00world")

    def test_normalizes_line_endings(self):
        from backend.app.utils import clean_text

        assert "\r" not in clean_text("hello\r\nworld")

    def test_preserves_punctuation(self):
        from backend.app.utils import clean_text

        text = "Hello, world! Is this working?"
        assert clean_text(text) == text
