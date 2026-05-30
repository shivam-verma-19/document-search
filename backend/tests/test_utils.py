"""
Tests for backend/app/utils.py

Covers:
  - normalize_text  (lowercasing, whitespace collapse)
  - clean_text      (control character stripping, preserves punctuation)
  - build_prompt    (template structure)
"""

import backend.app.utils as m

# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_lowercases(self):
        assert m.normalize_text("HELLO WORLD") == "hello world"

    def test_strips_leading_trailing_whitespace(self):
        assert m.normalize_text("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self):
        assert m.normalize_text("hello   world") == "hello world"

    def test_empty_string(self):
        assert m.normalize_text("") == ""

    def test_newlines_collapsed(self):
        assert m.normalize_text("hello\n\nworld") == "hello world"


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_removes_control_chars(self):
        # clean_text removes control chars (0x00-0x08 etc), not punctuation
        result = m.clean_text("hello\x00world")
        assert "\x00" not in result
        assert "hello" in result
        assert "world" in result

    def test_preserves_alphanumeric(self):
        assert "hello" in m.clean_text("hello world")

    def test_preserves_periods(self):
        assert "." in m.clean_text("end of sentence.")

    def test_preserves_special_chars(self):
        # clean_text does NOT strip @, #, ! — only control characters
        result = m.clean_text("hello! @world#")
        assert "!" in result
        assert "@" in result
        assert "#" in result

    def test_normalises_crlf(self):
        assert m.clean_text("a\r\nb") == "a\nb"

    def test_empty_string(self):
        assert m.clean_text("") == ""


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_contains_context(self):
        assert "some context" in m.build_prompt("some context", "what is X?")

    def test_contains_query(self):
        assert "what is X?" in m.build_prompt("ctx", "what is X?")

    def test_contains_instruction(self):
        assert "Answer ONLY using the context" in m.build_prompt("ctx", "q")

    def test_returns_string(self):
        assert isinstance(m.build_prompt("ctx", "q"), str)
