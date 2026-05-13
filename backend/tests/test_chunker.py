"""
Tests for backend/app/chunker.py

Covers:
  - empty string → empty list
  - text shorter than chunk_size → single chunk
  - long text → multiple chunks
  - overlap shifts window correctly
  - chunk_size equal to text length → single chunk
  - zero overlap → contiguous, non-overlapping chunks
  - return type is always list
"""


class TestChunkText:
    def _chunk(self, text, chunk_size=500, overlap=100):
        from backend.app.chunker import chunk_text

        return chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    def test_empty_string_returns_empty_list(self):
        assert self._chunk("") == []

    def test_short_text_is_single_chunk(self):
        result = self._chunk("hello", chunk_size=500, overlap=100)
        assert result == ["hello"]

    def test_long_text_produces_multiple_chunks(self):
        text = "a" * 1000
        result = self._chunk(text, chunk_size=500, overlap=100)
        assert len(result) > 1

    def test_chunks_overlap_correctly(self):
        # chunk_size=10, overlap=5 → windows start at 0, 5, 10, 15
        text = "0123456789ABCDEFGHIJ"  # 20 chars
        result = self._chunk(text, chunk_size=10, overlap=5)
        assert result[0] == "0123456789"
        assert result[1] == "56789ABCDE"

    def test_chunk_size_equals_text_length_gives_one_chunk(self):
        text = "x" * 50
        result = self._chunk(text, chunk_size=50, overlap=0)
        assert result == [text]

    def test_no_overlap_chunks_are_contiguous(self):
        text = "abcdef"
        result = self._chunk(text, chunk_size=2, overlap=0)
        assert "".join(result) == text

    def test_returns_list(self):
        assert isinstance(self._chunk("hello"), list)

    def test_all_chunks_respect_chunk_size(self):
        text = "z" * 999
        result = self._chunk(text, chunk_size=100, overlap=20)
        for chunk in result:
            assert len(chunk) <= 100
