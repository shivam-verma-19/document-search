from unittest.mock import MagicMock, patch

import pytest

import backend.app.s3_vectors_client as m


def _reset():
    m._client = None


class TestIndexDocument:
    def test_raises_without_embedding(self):
        _reset()
        with pytest.raises(ValueError, match="Embedding required"):
            m.index_document("id1", "text")

    def test_raises_on_dimension_mismatch(self):
        _reset()
        m.EXPECTED_DIMENSION = 768
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            m.index_document("id1", "text", embedding=[0.1, 0.2])

    def test_calls_put_vectors_and_returns_id(self):
        _reset()
        m.EXPECTED_DIMENSION = 3
        mock_client = MagicMock()
        with patch.object(m, "_get_client", return_value=mock_client):
            result = m.index_document("doc1", "hello", embedding=[0.1, 0.2, 0.3])
        mock_client.put_vectors.assert_called_once()
        assert result["_id"] == "doc1"
        assert result["result"] == "created"

    def test_metadata_merged_into_vector(self):
        _reset()
        m.EXPECTED_DIMENSION = 2
        mock_client = MagicMock()
        with patch.object(m, "_get_client", return_value=mock_client):
            m.index_document("d", "txt", embedding=[0.1, 0.2], metadata={"user": "u1"})
        call_kwargs = mock_client.put_vectors.call_args[1]
        vec_meta = call_kwargs["Vectors"][0]["Metadata"]
        assert vec_meta["user"] == "u1"
        assert vec_meta["text"] == "txt"


class TestSearchDocuments:
    def test_parses_response_correctly(self):
        _reset()
        mock_client = MagicMock()
        mock_client.query_vectors.return_value = {
            "Vectors": [
                {"Key": "k1", "Metadata": {"text": "hello", "source": "file.txt"}},
                {"Key": "k2", "Metadata": {"text": "world"}},
            ]
        }
        with patch.object(m, "_get_client", return_value=mock_client):
            result = m.search_documents([0.1] * 768, k=2)
        assert len(result) == 2
        assert result[0]["_id"] == "k1"
        assert result[0]["_source"]["text"] == "hello"
        assert result[0]["_source"]["metadata"]["source"] == "file.txt"
        assert "text" not in result[0]["_source"]["metadata"]

    def test_empty_response_returns_empty_list(self):
        _reset()
        mock_client = MagicMock()
        mock_client.query_vectors.return_value = {"Vectors": []}
        with patch.object(m, "_get_client", return_value=mock_client):
            result = m.search_documents([0.1] * 768, k=5)
        assert result == []


class TestSearchSimilar:
    def test_returns_texts_only(self):
        _reset()
        mock_client = MagicMock()
        mock_client.query_vectors.return_value = {
            "Vectors": [
                {"Key": "k1", "Metadata": {"text": "chunk one"}},
                {"Key": "k2", "Metadata": {"text": "chunk two"}},
            ]
        }
        with patch.object(m, "_get_client", return_value=mock_client):
            result = m.search_similar([0.1] * 768, k=2)
        assert result == ["chunk one", "chunk two"]


class TestGetDocument:
    def test_returns_none_when_not_found(self):
        _reset()
        mock_client = MagicMock()
        mock_client.get_vectors.return_value = {"Vectors": []}
        with patch.object(m, "_get_client", return_value=mock_client):
            assert m.get_document("missing") is None

    def test_returns_document_dict(self):
        _reset()
        mock_client = MagicMock()
        mock_client.get_vectors.return_value = {
            "Vectors": [{"Key": "k1", "Metadata": {"text": "content", "tag": "x"}}]
        }
        with patch.object(m, "_get_client", return_value=mock_client):
            result = m.get_document("k1")
        assert result["_id"] == "k1"  # type: ignore
        assert result["_source"]["text"] == "content"  # type: ignore


class TestDeleteDocument:
    def test_deletes_chunk_key_directly(self):
        _reset()
        mock_client = MagicMock()
        with patch.object(m, "_get_client", return_value=mock_client):
            result = m.delete_document("base#0")
        mock_client.delete_vectors.assert_called_once()
        assert result["result"] == "deleted"
        assert result["deleted_count"] == 1

    def test_not_found_returns_not_found(self):
        _reset()
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [{"Vectors": []}]
        mock_client.get_paginator.return_value = mock_paginator
        with patch.object(m, "_get_client", return_value=mock_client):
            result = m.delete_document("nonexistent-base-id")
        assert result["result"] == "not_found"
        assert result["deleted_count"] == 0

    def test_deletes_all_chunks_for_base_id(self):
        _reset()
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Vectors": [
                    {"Key": "base#0"},
                    {"Key": "base#1"},
                    {"Key": "other#0"},
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator
        with patch.object(m, "_get_client", return_value=mock_client):
            result = m.delete_document("base")
        assert result["deleted_count"] == 2


class TestGetAllDocuments:
    def test_returns_non_empty_texts(self):
        _reset()
        mock_client = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Vectors": [
                    {"Metadata": {"text": "text1"}},
                    {"Metadata": {"text": "text2"}},
                    {"Metadata": {}},
                    {"Metadata": {"text": ""}},
                ]
            }
        ]
        mock_client.get_paginator.return_value = mock_paginator
        with patch.object(m, "_get_client", return_value=mock_client):
            result = m.get_all_documents()
        assert result == ["text1", "text2"]


class TestGenerateDocId:
    def test_ids_are_unique(self):
        ids = {m.generate_doc_id() for _ in range(10)}
        assert len(ids) == 10
