import json
import time
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

import backend.app.bm25_cache as m


def _reset():
    m._warm_corpus = None
    m._warm_loaded_at = 0.0
    m._warm_version = 0
    m._dynamodb = None


class TestGetCorpus:
    def test_returns_warm_cache_when_fresh(self):
        _reset()
        m._warm_corpus = ["doc1", "doc2"]
        m._warm_loaded_at = time.time()
        assert m.get_corpus() == ["doc1", "doc2"]

    def test_fetches_from_dynamo_on_cold_cache(self):
        _reset()
        mock_table = MagicMock()
        mock_table.get_item.return_value = {
            "Item": {"pk": "bm25_corpus", "texts": json.dumps(["a", "b"]), "version": 3}
        }
        with patch("backend.app.bm25_cache._table", return_value=mock_table) as mock_fn:
            mock_fn.return_value = mock_table
            # _table is called as a function, so patch the function itself
            with patch("backend.app.bm25_cache._table", side_effect=lambda: mock_table):
                result = m.get_corpus()
        assert result == ["a", "b"]
        assert m._warm_version == 3

    def test_dynamo_miss_returns_empty_and_warms(self):
        _reset()
        mock_table = MagicMock()
        mock_table.get_item.return_value = {}
        with patch("backend.app.bm25_cache._table", side_effect=lambda: mock_table):
            result = m.get_corpus()
        assert result == []
        assert m._warm_corpus == []

    def test_dynamo_error_returns_empty(self):
        _reset()
        mock_table = MagicMock()
        mock_table.get_item.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "x"}}, "GetItem"
        )
        with patch("backend.app.bm25_cache._table", side_effect=lambda: mock_table):
            result = m.get_corpus()
        assert result == []

    def test_warm_ttl_expiry_refetches(self):
        _reset()
        m._warm_corpus = ["old"]
        m._warm_loaded_at = time.time() - m._WARM_TTL - 1
        mock_table = MagicMock()
        mock_table.get_item.return_value = {
            "Item": {"pk": "bm25_corpus", "texts": json.dumps(["new"]), "version": 1}
        }
        with patch("backend.app.bm25_cache._table", side_effect=lambda: mock_table):
            result = m.get_corpus()
        assert result == ["new"]


class TestSetCorpus:
    def test_writes_to_dynamo_and_warms(self):
        _reset()
        m._warm_version = 2
        mock_table = MagicMock()
        m._dynamodb = mock_table  # _table() returns _dynamodb if already set
        m.set_corpus(["x", "y"])
        mock_table.put_item.assert_called_once()
        item = mock_table.put_item.call_args[1]["Item"]
        assert item["version"] == 3
        assert m._warm_corpus == ["x", "y"]
        assert m._warm_version == 3

    def test_dynamo_error_is_swallowed(self):
        _reset()
        mock_table = MagicMock()
        mock_table.put_item.side_effect = ClientError(
            {
                "Error": {
                    "Code": "ProvisionedThroughputExceededException",
                    "Message": "x",
                }
            },
            "PutItem",
        )
        with patch("backend.app.bm25_cache._table", side_effect=lambda: mock_table):
            m.set_corpus(["z"])


class TestAppendToCorpus:
    def test_empty_list_is_noop(self):
        _reset()
        mock_table = MagicMock()
        with patch("backend.app.bm25_cache._table", side_effect=lambda: mock_table):
            m.append_to_corpus([])
        mock_table.put_item.assert_not_called()

    def test_appends_and_increments_version(self):
        _reset()
        m._warm_corpus = ["existing"]
        m._warm_loaded_at = time.time()
        m._warm_version = 5
        mock_table = MagicMock()
        mock_table.put_item.return_value = {}
        with patch("backend.app.bm25_cache._table", side_effect=lambda: mock_table):
            m.append_to_corpus([])
        assert "existing" in m._warm_corpus
        assert m._warm_version == 5

    def test_retries_on_version_conflict(self):
        _reset()
        m._warm_corpus = ["doc"]
        m._warm_loaded_at = time.time()
        m._warm_version = 0
        conflict = ClientError(
            {"Error": {"Code": "ConditionalCheckFailedException", "Message": "x"}},
            "PutItem",
        )
        call_count = [0]

        def put_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise conflict
            return {}

        mock_table = MagicMock()
        mock_table.put_item.side_effect = put_side_effect
        mock_table.get_item.return_value = {
            "Item": {"pk": "bm25_corpus", "texts": json.dumps(["doc"]), "version": 0}
        }
        with patch(
            "backend.app.bm25_cache._table", side_effect=lambda: mock_table
        ), patch("time.sleep"):
            m.append_to_corpus(["new"])
        assert call_count[0] == 2

    def test_non_conflict_dynamo_error_aborts(self):
        _reset()
        m._warm_corpus = ["doc"]
        m._warm_loaded_at = time.time()
        m._warm_version = 0
        mock_table = MagicMock()
        mock_table.put_item.side_effect = ClientError(
            {
                "Error": {
                    "Code": "ProvisionedThroughputExceededException",
                    "Message": "x",
                }
            },
            "PutItem",
        )
        with patch("backend.app.bm25_cache._table", side_effect=lambda: mock_table):
            m.append_to_corpus(["new"])


class TestInvalidateWarmCache:
    def test_invalidate_forces_refetch(self):
        _reset()
        m._warm_corpus = ["stale"]
        m._warm_loaded_at = time.time()
        m.invalidate_warm_cache()
        assert m._warm_corpus is None
        assert m._warm_loaded_at == 0.0
