"""
Tests for backend/app/cache.py

Covers:
  - hash_query        (determinism, collision-resistance basics)
  - get_cache         (miss, hit)
  - set_cache         (write then read back)
  - cache roundtrip   (set → get equality)
  - overwrite         (set twice, last wins)
"""

import os

import boto3
import pytest
from moto import mock_aws

os.environ.setdefault("AWS_DEFAULT_REGION", "ap-south-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")


def _make_table():
    db = boto3.resource("dynamodb", region_name="ap-south-1")
    db.create_table(
        TableName="rag-cache",
        KeySchema=[{"AttributeName": "query", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "query", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )
    return db.Table("rag-cache")


# ---------------------------------------------------------------------------
# hash_query
# ---------------------------------------------------------------------------


class TestHashQuery:
    def test_deterministic(self):
        from backend.app.cache import hash_query

        assert hash_query("hello") == hash_query("hello")

    def test_different_inputs_different_hashes(self):
        from backend.app.cache import hash_query

        assert hash_query("hello") != hash_query("world")

    def test_returns_string(self):
        from backend.app.cache import hash_query

        assert isinstance(hash_query("test"), str)

    def test_hex_digest_length(self):
        from backend.app.cache import hash_query

        # SHA-256 produces 64 hex characters
        assert len(hash_query("anything")) == 64

    def test_empty_string_hashed(self):
        from backend.app.cache import hash_query

        result = hash_query("")
        assert isinstance(result, str) and len(result) == 64


# ---------------------------------------------------------------------------
# get_cache / set_cache
# ---------------------------------------------------------------------------


class TestCacheOperations:
    @mock_aws
    def test_cache_miss_returns_none(self):
        _make_table()
        from backend.app.cache import get_cache

        assert get_cache("nonexistent query") is None

    @mock_aws
    def test_set_then_get_returns_answer(self):
        _make_table()
        from backend.app.cache import get_cache, set_cache

        set_cache("what is AI?", "AI is artificial intelligence.")
        assert get_cache("what is AI?") == "AI is artificial intelligence."

    @mock_aws
    def test_overwrite_returns_latest(self):
        _make_table()
        from backend.app.cache import get_cache, set_cache

        set_cache("q", "first answer")
        set_cache("q", "second answer")
        assert get_cache("q") == "second answer"

    @mock_aws
    def test_different_queries_independent(self):
        _make_table()
        from backend.app.cache import get_cache, set_cache

        set_cache("query A", "answer A")
        set_cache("query B", "answer B")
        assert get_cache("query A") == "answer A"
        assert get_cache("query B") == "answer B"

    @mock_aws
    def test_cache_stores_long_answer(self):
        _make_table()
        from backend.app.cache import get_cache, set_cache

        long_answer = "word " * 500
        set_cache("long q", long_answer)
        assert get_cache("long q") == long_answer

    @mock_aws
    def test_whitespace_sensitive(self):
        """'hello' and ' hello' are different cache keys."""
        _make_table()
        from backend.app.cache import get_cache, set_cache

        set_cache("hello", "a")
        assert get_cache(" hello") is None
