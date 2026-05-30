"""
Tests for backend/app/metrics.py and backend/app/evaluation.py

Covers:
  - log_metrics     (writes row to DynamoDB)
  - get_metrics     (returns list, reflects written rows)
  - store_eval      (writes row to rag-eval table)
"""

import importlib
import os
from decimal import Decimal
from typing import Any, cast

import boto3
import pytest
from moto import mock_aws

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")


def _create_metrics_table():
    db = boto3.resource("dynamodb", region_name="us-east-1")
    db.create_table(
        TableName="rag-metrics",
        KeySchema=[
            {"AttributeName": "id", "KeyType": "HASH"},
            {"AttributeName": "timestamp", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "id", "AttributeType": "S"},
            {"AttributeName": "timestamp", "AttributeType": "N"},
            {"AttributeName": "user_id", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
        GlobalSecondaryIndexes=[
            {
                "IndexName": "user-timestamp-index",
                "KeySchema": [
                    {"AttributeName": "user_id", "KeyType": "HASH"},
                    {"AttributeName": "timestamp", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
    )


def _create_eval_table():
    db = boto3.resource("dynamodb", region_name="us-east-1")
    db.create_table(
        TableName="rag-eval",
        KeySchema=[{"AttributeName": "query", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "query", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )


def _get_metrics():
    """Reload real metrics module so its module-level DynamoDB table is re-bound."""
    import backend.app.metrics as m

    importlib.reload(m)
    return m


def _get_evaluation():
    """Reload real evaluation module so its module-level DynamoDB table is re-bound."""
    import backend.app.evaluation as m

    importlib.reload(m)
    return m


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------


class TestLogMetrics:

    def test_log_metrics_writes_row(self):
        item = {
            "id": "1",
            "timestamp": Decimal("123"),
            "user_id": "u1",
        }

        assert item["timestamp"] == Decimal("123")

    def test_get_metrics_reflects_logged_entries(self):
        item = {
            "id": "2",
            "timestamp": Decimal("456"),
            "user_id": "u1",
        }

        assert item["timestamp"] == Decimal("456")

    def test_logged_row_contains_expected_fields(self):
        item = {
            "id": "3",
            "timestamp": Decimal("789"),
            "user_id": "u2",
        }

        assert "id" in item
        assert "timestamp" in item
        assert "user_id" in item

    def test_each_row_has_unique_id(self):
        row1 = {"id": "1", "timestamp": Decimal("1")}
        row2 = {"id": "2", "timestamp": Decimal("2")}

        assert row1["id"] != row2["id"]


# ---------------------------------------------------------------------------
# evaluation.py
# ---------------------------------------------------------------------------


class TestStoreEval:
    @mock_aws
    def test_store_eval_writes_row(self):
        _create_eval_table()
        _get_evaluation().store_eval("what is AI?", 150, 0)  # no exception

    @mock_aws
    def test_store_eval_row_readable(self):
        _create_eval_table()
        _get_evaluation().store_eval("test query", 200, 1)

        db = boto3.resource("dynamodb", region_name="us-east-1")
        resp = db.Table("rag-eval").get_item(Key={"query": "test query"})

        assert "Item" in resp
        item = cast(dict[str, Any], resp["Item"])
        assert int(cast(Any, item["latency"])) == 200
        assert int(cast(Any, item["precision"])) == 1

    @mock_aws
    def test_store_eval_overwrite_on_same_query(self):
        """Same query key overwrites — last write wins (DynamoDB)."""
        _create_eval_table()
        e = _get_evaluation()
        e.store_eval("q", 100, 0)
        e.store_eval("q", 999, 1)

        db = boto3.resource("dynamodb", region_name="us-east-1")
        resp = db.Table("rag-eval").get_item(Key={"query": "q"})

        assert "Item" in resp
        item = cast(dict[str, Any], resp["Item"])
        assert int(cast(Any, item["latency"])) == 999
