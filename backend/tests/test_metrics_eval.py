"""
Tests for backend/app/metrics.py and backend/app/evaluation.py

Covers:
  - log_metrics     (writes row to DynamoDB)
  - get_metrics     (returns list, reflects written rows)
  - store_eval      (writes row to rag-eval table)
"""

import os

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
        KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )


def _create_eval_table():
    db = boto3.resource("dynamodb", region_name="us-east-1")
    db.create_table(
        TableName="rag-eval",
        KeySchema=[{"AttributeName": "query", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "query", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST",
    )


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------


class TestLogMetrics:
    @mock_aws
    def test_log_metrics_writes_row(self):
        _create_metrics_table()
        from backend.app.metrics import log_metrics

        log_metrics("test query", 250, "rag")  # should not raise

    @mock_aws
    def test_get_metrics_empty_initially(self):
        _create_metrics_table()
        from backend.app.metrics import get_metrics

        items = get_metrics()
        assert items == []

    @mock_aws
    def test_get_metrics_reflects_logged_entries(self):
        _create_metrics_table()
        from backend.app.metrics import get_metrics, log_metrics

        log_metrics("q1", 100, "rag")
        log_metrics("q2", 200, "cache")
        items = get_metrics()
        assert len(items) == 2

    @mock_aws
    def test_logged_row_contains_expected_fields(self):
        _create_metrics_table()
        from backend.app.metrics import get_metrics, log_metrics

        log_metrics("my query", 300, "llm")
        items = get_metrics()
        row = items[0]
        assert row["query"] == "my query"
        assert int(row["latency"]) == 300
        assert row["source"] == "llm"

    @mock_aws
    def test_each_row_has_unique_id(self):
        _create_metrics_table()
        from backend.app.metrics import get_metrics, log_metrics

        log_metrics("q", 10, "rag")
        log_metrics("q", 20, "rag")
        items = get_metrics()
        ids = {item["id"] for item in items}
        assert len(ids) == 2  # two distinct UUIDs


# ---------------------------------------------------------------------------
# evaluation.py
# ---------------------------------------------------------------------------


class TestStoreEval:
    @mock_aws
    def test_store_eval_writes_row(self):
        _create_eval_table()
        from backend.app.evaluation import store_eval

        store_eval("what is AI?", 150, 0)  # no exception

    @mock_aws
    def test_store_eval_row_readable(self):
        _create_eval_table()
        from backend.app.evaluation import store_eval

        store_eval("test query", 200, 1)

        db = boto3.resource("dynamodb", region_name="us-east-1")
        table = db.Table("rag-eval")
        resp = table.get_item(Key={"query": "test query"})
        item = resp["Item"]
        assert int(item["latency"]) == 200
        assert int(item["precision"]) == 1

    @mock_aws
    def test_store_eval_overwrite_on_same_query(self):
        """Same query key overwrites — last write wins (DynamoDB)."""
        _create_eval_table()
        from backend.app.evaluation import store_eval

        store_eval("q", 100, 0)
        store_eval("q", 999, 1)

        db = boto3.resource("dynamodb", region_name="us-east-1")
        table = db.Table("rag-eval")
        resp = table.get_item(Key={"query": "q"})
        assert int(resp["Item"]["latency"]) == 999
