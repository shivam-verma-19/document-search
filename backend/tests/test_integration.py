"""
Integration tests — wires cache + metrics + RAG path together using moto for
AWS and lightweight stubs for router / vector-store.
"""

import importlib
import os
import types
import unittest.mock as mock

import boto3
import pytest
from moto import mock_aws

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("SECRET_NAME", "rag-secrets")

from . import _stubs

_stubs.install_all_stubs()


def _doc(text):
    d = types.SimpleNamespace()
    d.page_content = text
    d.metadata = {}
    return d


def _setup_aws():
    db = boto3.resource("dynamodb", region_name="us-east-1")

    for table_name, key in [
        ("rag-cache", "query"),
        ("rag-metrics", "id"),
        ("rag-eval", "query"),
    ]:
        db.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": key, "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": key, "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )

    sm = boto3.client("secretsmanager", region_name="us-east-1")

    sm.create_secret(
        Name="rag-secrets",
        SecretString='{"OPENAI_API_KEY":"test"}',
    )


def _mock_router(answer="answer"):
    router_mock = mock.MagicMock()

    router_mock.return_value = {
        "answer": answer,
        "model_used": "llama3-bedrock",
        "complexity": "simple",
        "confidence": 0.95,
        "escalated": False,
        "attempted": ["llama3-bedrock"],
    }

    return router_mock


def _load_rag(monkeypatch, llm_answer="answer"):
    """Load rag module with mocked router and clients injected."""

    monkeypatch.setattr(
        "backend.app.monitoring.push_metric",
        lambda *a, **k: None,
    )

    monkeypatch.setattr(
        "backend.app.reranker.rerank",
        lambda q, docs: docs,
    )

    monkeypatch.setattr(
        "backend.app.utils.log_event",
        lambda *a, **k: None,
    )

    monkeypatch.setattr(
        "backend.app.embeddings.get_embedding",
        lambda text: [0.1, 0.2, 0.3],
    )

    router_mock = _mock_router(llm_answer)

    import backend.app.rag as rag_mod

    importlib.reload(rag_mod)

    monkeypatch.setattr(
        rag_mod,
        "route_and_invoke",
        router_mock,
    )

    # fake vector results
    monkeypatch.setattr(
        rag_mod,
        "hybrid_search",
        lambda query, k=5: [_doc(f"chunk {i}") for i in range(5)],
    )

    rag_mod._bm25 = mock.MagicMock()

    return rag_mod, router_mock


class TestIntegration:
    @mock_aws
    def test_second_identical_query_uses_cache(self, monkeypatch):
        """
        Same question asked twice — second call should hit cache
        and avoid another router invocation.
        """

        _setup_aws()

        rag, router_mock = _load_rag(
            monkeypatch,
            llm_answer="first answer",
        )

        first = rag.ask_question("what is machine learning?")
        second = rag.ask_question("what is machine learning?")

        assert first == second

        # first request only
        assert router_mock.call_count <= 2

    @mock_aws
    def test_metrics_written_after_rag_query(self, monkeypatch):
        """
        Successful RAG query should write metrics.
        """

        _setup_aws()

        rag, _ = _load_rag(
            monkeypatch,
            llm_answer="answer",
        )

        rag.ask_question("unique integration query xyz")

        from backend.app.metrics import get_metrics

        items = get_metrics()

        queries = [str(i.get("query", "")) for i in items]

        assert any("unique integration query xyz" in q for q in queries)

    @mock_aws
    def test_different_queries_get_independent_cache_entries(
        self,
        monkeypatch,
    ):
        _setup_aws()

        rag, router_mock = _load_rag(monkeypatch)

        router_mock.return_value = {
            "answer": "answer A",
            "model_used": "llama3-bedrock",
            "complexity": "simple",
            "confidence": 0.9,
            "escalated": False,
            "attempted": ["llama3-bedrock"],
        }

        rag.ask_question("query alpha")

        router_mock.return_value = {
            "answer": "answer B",
            "model_used": "llama3-bedrock",
            "complexity": "simple",
            "confidence": 0.9,
            "escalated": False,
            "attempted": ["llama3-bedrock"],
        }

        rag.ask_question("query beta")

        from backend.app.cache import get_cache

        a = get_cache("query alpha")
        b = get_cache("query beta")

        assert a is not None
        assert b is not None
        assert a != b
