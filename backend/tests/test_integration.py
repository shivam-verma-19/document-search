"""
Integration tests — wires cache + metrics + RAG path together using moto for
AWS and lightweight stubs for LLM / vector-store.
"""

import importlib
import json
import os
import sys
import types
import unittest.mock as mock

import boto3
import pytest
from moto import mock_aws

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("SECRET_NAME", "rag-secrets")

from backend.tests._stubs import install_all_stubs

install_all_stubs()


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
        SecretString=json.dumps({"OPENAI_API_KEY": "sk-test"}),
    )


def _load_rag(monkeypatch, llm_answer="answer"):
    """Load rag module with mocked clients injected."""
    monkeypatch.setattr("backend.app.monitoring.push_metric", lambda *a: None)
    monkeypatch.setattr("backend.app.reranker.rerank", lambda q, docs: docs)
    monkeypatch.setattr("backend.app.utils.log_event", lambda *a: None)
    monkeypatch.setattr(
        "backend.app.utils.get_secrets", lambda: {"OPENAI_API_KEY": "sk-test"}
    )

    llm = mock.MagicMock()
    llm.invoke.return_value = mock.MagicMock(content=llm_answer)

    vdb = mock.MagicMock()
    vdb.similarity_search.return_value = [_doc(f"chunk {i}") for i in range(5)]

    bm25 = mock.MagicMock()
    bm25.search.return_value = []

    import backend.app.rag as rag_mod

    importlib.reload(rag_mod)
    rag_mod._get_clients.cache_clear()
    rag_mod.llm = llm
    rag_mod.vector_db = vdb
    rag_mod.bm25 = bm25

    return rag_mod


class TestIntegration:
    @mock_aws
    def test_second_identical_query_uses_cache(self, monkeypatch):
        """Same question asked twice — the second call must hit DynamoDB cache."""
        _setup_aws()
        rag = _load_rag(monkeypatch, llm_answer="first answer")

        first = rag.ask_question("what is machine learning?")
        second = rag.ask_question("what is machine learning?")

        assert first == second
        # LLM should only fire for rewrite + one generate (not twice)
        assert rag.llm.invoke.call_count <= 3

    @mock_aws
    def test_metrics_written_after_rag_query(self, monkeypatch):
        """A successful RAG query must persist at least one metrics row."""
        _setup_aws()
        rag = _load_rag(monkeypatch, llm_answer="answer")

        rag.ask_question("unique integration query xyz")

        from backend.app.metrics import get_metrics

        items = get_metrics()
        queries = [i["query"] for i in items]
        assert any("unique integration query xyz" in q for q in queries)

    @mock_aws
    def test_different_queries_get_independent_cache_entries(self, monkeypatch):
        _setup_aws()
        rag = _load_rag(monkeypatch)

        rag.llm.invoke.return_value = mock.MagicMock(content="answer A")
        rag.ask_question("query alpha")

        rag.llm.invoke.return_value = mock.MagicMock(content="answer B")
        rag.ask_question("query beta")

        # Both are in cache independently
        from backend.app.cache import get_cache

        a = get_cache("query alpha")
        b = get_cache("query beta")
        assert a is not None
        assert b is not None
        assert a != b
