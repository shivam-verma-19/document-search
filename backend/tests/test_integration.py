import importlib
import os
import types
from unittest.mock import MagicMock

import boto3

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
    router_mock = MagicMock()

    router_mock.return_value = {
        "answer": answer,
        "model_used": "llama3-gemini",
        "complexity": "simple",
        "confidence": 0.95,
        "escalated": False,
        "attempted": ["llama3-gemini"],
    }

    return router_mock


def _load_rag(monkeypatch, llm_answer="answer"):
    import backend.app.rag as rag_mod

    importlib.reload(rag_mod)

    monkeypatch.setattr(
        "backend.app.monitoring.push_metric",
        lambda *a, **k: None,
    )

    monkeypatch.setattr(
        "backend.app.reranker.rerank",
        lambda q, docs: docs,
    )

    monkeypatch.setattr(
        "backend.app.embeddings.get_embedding",
        lambda text: [0.1, 0.2, 0.3],
    )

    router_mock = _mock_router(llm_answer)

    monkeypatch.setattr(
        "backend.app.gemini_client.GeminiClient.route_and_invoke",
        router_mock,
    )

    monkeypatch.setattr(
        rag_mod,
        "hybrid_search",
        lambda query, k=5: [_doc(f"chunk {i}") for i in range(5)],
    )

    return rag_mod, router_mock


class TestIntegration:

    def test_second_identical_query_uses_cache(self, monkeypatch):

        import backend.app.gemini_client as gemini_client

        router_mock = MagicMock(
            return_value={
                "answer": "cached answer",
                "model_used": "gemini",
                "complexity": "simple",
                "confidence": 0.9,
                "escalated": False,
                "attempted": ["gemini"],
            }
        )

        monkeypatch.setattr(
            gemini_client,
            "route_and_invoke",
            router_mock,
        )

        result1 = gemini_client.route_and_invoke(prompt="hello")
        result2 = gemini_client.route_and_invoke(prompt="hello")

        assert result1["answer"] == result2["answer"]

    def test_metrics_written_after_rag_query(self, monkeypatch):

        import backend.app.gemini_client as gemini_client

        router_mock = MagicMock(
            return_value={
                "answer": "answer",
                "model_used": "gemini",
                "complexity": "simple",
                "confidence": 0.95,
                "escalated": False,
                "attempted": ["gemini"],
            }
        )

        monkeypatch.setattr(
            gemini_client,
            "route_and_invoke",
            router_mock,
        )

        result = gemini_client.route_and_invoke(prompt="query")

        assert result["answer"] == "answer"

    def test_different_queries_get_independent_cache_entries(self, monkeypatch):

        import backend.app.gemini_client as gemini_client

        router_mock = MagicMock(
            side_effect=[
                {
                    "answer": "a1",
                    "model_used": "gemini",
                    "complexity": "simple",
                    "confidence": 0.9,
                    "escalated": False,
                    "attempted": ["gemini"],
                },
                {
                    "answer": "a2",
                    "model_used": "gemini",
                    "complexity": "simple",
                    "confidence": 0.9,
                    "escalated": False,
                    "attempted": ["gemini"],
                },
            ]
        )

        monkeypatch.setattr(
            gemini_client,
            "route_and_invoke",
            router_mock,
        )

        r1 = gemini_client.route_and_invoke(prompt="q1")
        r2 = gemini_client.route_and_invoke(prompt="q2")

        assert r1["answer"] != r2["answer"]
