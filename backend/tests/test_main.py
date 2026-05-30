import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Env defaults (must happen before any app import)
# ---------------------------------------------------------------------------
os.environ.setdefault("AUTH_DISABLED", "true")

# Mock google.genai before any import so embeddings module loads cleanly
_mock_genai = MagicMock()
_mock_genai_client_instance = MagicMock()
_mock_genai.Client.return_value = _mock_genai_client_instance
import sys

sys.modules.setdefault("google", MagicMock())
sys.modules.setdefault("google.genai", _mock_genai)


class TestMainRoutes:
    @pytest.fixture
    def client(self, monkeypatch):
        import importlib

        monkeypatch.setenv("AUTH_DISABLED", "true")
        monkeypatch.setattr(
            "backend.app.auth.verify_token", lambda token=None: "user-id"
        )
        monkeypatch.setattr("backend.app.auth.verify_cognito_token", lambda: "user-id")
        monkeypatch.setattr("backend.app.auth.optional_auth", lambda: "user-id")
        monkeypatch.setattr("backend.app.rag.ask_question", lambda q: "mock answer")
        monkeypatch.setattr("backend.app.rag.summarize_doc", lambda d: "mock summary")
        monkeypatch.setattr(
            "backend.app.ingest.upload_file_to_s3", lambda f, u: "user/file.txt"
        )
        monkeypatch.setattr("backend.app.ingest.enqueue_file", lambda k, u=None: None)
        monkeypatch.setattr("backend.app.metrics.get_metrics", lambda: [{"id": "1"}])
        monkeypatch.setattr(
            "backend.app.s3_vectors_client.delete_document",
            lambda doc_id: {"result": "deleted", "_id": doc_id},
        )

        import backend.app.main as main_mod

        importlib.reload(main_mod)

        from fastapi.testclient import TestClient

        with TestClient(main_mod.app, raise_server_exceptions=False) as c:
            yield c

    def test_root_returns_running(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_ask_requires_query_param(self, client):
        resp = client.get("/ask", headers={"Authorization": "Bearer fake"})
        assert resp.status_code in (400, 422)

    def test_ask_returns_answer(self, client):
        resp = client.get("/ask?q=hello", headers={"Authorization": "Bearer fake"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "mock answer"

    def test_summary_returns_summary(self, client):
        resp = client.get(
            "/summary?doc_id=abc123", headers={"Authorization": "Bearer fake"}
        )
        assert resp.status_code == 200
        assert resp.json()["summary"] == "mock summary"

    def test_delete_document(self, client):
        resp = client.delete(
            "/document/doc123", headers={"Authorization": "Bearer fake"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_id"] == "doc123"
        assert data["message"] == "deleted"

    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics", headers={"Authorization": "Bearer fake"})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_upload_endpoint(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
            headers={"Authorization": "Bearer fake"},
        )
        assert resp.status_code == 200
        assert resp.json()["message"] == "queued"
