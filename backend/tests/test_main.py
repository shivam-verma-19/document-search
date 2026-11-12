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
        monkeypatch.setattr(
            "backend.app.rag.ask_question", lambda q, *a, **kw: "mock answer"
        )
        monkeypatch.setattr(
            "backend.app.rag.summarize_doc",
            lambda d, user_id=None: "mock summary",
        )
        monkeypatch.setattr(
            "backend.app.ingest.upload_file_to_s3", lambda f, u: "user/file.txt"
        )
        monkeypatch.setattr("backend.app.ingest.enqueue_file", lambda k, u=None: None)
        monkeypatch.setattr(
            "backend.app.metrics.get_metrics",
            lambda user_id=None: [{"id": "1"}],
        )
        monkeypatch.setattr(
            "backend.app.s3_vectors_client.delete_document",
            lambda doc_id, user_id=None: {"result": "deleted", "_id": doc_id},
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

    def test_ask_preflight_allows_local_frontend(self, client):
        resp = client.options(
            "/ask?q=ACP%3F",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"
        assert "authorization" in resp.headers["access-control-allow-headers"].lower()

    def test_ask_requires_query_param(self, client):
        resp = client.get("/ask", headers={"Authorization": "Bearer fake"})
        assert resp.status_code in (400, 422)

    def test_ask_returns_answer(self, client):
        resp = client.get("/ask?q=hello", headers={"Authorization": "Bearer fake"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "mock answer"

    def test_ask_with_session_and_filters(self, client):
        resp = client.get(
            "/ask?q=hello&session_id=s123&file_type=pdf&section=intro",
            headers={"Authorization": "Bearer fake"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "mock answer"
        assert data["session_id"] == "s123"

    def test_delete_session(self, client, monkeypatch):
        monkeypatch.setattr(
            "backend.app.session_service.clear_session", lambda u, s: True
        )
        resp = client.delete("/session/s123", headers={"Authorization": "Bearer fake"})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "s123"
        assert resp.json()["cleared"] is True

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

    def test_upload_requires_authentication(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 401
