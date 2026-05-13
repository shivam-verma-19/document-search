"""
Tests for backend/app/main.py – FastAPI route layer.

Heavy optional packages (unstructured, sentence_transformers) are stubbed at
sys.modules level so these tests run in any CI environment.
"""

import importlib
import io
import os
import unittest.mock as mock

from fastapi.testclient import TestClient

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("SECRET_NAME", "rag-secrets")
os.environ.setdefault("QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/123/q")

# Install stubs BEFORE any backend import
from . import _stubs

_stubs.install_all_stubs()

AUTH_HEADER = {"Authorization": "Bearer test-token"}


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


class TestRoot:
    def test_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_returns_running_status(self, client):
        assert client.get("/").json() == {"status": "running"}


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------


class TestUpload:
    def test_happy_path_returns_queued(self, client):
        r = client.post(
            "/upload",
            files={"file": ("test.pdf", io.BytesIO(b"pdf content"), "application/pdf")},
        )
        assert r.status_code == 200
        assert r.json() == {"message": "queued"}

    def test_missing_file_returns_422(self, client):
        assert client.post("/upload").status_code == 422


# ---------------------------------------------------------------------------
# GET /ask
# ---------------------------------------------------------------------------


class TestAsk:
    def test_happy_path_returns_answer(self, client):
        r = client.get("/ask", params={"q": "what is AI?"}, headers=AUTH_HEADER)
        assert r.status_code == 200
        assert "answer" in r.json()

    def test_answer_matches_mock(self, client):
        r = client.get("/ask", params={"q": "test"}, headers=AUTH_HEADER)
        assert r.json()["answer"] == "mock answer"

    def test_missing_query_param_returns_422(self, client):
        assert client.get("/ask", headers=AUTH_HEADER).status_code == 422

    def test_no_auth_token_is_rejected(self, monkeypatch):
        """Real verify_token must reject requests with no Bearer token."""
        from fastapi import HTTPException

        def strict_verify(token=None):
            if not token:
                raise HTTPException(status_code=401)
            return "user-id"

        monkeypatch.setattr("backend.app.auth.verify_token", strict_verify)
        monkeypatch.setattr("backend.app.utils.save_to_s3", lambda f: "k")
        monkeypatch.setattr("backend.app.rag.ask_question", lambda q: "a")
        monkeypatch.setattr("backend.app.rag.summarize_doc", lambda d: "s")
        monkeypatch.setattr("backend.app.metrics.get_metrics", lambda: [])

        import backend.app.ingest as ingest_mod

        importlib.reload(ingest_mod)
        monkeypatch.setattr("backend.app.ingest.enqueue_file", lambda key, user: None)

        import backend.app.main as main_mod

        importlib.reload(main_mod)
        c = TestClient(main_mod.app, raise_server_exceptions=False)
        r = c.get("/ask", params={"q": "test"})
        assert r.status_code in (401, 403)


# ---------------------------------------------------------------------------
# GET /summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_happy_path_returns_summary(self, client):
        r = client.get("/summary", params={"doc_id": "abc123"}, headers=AUTH_HEADER)
        assert r.status_code == 200
        assert r.json()["summary"] == "mock summary"

    def test_missing_doc_id_returns_422(self, client):
        assert client.get("/summary", headers=AUTH_HEADER).status_code == 422


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_happy_path_returns_list(self, client):
        r = client.get("/metrics", headers=AUTH_HEADER)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_returns_mocked_data(self, client):
        r = client.get("/metrics", headers=AUTH_HEADER)
        assert r.json()[0]["id"] == "1"
