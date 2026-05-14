import json

from backend.app import embeddings


class FakeBody:
    def read(self):
        return json.dumps({"embedding": [0.1, 0.2]}).encode()


class FakeClient:
    def invoke_model(self, **kwargs):
        return {"body": FakeBody()}


class TestGetEmbedding:
    def test_returns_list(self, monkeypatch):
        monkeypatch.setattr(
            embeddings,
            "_get_client",
            lambda: FakeClient(),
        )

        result = embeddings.get_embedding("hello")

        assert isinstance(result, list)
        assert result == [0.1, 0.2]
