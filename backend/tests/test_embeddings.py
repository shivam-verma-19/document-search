import json

from backend.app import embeddings


class FakeBody:
    def read(self):
        return json.dumps({"embedding": [0.1, 0.2]}).encode()


class FakeClient:
    def __init__(self):
        self.last_kwargs = {}

    def invoke_model(self, **kwargs):
        self.last_kwargs = kwargs
        return {"body": FakeBody()}


class TestGetEmbedding:
    def test_returns_list(self, monkeypatch):
        monkeypatch.setattr(embeddings, "_get_client", lambda: FakeClient())
        result = embeddings.get_embedding("hello")
        assert isinstance(result, list)
        assert result == [0.1, 0.2]

    def test_uses_titan_embed_v2(self, monkeypatch):
        """Ensure we're using v2 (available in ap-south-1), not v1."""
        client = FakeClient()
        monkeypatch.setattr(embeddings, "_get_client", lambda: client)
        embeddings.get_embedding("test")
        assert client.last_kwargs["modelId"] == "amazon.titan-embed-text-v2:0"

    def test_passes_input_text(self, monkeypatch):
        client = FakeClient()
        monkeypatch.setattr(embeddings, "_get_client", lambda: client)
        embeddings.get_embedding("hello world")
        body = json.loads(client.last_kwargs["body"])
        assert body["inputText"] == "hello world"

    def test_empty_string_input(self, monkeypatch):
        monkeypatch.setattr(embeddings, "_get_client", lambda: FakeClient())
        result = embeddings.get_embedding("")
        assert isinstance(result, list)
