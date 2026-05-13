"""
Tests for backend/app/config.py and backend/app/embeddings.py

config covers:
  - Settings._parse_list: comma-separated, JSON list, empty string, whitespace, casing
  - Settings properties: allowed_upload_extensions_list, allowed_upload_mimes_list,
    forbidden_upload_patterns_list

embeddings covers:
  - get_embedding dispatches to openai_embedding when USE_BEDROCK=false
  - get_embedding dispatches to bedrock_embedding when USE_BEDROCK=true
"""

import importlib
import os

# ===========================================================================
# config – Settings._parse_list and properties
# ===========================================================================


class TestSettingsParseList:
    def test_comma_separated_string_parsed(self):
        from backend.app.config import Settings

        s = Settings(allowed_upload_extensions="pdf,txt,docx")
        assert s.allowed_upload_extensions_list == ["pdf", "txt", "docx"]

    def test_json_list_string_parsed(self):
        from backend.app.config import Settings

        s = Settings(allowed_upload_extensions='["pdf", "txt"]')
        assert s.allowed_upload_extensions_list == ["pdf", "txt"]

    def test_empty_string_returns_empty_list(self):
        from backend.app.config import Settings

        s = Settings(allowed_upload_extensions="")
        assert s.allowed_upload_extensions_list == []

    def test_values_are_lowercased(self):
        from backend.app.config import Settings

        s = Settings(allowed_upload_extensions="PDF,TXT")
        assert s.allowed_upload_extensions_list == ["pdf", "txt"]

    def test_whitespace_around_items_is_stripped(self):
        from backend.app.config import Settings

        s = Settings(allowed_upload_extensions=" pdf , txt ")
        assert s.allowed_upload_extensions_list == ["pdf", "txt"]

    def test_allowed_mimes_list_property(self):
        from backend.app.config import Settings

        s = Settings(allowed_upload_mimes="application/pdf,text/plain")
        assert "application/pdf" in s.allowed_upload_mimes_list
        assert "text/plain" in s.allowed_upload_mimes_list

    def test_forbidden_patterns_list_property(self):
        from backend.app.config import Settings

        s = Settings(forbidden_upload_patterns="virus,malware")
        assert "virus" in s.forbidden_upload_patterns_list

    def test_forbidden_patterns_empty_by_default(self):
        from backend.app.config import Settings

        s = Settings(forbidden_upload_patterns="")
        assert s.forbidden_upload_patterns_list == []

    def test_json_list_with_empty_items_filtered(self):
        from backend.app.config import Settings

        s = Settings(allowed_upload_extensions='["pdf", "", "txt"]')
        assert "" not in s.allowed_upload_extensions_list


# ===========================================================================
# embeddings – get_embedding dispatch
# ===========================================================================


class TestGetEmbeddingDispatch:
    def test_uses_openai_when_use_bedrock_false(self, monkeypatch):
        monkeypatch.setenv("USE_BEDROCK", "false")

        import backend.app.embeddings as emb

        importlib.reload(emb)

        called = {}

        def fake_openai(text):
            called["fn"] = "openai"
            return [0.1]

        monkeypatch.setattr(emb, "openai_embedding", fake_openai)
        emb.get_embedding("hello")

        assert called.get("fn") == "openai"

    def test_uses_bedrock_when_use_bedrock_true(self, monkeypatch):
        monkeypatch.setenv("USE_BEDROCK", "true")

        import backend.app.embeddings as emb

        importlib.reload(emb)

        called = {}

        def fake_bedrock(text):
            called["fn"] = "bedrock"
            return [0.2]

        monkeypatch.setattr(emb, "bedrock_embedding", fake_bedrock)
        emb.get_embedding("world")

        assert called.get("fn") == "bedrock"

    def test_returns_list(self, monkeypatch):
        monkeypatch.setenv("USE_BEDROCK", "false")

        import backend.app.embeddings as emb

        importlib.reload(emb)
        monkeypatch.setattr(emb, "openai_embedding", lambda t: [0.5, 0.6])

        result = emb.get_embedding("test")
        assert isinstance(result, list)
