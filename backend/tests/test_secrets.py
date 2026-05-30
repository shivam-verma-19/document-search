import json
from unittest.mock import patch

from aiohttp import ClientError

import backend.app.secrets as m


def _reset():
    m._cache.clear()


class TestGetSecret:
    def test_env_var_takes_precedence(self, monkeypatch):
        _reset()
        monkeypatch.setenv("MY_SECRET_KEY", "env-value")
        val = m.get_secret("MY_SECRET_KEY")
        assert val == "env-value"

    def test_returns_empty_string_when_missing(self, monkeypatch):
        _reset()
        monkeypatch.delenv("NONEXISTENT_KEY_XYZ", raising=False)
        from botocore.exceptions import ClientError as BotocoreClientError

        with patch("boto3.client") as mock_boto:
            mock_boto.return_value.get_secret_value.side_effect = BotocoreClientError(
                {
                    "Error": {
                        "Code": "ResourceNotFoundException",
                        "Message": "not found",
                    }
                },
                "GetSecretValue",
            )
            val = m.get_secret("NONEXISTENT_KEY_XYZ")
        assert val == ""

    def test_loads_from_secrets_manager(self, monkeypatch):
        _reset()
        monkeypatch.delenv("SOME_REMOTE_KEY", raising=False)
        with patch("boto3.client") as mock_boto:
            mock_boto.return_value.get_secret_value.return_value = {
                "SecretString": json.dumps({"SOME_REMOTE_KEY": "secret-value"})
            }
            val = m.get_secret("SOME_REMOTE_KEY")
        assert val == "secret-value"
