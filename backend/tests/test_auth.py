import importlib
import os

import pytest
from fastapi import HTTPException

os.environ.setdefault("AUTH_DISABLED", "true")


class TestVerifyToken:
    def test_no_token_raises_401(self):
        import backend.app.auth as auth_mod

        importlib.reload(auth_mod)
        auth_mod.JWK_CLIENT = None
        auth_mod._JWK_CLIENT_INITIALISED = False
        with pytest.raises(HTTPException) as exc:
            auth_mod.verify_token(None)
        assert exc.value.status_code == 401

    def test_empty_string_raises_401(self):
        import backend.app.auth as auth_mod

        importlib.reload(auth_mod)
        auth_mod.JWK_CLIENT = None
        auth_mod._JWK_CLIENT_INITIALISED = False
        with pytest.raises(HTTPException) as exc:
            auth_mod.verify_token("")
        assert exc.value.status_code == 401

    def test_auth_disabled_no_jwk_returns_user_id(self, monkeypatch):
        monkeypatch.setenv("AUTH_DISABLED", "true")
        import backend.app.auth as auth_mod

        importlib.reload(auth_mod)
        auth_mod.JWK_CLIENT = None
        auth_mod._JWK_CLIENT = None
        auth_mod._JWK_CLIENT_INITIALISED = False
        result = auth_mod.verify_token("any-token")
        assert result == "user-id"

    def test_no_jwk_auth_enabled_raises_500(self, monkeypatch):
        monkeypatch.setenv("AUTH_DISABLED", "false")
        import backend.app.auth as auth_mod

        importlib.reload(auth_mod)
        auth_mod.JWK_CLIENT = None
        auth_mod._JWK_CLIENT = None
        auth_mod._JWK_CLIENT_INITIALISED = (
            True  # skip lazy init, simulate missing config
        )
        with pytest.raises(HTTPException) as exc:
            auth_mod.verify_token("some-token")
        assert exc.value.status_code == 500

    def test_credentials_object_extracted(self, monkeypatch):
        monkeypatch.setenv("AUTH_DISABLED", "true")
        import backend.app.auth as auth_mod

        importlib.reload(auth_mod)
        auth_mod.JWK_CLIENT = None
        auth_mod._JWK_CLIENT = None
        auth_mod._JWK_CLIENT_INITIALISED = False
        from fastapi.security import HTTPAuthorizationCredentials

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="tok")
        result = auth_mod.verify_token(creds)
        assert result == "user-id"

    def test_accepts_access_token_for_configured_client(self, monkeypatch):
        from unittest.mock import MagicMock, patch
        from types import SimpleNamespace

        import backend.app.auth as auth_mod

        monkeypatch.setenv("AUTH_DISABLED", "false")
        monkeypatch.setenv("COGNITO_CLIENT_ID", "app-client-id")
        client = MagicMock()
        client.get_signing_key_from_jwt.return_value = "signing-key"
        monkeypatch.setattr(auth_mod, "JWK_CLIENT", client)
        claims = {
            "token_use": "access",
            "client_id": "app-client-id",
            "sub": "user-id",
        }

        with patch(
            "backend.app.config.get_settings",
            return_value=SimpleNamespace(
                aws_region="ap-south-1", cognito_user_pool_id="pool-id"
            ),
        ), patch("backend.app.auth.jwt.decode", return_value=claims) as decode:
            result = auth_mod.verify_token("access-token")

        assert result == claims
        decode.assert_called_once_with(
            "access-token",
            "signing-key",
            algorithms=["RS256"],
            issuer="https://cognito-idp.ap-south-1.amazonaws.com/pool-id",
            options={"verify_aud": False},
        )

    @pytest.mark.parametrize(
        "claims",
        [
            {"token_use": "id", "aud": "app-client-id"},
            {"token_use": "access", "client_id": "another-client"},
        ],
    )
    def test_rejects_non_access_tokens_or_wrong_client(self, monkeypatch, claims):
        from unittest.mock import MagicMock, patch

        import backend.app.auth as auth_mod

        monkeypatch.setenv("AUTH_DISABLED", "false")
        monkeypatch.setenv("COGNITO_CLIENT_ID", "app-client-id")
        client = MagicMock()
        client.get_signing_key_from_jwt.return_value = "signing-key"
        monkeypatch.setattr(auth_mod, "JWK_CLIENT", client)

        with patch("backend.app.auth.jwt.decode", return_value=claims):
            with pytest.raises(HTTPException) as exc:
                auth_mod.verify_token("some-token")

        assert exc.value.status_code == 401


class TestVerifyCognitoToken:
    def test_no_credentials_raises_401(self):
        import backend.app.auth as auth_mod

        importlib.reload(auth_mod)
        with pytest.raises(HTTPException) as exc:
            auth_mod.verify_cognito_token(credentials=None)
        assert exc.value.status_code == 401

    def test_auth_disabled_returns_user_id(self, monkeypatch):
        monkeypatch.setenv("AUTH_DISABLED", "true")
        import backend.app.auth as auth_mod

        importlib.reload(auth_mod)
        auth_mod.JWK_CLIENT = None
        auth_mod._JWK_CLIENT = None
        auth_mod._JWK_CLIENT_INITIALISED = False
        from fastapi.security import HTTPAuthorizationCredentials

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="tok")
        result = auth_mod.verify_cognito_token(credentials=creds)
        assert result == "user-id"


class TestOptionalAuth:
    def test_no_credentials_returns_user_id(self):
        import backend.app.auth as auth_mod

        importlib.reload(auth_mod)
        result = auth_mod.optional_auth(credentials=None)
        assert result == "user-id"

    def test_with_credentials_auth_disabled_returns_user_id(self, monkeypatch):
        monkeypatch.setenv("AUTH_DISABLED", "true")
        import backend.app.auth as auth_mod

        importlib.reload(auth_mod)
        auth_mod.JWK_CLIENT = None
        auth_mod._JWK_CLIENT = None
        auth_mod._JWK_CLIENT_INITIALISED = False
        from fastapi.security import HTTPAuthorizationCredentials

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="tok")
        result = auth_mod.optional_auth(credentials=creds)
        assert result == "user-id"


class TestCognitoJWKClient:
    def test_jwks_are_cached(self):
        from unittest.mock import MagicMock, patch

        import backend.app.auth as auth_mod

        client = auth_mod.CognitoJWKClient("https://example.com/jwks.json")
        response = MagicMock()
        response.json.return_value = {"keys": []}
        with patch("backend.app.auth.requests.get", return_value=response) as get:
            assert client._get_jwks() == {"keys": []}
            assert client._get_jwks() == {"keys": []}

        get.assert_called_once_with(
            "https://example.com/jwks.json",
            timeout=(3, 5),
        )

    def test_missing_kid_raises(self):
        from unittest.mock import MagicMock, patch

        import backend.app.auth as auth_mod

        client = auth_mod.CognitoJWKClient("https://example.com/jwks.json")
        fake_token = "eyJhbGciOiJSUzI1NiJ9.e30.sig"
        with patch("backend.app.auth.jwt.get_unverified_header", return_value={}):
            with pytest.raises(Exception):
                client.get_signing_key_from_jwt(fake_token)

    def test_missing_kid_in_jwks_raises(self):
        from unittest.mock import MagicMock, patch

        import backend.app.auth as auth_mod

        client = auth_mod.CognitoJWKClient("https://example.com/jwks.json")
        with patch(
            "backend.app.auth.jwt.get_unverified_header", return_value={"kid": "abc"}
        ), patch.object(client, "_get_jwks", return_value={"keys": [{"kid": "other"}]}):
            with pytest.raises(Exception, match="Unable to find the token signing key"):
                client.get_signing_key_from_jwt("tok")
