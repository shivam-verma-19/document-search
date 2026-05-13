"""
Tests for backend/app/auth.py

Covers:
  - verify_token with a valid token returns user id
  - verify_token raises 401 when called with no / falsy token

NOTE: The current implementation accepts ANY non-empty token
(no real JWT validation). These tests document and verify that
behaviour so a regression is caught if real validation is added.
"""

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials


class TestVerifyToken:
    def _make_token(self, credentials="Bearer some-token"):
        """Create an HTTPAuthorizationCredentials-like stub."""
        scheme, _, token = credentials.partition(" ")
        return HTTPAuthorizationCredentials(scheme=scheme, credentials=token)

    def test_valid_token_returns_user_id(self):
        from backend.app.auth import verify_token

        result = verify_token(self._make_token("Bearer abc123"))
        assert result == "user-id"

    def test_any_non_empty_token_accepted(self):
        from backend.app.auth import verify_token

        result = verify_token(self._make_token("Bearer totally-fake-token"))
        assert result is not None

    def test_none_token_raises_401(self):
        from backend.app.auth import verify_token

        with pytest.raises(HTTPException) as exc_info:
            verify_token(None)
        assert exc_info.value.status_code == 401

    def test_empty_string_token_raises_401(self):
        """Empty string is falsy — should also be rejected."""
        from backend.app.auth import verify_token

        with pytest.raises(HTTPException) as exc_info:
            verify_token("")
        assert exc_info.value.status_code == 401


class TestVerifyCognitoToken:
    def test_no_credentials_raises_401(self):
        from backend.app.auth import verify_cognito_token

        with pytest.raises(HTTPException) as exc_info:
            verify_cognito_token(credentials=None)
        assert exc_info.value.status_code == 401

    def test_valid_credentials_returns_user_id(self):
        from fastapi.security import HTTPAuthorizationCredentials

        from backend.app.auth import verify_cognito_token

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="token")
        result = verify_cognito_token(credentials=creds)
        assert result == "user-id"


class TestOptionalAuth:
    def test_no_credentials_returns_user_id_not_401(self):
        """optional_auth must NOT raise when there is no token."""
        from backend.app.auth import optional_auth

        result = optional_auth(credentials=None)
        assert result == "user-id"

    def test_with_credentials_returns_user_id(self):
        from fastapi.security import HTTPAuthorizationCredentials

        from backend.app.auth import optional_auth

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="some-token")
        result = optional_auth(credentials=creds)
        assert result == "user-id"
