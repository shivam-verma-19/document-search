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

    def test_false_token_raises_401(self):
        from backend.app.auth import verify_token
        with pytest.raises(HTTPException) as exc_info:
            verify_token(False)
        assert exc_info.value.status_code == 401

    def test_empty_string_token_raises_401(self):
        """Empty string is falsy — should also be rejected."""
        from backend.app.auth import verify_token
        with pytest.raises(HTTPException) as exc_info:
            verify_token("")
        assert exc_info.value.status_code == 401
