import logging
import os
import threading
import time
from typing import Union

import requests
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwk, jwt

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

# FIX: no longer initialised at module import time — built lazily on first
# token verification so tests can import the module without network access.
_JWK_CLIENT = None
_JWK_CLIENT_INITIALISED = False


class CognitoJWKClient:
    """Fetch and cache Cognito signing keys."""

    def __init__(self, jwks_url: str, cache_ttl: int = 3600):
        self.jwks_url = jwks_url
        self._cache_ttl = cache_ttl
        self._jwks_cache = None
        self._cache_expires_at = 0.0
        self._cache_lock = threading.Lock()

    def _get_jwks(self, force_refresh: bool = False):
        with self._cache_lock:
            if (
                not force_refresh
                and self._jwks_cache is not None
                and time.monotonic() < self._cache_expires_at
            ):
                return self._jwks_cache

            try:
                resp = requests.get(self.jwks_url, timeout=(3, 5))
                resp.raise_for_status()
                jwks = resp.json()
            except (requests.RequestException, ValueError) as exc:
                logger.error("Failed to fetch Cognito JWKS: %s", exc)
                raise

            self._jwks_cache = jwks
            self._cache_expires_at = time.monotonic() + self._cache_ttl
            return jwks

    def get_signing_key_from_jwt(self, token: str):
        try:
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")
            if not kid:
                raise JWTError("Token missing 'kid' in header")
        except JWTError:
            raise

        for force_refresh in (False, True):
            jwks = self._get_jwks(force_refresh=force_refresh)
            for key_data in jwks.get("keys", []):
                if key_data.get("kid") == kid:
                    return jwk.construct(
                        key_data, algorithm=key_data.get("alg", "RS256")
                    )

        raise JWTError("Unable to find the token signing key")


def _get_jwk_client():
    """Lazily initialise the JWK client on first call."""
    global _JWK_CLIENT, _JWK_CLIENT_INITIALISED

    if _JWK_CLIENT_INITIALISED:
        return _JWK_CLIENT

    _JWK_CLIENT_INITIALISED = True

    if os.getenv("AUTH_DISABLED", "false").lower() == "true":
        logger.info("Authentication disabled via AUTH_DISABLED=true")
        return None

    from .config import get_settings

    settings = get_settings()

    if not settings.cognito_user_pool_id or not settings.aws_region:
        logger.error(
            "Cannot initialise JWK_CLIENT: missing COGNITO_USER_POOL_ID or AWS_REGION. "
            "Either set these or enable AUTH_DISABLED=true"
        )
        return None

    try:
        jwks_url = (
            f"https://cognito-idp.{settings.aws_region}.amazonaws.com/"
            f"{settings.cognito_user_pool_id}/.well-known/jwks.json"
        )
        _JWK_CLIENT = CognitoJWKClient(jwks_url)
        logger.info(
            f"JWK_CLIENT initialised for user pool {settings.cognito_user_pool_id}"
        )
    except Exception as e:
        logger.error(f"Failed to initialise JWK_CLIENT: {e}")

    return _JWK_CLIENT


# Keep module-level alias for backwards compatibility with tests that patch it.
JWK_CLIENT = None


def verify_token(
    token: Union[str, HTTPAuthorizationCredentials, None],
):
    if isinstance(token, HTTPAuthorizationCredentials):
        token = token.credentials

    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    client = JWK_CLIENT or _get_jwk_client()

    if not client:
        if os.getenv("AUTH_DISABLED", "false").lower() == "true":
            return "user-id"
        raise HTTPException(status_code=500, detail="Auth provider not configured")

    try:
        signing_key = client.get_signing_key_from_jwt(token)

        client_id = os.getenv("COGNITO_CLIENT_ID")
        if not client_id:
            from .config import get_settings

            client_id = get_settings().cognito_client_id
        if not client_id:
            logger.error("COGNITO_CLIENT_ID is not configured")
            raise HTTPException(status_code=500, detail="Auth provider not configured")

        from .config import get_settings

        settings = get_settings()
        issuer = (
            f"https://cognito-idp.{settings.aws_region}.amazonaws.com/"
            f"{settings.cognito_user_pool_id}"
        )
        claims = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            issuer=issuer,
            options={"verify_aud": False},
        )

        if (
            claims.get("token_use") != "access"
            or claims.get("client_id") != client_id
            or not claims.get("sub")
        ):
            raise JWTError("Invalid Cognito access token")

        return claims

    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Unauthorized") from exc
    except (requests.RequestException, ValueError) as exc:
        logger.error("Cognito signing keys are unavailable: %s", exc)
        raise HTTPException(
            status_code=503, detail="Authentication provider unavailable"
        ) from exc


def verify_cognito_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = credentials.credentials

    client = JWK_CLIENT or _get_jwk_client()

    if not client:
        if os.getenv("AUTH_DISABLED", "false").lower() == "true":
            return "user-id"
        raise HTTPException(status_code=500, detail="Auth provider not configured")

    return verify_token(token)


def optional_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials is None:
        return "user-id"

    token = credentials.credentials

    client = JWK_CLIENT or _get_jwk_client()

    if not client:
        if os.getenv("AUTH_DISABLED", "false").lower() == "true":
            return "user-id"
        raise HTTPException(status_code=500, detail="Auth provider not configured")

    return verify_token(token)
