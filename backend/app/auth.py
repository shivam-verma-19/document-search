# backend/app/auth.py

import os
from typing import Any, Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient, decode
from jwt.exceptions import InvalidTokenError

from .config import get_settings

settings = get_settings()
security = HTTPBearer()

COGNITO_ISSUER = (
    f"https://cognito-idp.{settings.aws_region}.amazonaws.com/"
    f"{settings.cognito_user_pool_id}"
)

JWKS_URL = f"{COGNITO_ISSUER}/.well-known/jwks.json"

jwk_client: Optional[PyJWKClient] = None

if settings.cognito_user_pool_id:
    jwk_client = PyJWKClient(JWKS_URL)


# =========================
# ✅ CORE LOGIC (REUSABLE)
# =========================
def verify_token_logic(token: str) -> dict[str, Any]:
    # ✅ TEST MODE
    if os.getenv("ENV") == "test":
        if not token:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return {"user_id": "test-user"}

    # ✅ CONFIG CHECK
    if jwk_client is None:
        raise HTTPException(
            status_code=500,
            detail="Auth not configured properly",
        )

    try:
        signing_key = jwk_client.get_signing_key_from_jwt(token).key

        claims = decode(
            token,
            signing_key,
            algorithms=["RS256"],
            audience=settings.cognito_client_id,
            issuer=COGNITO_ISSUER,
        )

        return claims

    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Unauthorized")


# =========================
# ✅ FASTAPI DEPENDENCY
# =========================
def verify_cognito_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict[str, Any]:
    return verify_token_logic(credentials.credentials)
