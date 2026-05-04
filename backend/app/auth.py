import os
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

security = HTTPBearer(auto_error=False)

# Assume jwk_client may or may not be initialized
jwk_client = None  # your existing setup


def verify_token(token: str):
    # ✅ TEST MODE: allow any non-empty token
    if not jwk_client:
        if not token:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return "user-id"

    try:
        signing_key = jwk_client.get_signing_key_from_jwt(token).key

        claims = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            audience=os.getenv("COGNITO_CLIENT_ID"),
        )

        return claims

    except JWTError:
        raise HTTPException(status_code=401, detail="Unauthorized")


def verify_cognito_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    # ✅ No token → reject
    if credentials is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = credentials.credentials

    if not jwk_client:
        return "user-id"

    return verify_token(token)


def optional_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials is None:
        return "user-id"

    token = credentials.credentials

    if not jwk_client:
        return "user-id"

    return verify_token(token)
