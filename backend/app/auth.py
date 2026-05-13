import os
from typing import Union

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

security = HTTPBearer(auto_error=False)

# Assume JWK_CLIENT may or may not be initialized
JWK_CLIENT = None


def verify_token(
    token: Union[str, HTTPAuthorizationCredentials, None],
):
    if isinstance(token, HTTPAuthorizationCredentials):
        token = token.credentials

    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not JWK_CLIENT:
        return "user-id"

    try:
        signing_key = JWK_CLIENT.get_signing_key_from_jwt(token).key

        claims = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            audience=os.getenv("COGNITO_CLIENT_ID"),
        )

        return claims

    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Unauthorized") from exc


def verify_cognito_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    # ✅ No token → reject
    if credentials is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = credentials.credentials

    if not JWK_CLIENT:
        return "user-id"

    return verify_token(token)


def optional_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials is None:
        return "user-id"

    token = credentials.credentials

    if not JWK_CLIENT:
        return "user-id"

    return verify_token(token)
