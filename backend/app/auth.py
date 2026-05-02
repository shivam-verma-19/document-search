from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError, PyJWKClient, decode

from .config import get_settings

security = HTTPBearer()
settings = get_settings()

COGNITO_ISSUER = (
    f"https://cognito-idp.{settings.aws_region}.amazonaws.com/{settings.cognito_user_pool_id}"
)
JWKS_URL = f"{COGNITO_ISSUER}/.well-known/jwks.json"
jwk_client = PyJWKClient(JWKS_URL)


def verify_cognito_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict[str, Any]:
    token = credentials.credentials
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
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc