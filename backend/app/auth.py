from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()


def verify_token(token=Depends(security)):
    if not token:
        raise HTTPException(status_code=401)
    return "user-id"
