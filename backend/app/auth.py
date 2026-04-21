from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()


def verify_token(token=Depends(security)):
    if not token:
        raise HTTPException(status_code=401)
    return "user-id"
