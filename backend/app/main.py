import asyncio
import logging
import os
import sys

from fastapi import Depends, FastAPI, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from mangum import Mangum
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Must be before any other app code
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
    force=True,
)

from backend.app import ingest

from .auth import optional_auth, verify_cognito_token, verify_token
from .config import get_settings
from .errors import ErrorResponse, RAGException
from .metrics import get_metrics
from .rag import ask_question, summarize_doc

settings = get_settings()
app = FastAPI()


# ─── Exception handlers ───────────────────────────────────────────────────────


@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    error_response = ErrorResponse.from_exception(exc)
    return JSONResponse(
        status_code=error_response.status_code, content={"error": error_response.error}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    error_response = ErrorResponse.from_unknown_error(exc)
    return JSONResponse(
        status_code=error_response.status_code, content={"error": error_response.error}
    )


# ─── Rate limiter ─────────────────────────────────────────────────────────────


def get_user_id_from_request(request: Request):
    return getattr(request.state, "user_id", "anonymous")


limiter = Limiter(key_func=get_user_id_from_request)
app.state.limiter = limiter


# ─── Auth middleware ──────────────────────────────────────────────────────────


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                user = verify_token(token)
                if isinstance(user, str):
                    request.state.user_id = user
                else:
                    request.state.user_id = (
                        user.get("sub")
                        or user.get("email")
                        or user.get("user_id")
                        or "anonymous"
                    )
            else:
                request.state.user_id = "anonymous"
        except Exception:
            request.state.user_id = "anonymous"
        return await call_next(request)


app.add_middleware(AuthMiddleware)
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {"status": "running"}


@limiter.limit("5/minute")
@app.post("/upload")
async def upload(request: Request, file: UploadFile, user=Depends(optional_auth)):
    if isinstance(user, str):
        user_id = user
    else:
        user_id = user.get("sub") or user.get("email") or "anonymous"

    key = await asyncio.to_thread(ingest.upload_file_to_s3, file, user_id)
    await asyncio.to_thread(ingest.enqueue_file, key, user_id)
    return {"message": "queued", "key": key}


@limiter.limit("10/minute")
@app.get("/ask")
def ask(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500),
    user: dict = Depends(verify_cognito_token),
):
    answer = ask_question(q)
    if answer.startswith("Error:") or "all model tiers failed" in answer:
        return JSONResponse(status_code=503, content={"answer": answer})
    return {"answer": answer}


@limiter.limit("20/minute")
@app.get("/summary")
def summary(
    request: Request,
    doc_id: str = Query(..., min_length=1, max_length=100),
    user: dict = Depends(verify_cognito_token),
):
    return {"summary": summarize_doc(doc_id)}


@limiter.limit("10/minute")
@app.delete("/document/{doc_id}")
def delete_document(
    request: Request,
    doc_id: str,
    user: dict = Depends(verify_cognito_token),
):
    """
    Delete a document from the vector store.
    Previously missing — without this, stale/outdated docs polluted search forever.
    """
    from .s3_vectors_client import delete_document as _delete

    result = _delete(doc_id)
    return {"message": "deleted", "doc_id": doc_id, "result": result}


@limiter.limit("30/minute")
@app.get("/metrics")
def metrics(request: Request, user: dict = Depends(verify_cognito_token)):
    return get_metrics()


# ─── Lambda handler ───────────────────────────────────────────────────────────

handler = Mangum(app)
