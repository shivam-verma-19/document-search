import asyncio
import logging
import os
import sys
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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

from .auth import verify_token
from .config import get_settings
from .errors import ErrorResponse, RAGException
from .metrics import get_metrics
from .rag import ask_question, ask_question_stream, summarize_doc

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
        request.state.auth_user = None
        request.state.auth_error = None
        request.state.user_id = "anonymous"

        auth_header = request.headers.get("Authorization", "")
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() == "bearer" and token:
            try:
                user = verify_token(token)
            except HTTPException as exc:
                if exc.status_code != 401:
                    raise
                request.state.auth_error = exc
            else:
                request.state.auth_user = user
                request.state.user_id = (
                    user if isinstance(user, str) else user.get("sub", "anonymous")
                )

        return await call_next(request)


app.add_middleware(AuthMiddleware)
app.add_middleware(SlowAPIMiddleware)
cors_allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


# ─── Routes ───────────────────────────────────────────────────────────────────


def get_current_user(request: Request):
    if request.state.auth_error is not None:
        raise request.state.auth_error
    if request.state.auth_user is not None:
        return request.state.auth_user
    raise HTTPException(status_code=401, detail="Unauthorized")


def get_user_id(user) -> str:
    if isinstance(user, str):
        return user
    return user.get("sub", "")


@app.get("/")
def root():
    return {"status": "running"}


@limiter.limit("5/minute")
@app.post("/upload")
async def upload(
    request: Request,
    file: UploadFile,
    user=Depends(get_current_user),
):
    user_id = get_user_id(user)
    key = await asyncio.to_thread(ingest.upload_file_to_s3, file, user_id)
    await asyncio.to_thread(ingest.enqueue_file, key, user_id)
    return {"message": "queued", "key": key}


@limiter.limit("10/minute")
@app.get("/ask")
def ask(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500),
    session_id: Optional[str] = Query(default=None),
    file_type: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    section: Optional[str] = Query(default=None),
    user=Depends(get_current_user),
):
    user_id = get_user_id(user)
    filters = {"user_id": user_id}
    if file_type:
        filters["file_type"] = file_type
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to
    if section:
        filters["section"] = section

    answer = ask_question(
        q,
        user_id=user_id,
        session_id=session_id,
        filters=filters or None,
    )
    if answer.startswith("Error:") or "all model tiers failed" in answer:
        return JSONResponse(status_code=503, content={"answer": answer})
    return {"answer": answer, "session_id": session_id}


@limiter.limit("10/minute")
@app.get("/ask/stream")
async def ask_stream(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500),
    session_id: Optional[str] = Query(default=None),
    file_type: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    section: Optional[str] = Query(default=None),
    user=Depends(get_current_user),
):
    """
    Streaming ask — returns tokens via Server-Sent Events as they arrive.

    Client reads the stream until it receives "data: [DONE]".

    Example (JavaScript):
        const es = new EventSource('/ask/stream?q=your+question');
        es.onmessage = (e) => {
          if (e.data === '[DONE]') { es.close(); return; }
          output.textContent += e.data.replace(/\\\\n/g, '\\n');
        };
    """
    user_id = get_user_id(user)
    filters = {"user_id": user_id}
    if file_type:
        filters["file_type"] = file_type
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to
    if section:
        filters["section"] = section

    def _generate():
        for chunk in ask_question_stream(
            q,
            user_id=user_id,
            session_id=session_id,
            filters=filters or None,
        ):
            yield chunk

    return StreamingResponse(
        asyncio.to_thread(_generate) if False else _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@limiter.limit("20/minute")
@app.delete("/session/{session_id}")
def delete_session(
    request: Request,
    session_id: str,
    user=Depends(get_current_user),
):
    from .session_service import clear_session

    user_id = get_user_id(user)
    cleared = clear_session(user_id, session_id)
    return {"cleared": cleared, "session_id": session_id}


@limiter.limit("20/minute")
@app.get("/summary")
def summary(
    request: Request,
    doc_id: str = Query(..., min_length=1, max_length=100),
    user=Depends(get_current_user),
):
    return {"summary": summarize_doc(doc_id, user_id=get_user_id(user))}


@limiter.limit("10/minute")
@app.delete("/document/{doc_id}")
def delete_document(
    request: Request,
    doc_id: str,
    user=Depends(get_current_user),
):
    """
    Delete a document from the vector store.
    Previously missing — without this, stale/outdated docs polluted search forever.
    """
    from .s3_vectors_client import delete_document as _delete

    user_id = get_user_id(user)
    result = _delete(doc_id, user_id=user_id)
    if result["result"] == "not_found":
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "deleted", "doc_id": doc_id, "result": result}


@limiter.limit("30/minute")
@app.get("/metrics")
def metrics(request: Request, user=Depends(get_current_user)):
    return get_metrics(user_id=get_user_id(user))


@limiter.limit("10/minute")
@app.get("/eval")
def eval_records(
    request: Request,
    window_hours: int = Query(default=24, ge=1, le=720),
    user=Depends(get_current_user),
):
    """
    Return eval records (faithfulness, relevance, recall@k, MRR) for the
    last `window_hours` hours. Used for dashboard visibility and regression
    detection.
    """
    from .eval import get_eval_records

    records = get_eval_records(
        window_seconds=window_hours * 3600,
        user_id=get_user_id(user),
    )
    return {"records": records, "count": len(records), "window_hours": window_hours}


# ─── Lambda handler ───────────────────────────────────────────────────────────

handler = Mangum(app)
