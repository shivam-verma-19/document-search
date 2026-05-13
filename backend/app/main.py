from fastapi import Depends, FastAPI, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from mangum import Mangum
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app import ingest

from .auth import optional_auth, verify_cognito_token, verify_token
from .config import get_settings
from .metrics import get_metrics
from .rag import ask_question, summarize_doc

settings = get_settings()

app = FastAPI()


# =========================
# ✅ USER-BASED KEY FUNCTION
# =========================
def get_user_id_from_request(request: Request):
    return getattr(request.state, "user_id", "anonymous")


limiter = Limiter(key_func=get_user_id_from_request)
app.state.limiter = limiter


# =========================
# ✅ AUTH MIDDLEWARE
# =========================
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


# =========================
# ✅ REGISTER MIDDLEWARE
# =========================
app.add_middleware(AuthMiddleware)
app.add_middleware(SlowAPIMiddleware)


# =========================
# ✅ RATE LIMIT HANDLER
# =========================
@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"},
    )


# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    return {"status": "running"}


@limiter.limit("5/minute")
@app.post("/upload")
def upload(
    request: Request,
    file: UploadFile,
    user=Depends(optional_auth),
):
    if isinstance(user, str):
        user_id = user
    else:
        user_id = user.get("sub") or user.get("email") or "anonymous"

    key = ingest.upload_file_to_s3(file, user_id)
    ingest.enqueue_file(key, user_id)

    return {"message": "queued"}


@limiter.limit("10/minute")
@app.get("/ask")
def ask(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500),
    user: dict = Depends(verify_cognito_token),
):
    return {"answer": ask_question(q)}


@limiter.limit("20/minute")
@app.get("/summary")
def summary(
    request: Request,
    doc_id: str = Query(..., min_length=1, max_length=100),
    user: dict = Depends(verify_cognito_token),
):
    return {"summary": summarize_doc(doc_id)}


@limiter.limit("30/minute")
@app.get("/metrics")
def metrics(request: Request, user: dict = Depends(verify_cognito_token)):
    return get_metrics()


# =========================
# LAMBDA HANDLER
# =========================
handler = Mangum(app)
