import os

from fastapi import Depends, FastAPI, Query, UploadFile
from mangum import Mangum

from .auth import verify_cognito_token
from .config import get_settings
from .ingest import enqueue_file, upload_file_to_s3
from .metrics import get_metrics
from .rag import ask_question, summarize_doc

settings = get_settings()
app = FastAPI()


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/upload")
def upload(
    file: UploadFile,
    user: dict = Depends(verify_cognito_token),
):
    user_id = user.get("sub") or user.get("email") or "anonymous"
    key = upload_file_to_s3(file, user_id)
    enqueue_file(settings.bucket_name, key, user_id)
    return {"status": "queued", "key": key}


@app.get("/ask")
def ask(
    q: str = Query(..., min_length=1, max_length=500),
    user: dict = Depends(verify_cognito_token),
):
    return {"answer": ask_question(q)}


@app.get("/summary")
def summary(
    doc_id: str = Query(..., min_length=1, max_length=100),
    user: dict = Depends(verify_cognito_token),
):
    return {"summary": summarize_doc(doc_id)}


@app.get("/metrics")
def metrics(user: dict = Depends(verify_cognito_token)):
    return get_metrics()


handler = Mangum(app)