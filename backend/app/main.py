import os

from fastapi import Depends, FastAPI, UploadFile
from mangum import Mangum

from .auth import verify_token
from .ingest import enqueue_file, upload_file_to_s3

app = FastAPI()

BUCKET_NAME = os.environ.get("BUCKET_NAME", "rag-pipeline-upload-bucket")


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/upload")
def upload(file: UploadFile, user=Depends(verify_token)):
    key = upload_file_to_s3(file, user)
    enqueue_file(BUCKET_NAME, key, user)
    return {"status": "queued", "key": key}


@app.get("/ask")
def ask(q: str, user=Depends(verify_token)):
    from .rag import ask_question

    return {"answer": ask_question(q)}


@app.get("/summary")
def summary(doc_id: str, user=Depends(verify_token)):
    from .rag import summarize_doc

    return {"summary": summarize_doc(doc_id)}


@app.get("/metrics")
def metrics(user=Depends(verify_token)):
    from .metrics import get_metrics

    return get_metrics()


handler = Mangum(app)
