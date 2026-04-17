from fastapi import FastAPI, UploadFile, Depends
from .auth import verify_token
from .rag import ask_question, summarize_doc
from .ingestion import process_upload
from .metrics import get_metrics

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/upload")
def upload(file: UploadFile, user=Depends(verify_token)):
    return process_upload(file, user)

@app.get("/ask")
def ask(q: str, user=Depends(verify_token)):
    return {"answer": ask_question(q)}

@app.get("/summary")
def summary(doc_id: str, user=Depends(verify_token)):
    return {"summary": summarize_doc(doc_id)}

@app.get("/metrics")
def metrics(user=Depends(verify_token)):
    return get_metrics()