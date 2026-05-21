# backend/app/processor.py
import uuid
from io import BytesIO

import boto3
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import get_settings
from .embeddings import get_embedding
from .faiss_client import index_document

settings = get_settings()
s3 = boto3.client("s3")  # type: ignore

# =========================
# ✅ IDEMPOTENCY (TEMP - IN MEMORY)
# =========================
processed_files = set()


def already_processed(key: str) -> bool:
    return key in processed_files


def mark_processed(key: str):
    processed_files.add(key)


# =========================
# ✅ FILE EXTRACTION
# =========================
def extract_text_from_file(content: bytes, filename: str) -> str:
    """Extract text from file based on extension."""
    if filename.lower().endswith(".pdf"):
        return _extract_from_pdf(content)
    else:
        # Fallback for txt, csv, etc. — read as plain text
        return content.decode("utf-8", errors="ignore")


def _extract_from_pdf(content: bytes) -> str:
    """Extract text from PDF using PyPDF."""
    try:
        reader = PdfReader(BytesIO(content))
        text = "\n".join([page.extract_text() for page in reader.pages])
        return text
    except Exception as e:
        # Log or handle PDF parse errors
        print(f"Error parsing PDF: {e}")
        return ""


# =========================
# ✅ MAIN PROCESSOR
# =========================
def process_file_from_s3(bucket: str, key: str):
    # 🔁 Prevent duplicate processing
    if already_processed(key):
        return {"status": "skipped", "key": key}

    # =========================
    # DOWNLOAD FILE FROM S3
    # =========================
    obj = s3.get_object(Bucket=bucket, Key=key)
    content = obj["Body"].read()

    # =========================
    # EXTRACT TEXT FROM FILE
    # =========================
    text = extract_text_from_file(content, key)

    if not text.strip():
        return {"status": "empty", "key": key}

    # =========================
    # CHUNKING
    # =========================
    docs = [Document(page_content=text)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    chunks = splitter.split_documents(docs)

    # =========================
    # EMBEDDINGS + VECTOR STORE
    # =========================
    doc_id = str(uuid.uuid4())

    for idx, chunk in enumerate(chunks):
        embedding = get_embedding(chunk.page_content)

        index_document(
            doc_id=doc_id,
            text=chunk.page_content,
            embedding=embedding,
            metadata={
                "user_id": key.split("/")[0],
                "chunk_id": idx,
            },
        )

    # =========================
    # MARK AS PROCESSED
    # =========================
    mark_processed(key)

    return {
        "status": "processed",
        "chunks": len(chunks),
        "key": key,
    }
