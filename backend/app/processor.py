# backend/app/processor.py

from io import BytesIO

import boto3
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition

from .config import get_settings

settings = get_settings()
s3 = boto3.client("s3")

# =========================
# ✅ IDEMPOTENCY (TEMP - IN MEMORY)
# =========================
processed_files = set()


def already_processed(key: str) -> bool:
    return key in processed_files


def mark_processed(key: str):
    processed_files.add(key)


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
    # PARSE FILE CONTENT
    # =========================
    file_obj = BytesIO(content)

    elements = partition(file=file_obj)

    text = "\n".join([el.text for el in elements if el.text])

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
    embeddings = OpenAIEmbeddings()

    PineconeVectorStore.from_documents(chunks, embeddings, index_name="rag-index")

    # =========================
    # MARK AS PROCESSED
    # =========================
    mark_processed(key)

    return {
        "status": "processed",
        "chunks": len(chunks),
        "key": key,
    }
