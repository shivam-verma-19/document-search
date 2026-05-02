import json
import os

import boto3
from io import BytesIO
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition


s3 = boto3.client("s3")  # type: ignore
sqs = boto3.client("sqs")
QUEUE_URL = os.environ["QUEUE_URL"]

def process_upload(file, user):
    # ✅ Read ONCE
    content = file.file.read()

    # ✅ Reset original stream (important for tests)
    file.file.seek(0)

    # ✅ Upload using fresh stream
    s3.upload_fileobj(BytesIO(content), "rag-upload-bucket", f"{user}/{file.filename}")

    # ✅ Use separate stream for parsing
    elements = partition(file=BytesIO(content))

    text = "\n".join([el.text for el in elements if getattr(el, "text", None)])

    if not text.strip():
        return {"message": "Empty document"}

    docs = [Document(page_content=text)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        return {"message": "No chunks generated"}

    embeddings = OpenAIEmbeddings()

    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name="rag-index"
    )

    return {"message": "Uploaded & processed"}


def enqueue_file(file_name):
    sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=json.dumps({"file": file_name}))
