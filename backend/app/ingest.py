import json
import os
from io import BytesIO

import boto3
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition

s3 = boto3.client("s3")  # type: ignore
sqs = boto3.client("sqs")
QUEUE_URL = os.environ["QUEUE_URL"]
BUCKET_NAME = os.environ.get("BUCKET_NAME", "rag-pipeline-upload-bucket")


def upload_file_to_s3(file, user):
    content = file.file.read()
    file.file.seek(0)

    key = f"{user}/{file.filename}"
    s3.upload_fileobj(BytesIO(content), BUCKET_NAME, key)

    return key


def enqueue_file(bucket, key, user):
    sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps({"bucket": bucket, "key": key, "user": user}),
    )


def process_s3_upload(bucket, key, user=None):
    obj = s3.get_object(Bucket=bucket, Key=key)
    content = obj["Body"].read()

    if not content.strip():
        return {"message": "Empty document"}

    elements = partition(file=BytesIO(content))
    text = "\n".join([el.text for el in elements if getattr(el, "text", None)])

    if not text.strip():
        return {"message": "Empty document"}

    docs = [Document(page_content=text)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    if not chunks:
        return {"message": "No chunks generated"}

    embeddings = OpenAIEmbeddings()
    PineconeVectorStore.from_documents(chunks, embeddings, index_name="rag-index")

    return {"message": "Uploaded & processed", "key": key}


def process_upload(file, user):
    key = upload_file_to_s3(file, user)
    enqueue_file(BUCKET_NAME, key, user)
    return {"message": "Queued", "key": key}
