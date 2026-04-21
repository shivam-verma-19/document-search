import io
import json
import os

import boto3
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition

s3 = boto3.client("s3")  # type: ignore
sqs = boto3.client("sqs")
QUEUE_URL = os.environ["QUEUE_URL"]


def process_upload(file, user):
    key = f"{user}/{file.filename}"

    # Read file content once so we can reuse it for both S3 and partitioning.
    file_bytes = file.file.read()

    s3.upload_fileobj(io.BytesIO(file_bytes), "rag-upload-bucket", key)

    elements = partition(file=io.BytesIO(file_bytes))
    text = "\n".join([el.text for el in elements if el.text])

    docs = [Document(page_content=text)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    PineconeVectorStore.from_documents(chunks, embeddings, index_name="rag-index")

    return {"message": "Uploaded & processed"}


def enqueue_file(file_name):
    sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=json.dumps({"file": file_name}))
