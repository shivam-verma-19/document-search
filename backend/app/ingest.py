import boto3
import json, os

from unstructured.partition.auto import partition
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

s3 = boto3.client("s3")
sqs = boto3.client("sqs")
QUEUE_URL = os.environ["QUEUE_URL"]


def process_upload(file, user):
    key = f"{user}/{file.filename}"
    s3.upload_fileobj(file.file, "rag-upload-bucket", key)

    elements = partition(file=file.file)
    text = "\n".join([el.text for el in elements if el.text])

    docs = [Document(page_content=text)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    PineconeVectorStore.from_documents(chunks, embeddings, index_name="rag-index")

    return {"message": "Uploaded & processed"}


def enqueue_file(file_name):
    sqs.send_message(QueueUrl=QUEUE_URL, MessageBody=json.dumps({"file": file_name}))
