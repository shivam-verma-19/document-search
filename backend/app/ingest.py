import boto3
from langchain.schema import Document
from unstructured.partition.auto import partition
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

s3 = boto3.client("s3")

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