import os

import boto3
import requests
from requests_aws4auth import AWS4Auth

region = os.getenv("AWS_REGION", "ap-south-1")
service = "aoss"

session = boto3.Session()
credentials = session.get_credentials()

auth = None

if credentials:
    frozen = credentials.get_frozen_credentials()

    auth = AWS4Auth(
        frozen.access_key,
        frozen.secret_key,
        region,
        service,
        session_token=frozen.token,
    )

ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")
INDEX = "documents"


def index_document(doc_id, user_id, chunk_id, text, embedding):
    url = f"{ENDPOINT}/{INDEX}/_doc/{doc_id}_{chunk_id}"

    body = {
        "text": text,
        "embedding": embedding,
        "doc_id": doc_id,
        "user_id": user_id,
        "chunk_id": chunk_id,
    }

    response = requests.put(url, auth=auth, json=body, timeout=30)

    response.raise_for_status()


def search_similar(embedding, k=5):
    url = f"{ENDPOINT}/{INDEX}/_search"

    body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": embedding,
                    "k": k,
                }
            }
        },
    }

    response = requests.post(
        url,
        auth=auth,
        json=body,
        timeout=30,
    )

    response.raise_for_status()

    hits = response.json()["hits"]["hits"]

    return [h["_source"]["text"] for h in hits]
