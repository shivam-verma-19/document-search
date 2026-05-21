import os
import uuid
import chromadb
from chromadb.config import Settings

CHROMA_PATH = os.getenv(
    "CHROMA_PERSIST_DIR",
    "./chroma_db"
)

CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION",
    "documents"
)

_client = None
_collection = None


def get_client():
    global _client

    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(
                anonymized_telemetry=False
            )
        )

    return _client


def get_collection():
    global _collection

    if _collection is None:
        client = get_client()

        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION
        )

    return _collection


# =========================================================
# Compatibility Layer (replaces old OpenSearch API)
# =========================================================

def index_document(
    doc_id,
    text,
    embedding=None,
    metadata=None
):
    collection = get_collection()

    if metadata is None:
        metadata = {}

    collection.upsert(
        ids=[str(doc_id)],
        documents=[text],
        embeddings=[embedding] if embedding else None,
        metadatas=[metadata]
    )

    return {
        "result": "created",
        "_id": doc_id
    }


def search_documents(
    query_embedding,
    k=5
):
    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    docs = []

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    for i in range(len(ids)):
        docs.append({
            "_id": ids[i],
            "_source": {
                "text": documents[i],
                "metadata": metadatas[i]
            }
        })

    return docs


def delete_document(doc_id):
    collection = get_collection()

    collection.delete(
        ids=[str(doc_id)]
    )

    return {
        "result": "deleted",
        "_id": doc_id
    }


def get_document(doc_id):
    collection = get_collection()

    results = collection.get(
        ids=[str(doc_id)]
    )

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    ids = results.get("ids") or []
    
    if not ids:
        return None
    
    return {
        "_id": ids[0],
        "_source": {
            "text": documents[0] if documents else "",
            "metadata": metadatas[0] if metadatas else {}
        }
    }


def reset_collection():
    global _collection

    client = get_client()

    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass

    _collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION
    )


def generate_doc_id():
    return str(uuid.uuid4())

def search_similar(query_embedding, k=5):
    """
    Compatibility wrapper expected by rag.py and tests.
    Returns list[str]
    """

    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    documents = results.get("documents") or [[]]

    if not documents:
        return []

    return [
        doc
        for doc in documents[0]
        if doc
    ]