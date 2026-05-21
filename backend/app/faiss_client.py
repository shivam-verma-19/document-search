import os
import pickle
import uuid

import faiss
import numpy as np

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "/tmp/faiss")

INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "index.faiss")

METADATA_FILE = os.path.join(FAISS_INDEX_DIR, "metadata.pkl")

DIMENSION = 1536

_index = None
_documents = {}


def _ensure_dir():
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)


def _load_index():
    global _index
    global _documents

    _ensure_dir()

    if os.path.exists(INDEX_FILE):
        _index = faiss.read_index(INDEX_FILE)

    else:
        _index = faiss.IndexFlatL2(DIMENSION)

    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "rb") as f:
            _documents = pickle.load(f)

    else:
        _documents = {}


def _save_index():

    _ensure_dir()

    faiss.write_index(_index, INDEX_FILE)

    with open(METADATA_FILE, "wb") as f:
        pickle.dump(_documents, f)


def get_index() -> faiss.IndexFlatL2:

    if _index is None:
        _load_index()

    assert _index is not None

    return _index


# =========================================================
# Compatibility Layer
# =========================================================


def index_document(
    doc_id,
    text,
    embedding=None,
    metadata=None,
):

    if embedding is None:
        raise ValueError("Embedding required")

    if metadata is None:
        metadata = {}

    index = get_index()

    vector = np.asarray(
        [embedding],
        dtype=np.float32,
    )

    index.add(vector)  # type: ignore

    position = int(index.ntotal) - 1

    _documents[position] = {
        "_id": str(doc_id),
        "text": text,
        "metadata": metadata,
        "embedding": embedding,
    }

    _save_index()

    return {
        "result": "created",
        "_id": str(doc_id),
    }


def search_documents(
    query_embedding,
    k=5,
):
    index = get_index()

    if int(index.ntotal) == 0:
        return []

    query_vector = np.asarray(
        [query_embedding],
        dtype=np.float32,
    )

    distances, indices = index.search(  # type: ignore
        query_vector,
        int(k),
    )

    docs = []

    for idx in indices[0]:
        idx = int(idx)

        if idx == -1:
            continue

        doc = _documents.get(idx)

        if not doc:
            continue

        docs.append(
            {
                "_id": doc["_id"],
                "_source": {
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                },
            }
        )

    return docs


def search_similar(
    query_embedding,
    k=5,
):
    docs = search_documents(query_embedding, k)

    return [d["_source"]["text"] for d in docs]


def get_document(doc_id):

    for doc in _documents.values():
        if doc["_id"] == str(doc_id):
            return {
                "_id": doc["_id"],
                "_source": {
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                },
            }

    return None


def delete_document(doc_id):

    to_delete = None

    for idx, doc in _documents.items():
        if doc["_id"] == str(doc_id):
            to_delete = idx
            break

    if to_delete is not None:
        del _documents[to_delete]
        _save_index()

    return {
        "result": "deleted",
        "_id": str(doc_id),
    }


def reset_collection():
    global _index
    global _documents

    _index = faiss.IndexFlatL2(DIMENSION)

    _documents = {}

    _save_index()


def generate_doc_id():
    return str(uuid.uuid4())
