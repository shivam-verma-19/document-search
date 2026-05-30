# 🚀 Hybrid RAG Platform

A production-ready **Retrieval-Augmented Generation (RAG)** platform built with a fully serverless AWS architecture using:

- FastAPI
- AWS Lambda
- S3 Vectors (vector store)
- Google Gemini (LLM + embeddings)
- Hybrid Search (vector + BM25)
- Cross-Encoder Reranking
- Semantic Caching

The platform allows users to upload documents, retrieve context-aware answers, summarize documents, and perform low-latency semantic search at scale.

---

## ✨ Features

### 📄 Intelligent Document Ingestion

Supports PDF, DOCX, and TXT files. Pipeline includes automatic parsing, recursive chunking, metadata enrichment, embedding generation via Gemini `text-embedding-004`, and vector indexing into AWS S3 Vectors.

### 🔍 Hybrid Search Pipeline

Combines semantic vector search (S3 Vectors) with BM25 keyword search (warm-cached in DynamoDB) using Reciprocal Rank Fusion (RRF), followed by cross-encoder reranking for improved accuracy, recall, and context quality.

### 🧠 Context-Aware Q&A

Uses Google Gemini 2.5 Flash with context injection, prompt orchestration, hallucination reduction via forbidden-query filtering, and LLM fallback handling when retrieved documents are insufficient.

### 📝 Document Summarization

Generate full summaries or per-chunk overviews by doc ID or chunk key, with graceful fallback when content is unavailable.

### ⚡ Optimized RAG Pipeline

| Optimization | Savings |
|---|---|
| Semantic cache (DynamoDB) | ~20–40% |
| Hybrid retrieval (RRF) | ~15–25% |
| BM25 warm cache (DynamoDB) | Avoids S3 paginate on every query |
| Gemini embeddings | Cost-effective vs. OpenAI |
| Retrieval-first answering | Fewer LLM tokens |

---

## ☁️ AWS Serverless Architecture

```text
User
↓
API Gateway
↓
Lambda (FastAPI RAG API)
↓
Hybrid Retrieval
├── S3 Vectors (Semantic Search)
└── BM25 (DynamoDB warm cache)
↓
Cross-Encoder Reranker
↓
Google Gemini 2.5 Flash
↓
Response (DynamoDB semantic cache)
```

Document ingestion runs asynchronously:

```text
User Upload
↓
S3 Bucket
↓
SQS Queue
↓
Worker Lambda (processor.py)
├── Text extraction (PDF/DOCX/TXT)
├── Chunking
├── Gemini embeddings
├── S3 Vectors index
└── BM25 cache update (DynamoDB)
```

---

## 🧪 Test Coverage

Current coverage: **≥ 90%** across all application modules.

```
pytest backend/tests/ --cov=backend --cov-report=term-missing
```

Key test modules:

| Module | What's tested |
|---|---|
| `test_bm25_cache.py` | Warm cache TTL, versioned writes, conflict retries |
| `test_cache_service.py` | Cache hit/miss/exception paths |
| `test_embeddings.py` | Gemini embedding happy/error paths |
| `test_faiss_client.py` | Index/search/delete/reset (legacy client) |
| `test_s3_vectors_client.py` | S3 Vectors CRUD, pagination, chunk deletion |
| `test_processor.py` | Idempotency, S3 download, PDF extraction |
| `test_rag.py` | Full RAG pipeline, forbidden query filter, cache |
| `test_search_service.py` | RRF fusion, hybrid search, reranker fallback |
| `test_auth.py` | Cognito JWT, lazy JWK client init |
| `test_retry.py` | Backoff, custom error lists, delay cap |
| `test_secrets.py` | Env var precedence, Secrets Manager load |
| `test_metric.py` | DynamoDB writes, pagination, CloudWatch push |

---

## 🛠️ Local Development

### Requirements

- Python 3.10+
- AWS credentials (or moto for tests)
- Google Gemini API key

### Install

```bash
pip install -r backend/requirements.txt
```

### Run tests

```bash
pytest backend/tests/ --cov=backend --cov-report=term-missing
```

### Environment variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key |
| `S3_VECTOR_BUCKET_NAME` | S3 Vectors bucket |
| `S3_VECTOR_INDEX_NAME` | S3 Vectors index name |
| `DYNAMODB_BM25_TABLE` | BM25 corpus cache table |
| `DYNAMODB_METRICS_TABLE` | Metrics table |
| `SECRET_NAME` | AWS Secrets Manager secret name |
| `COGNITO_USER_POOL_ID` | Cognito user pool (auth) |
| `AUTH_DISABLED` | Set `true` to skip auth in dev |
| `BUCKET_NAME` | S3 upload bucket |
| `QUEUE_URL` | SQS queue URL for async ingestion |

---

## 📁 Project Structure

```
backend/
├── app/
│   ├── auth.py              # Cognito JWT verification
│   ├── bm25_cache.py        # DynamoDB-backed BM25 warm cache
│   ├── cache_service.py     # Semantic answer cache
│   ├── config.py            # Pydantic settings
│   ├── document_repository.py  # Repository abstraction
│   ├── embeddings.py        # Gemini text-embedding-004
│   ├── faiss_client.py      # Legacy FAISS client (kept for compatibility)
│   ├── gemini_client.py     # Gemini LLM router
│   ├── hybrid.py            # BM25Retriever
│   ├── ingest.py            # Upload + SQS enqueue
│   ├── main.py              # FastAPI app + routes
│   ├── metrics.py           # DynamoDB + CloudWatch metrics
│   ├── monitoring.py        # CloudWatch push_metric
│   ├── processor.py         # SQS worker: extract + embed + index
│   ├── rag.py               # ask_question + summarize_doc pipeline
│   ├── reranker.py          # Cross-encoder reranker
│   ├── retry.py             # Exponential backoff
│   ├── s3_vectors_client.py # AWS S3 Vectors CRUD
│   ├── search_service.py    # hybrid_search + rerank_documents
│   ├── secrets.py           # AWS Secrets Manager
│   └── worker_lambda.py     # SQS Lambda handler
└── tests/
    ├── conftest.py          # Shared fixtures + moto mocks
    └── test_*.py            # Per-module test files
```
