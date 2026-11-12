# 🚀 Hybrid RAG Platform

A production-ready **Retrieval-Augmented Generation (RAG)** platform built with a fully serverless AWS architecture using:

- FastAPI
- AWS Lambda
- S3 Vectors (vector store)
- Google Gemini (LLM + embeddings)
- Hybrid Search (vector + BM25 + RRF fusion)
- Cross-Encoder Reranking
- Semantic Caching

The platform allows users to upload documents, retrieve context-aware answers, summarize documents, and perform low-latency semantic search at scale.

---

## ✨ Features

### 📄 Intelligent Document Ingestion

Supports PDF, DOCX, and TXT files. Pipeline includes automatic parsing, recursive chunking, metadata enrichment, embedding generation via Gemini `gemini-embedding-001`, and vector indexing into AWS S3 Vectors.

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

## 🌟 Advanced RAG Enhancements

The platform includes the following production-grade Hybrid RAG capabilities:

### 🔢 Adaptive Chunking
Splits documents on natural section headers (`#`, `##`, title caps) and semantic boundaries using sentence cosine similarity with fallback to recursive chunking. Chunks retain source metadata and active `section_header`.

### 🏷️ Metadata-Aware Filtering
Structured metadata (`file_type`, `upload_date`, `user_id`, `section_header`) is tagged on each chunk during ingestion. The search API exposes filter parameters (`file_type`, `date_from`, `date_to`, `section`) for scoped retrieval.

### 🔁 Multi-Step Query Decomposition
Complex, multi-hop, or comparative questions are automatically broken down into atomic sub-queries by `query_planner.py`. Documents are retrieved for each sub-query, merged via reciprocal rank fusion, and synthesized into a coherent answer.

### 📊 RAGAS-Based Evaluation
Full implementation of the 4 standard RAGAS metrics:
1. **Faithfulness**: Verifies claims against retrieved context.
2. **Answer Relevancy**: Assesses directness and conciseness to the question.
3. **Context Precision**: Evaluates signal-to-noise ratio and ranking quality of retrieved context.
4. **Context Recall**: Measures coverage of essential facts.
Scores are persisted to DynamoDB (`rag-eval`) and emitted to CloudWatch.

### 🔄 Conversation Memory
Short-term multi-turn conversation buffer backed by DynamoDB (`rag-sessions` table, keyed by `user_id + session_id`). Follow-up questions maintain dialogue context across turns without manual re-prompting.

### 🏗️ True Cross-Encoder Reranking
Configurable cross-encoder model (`BAAI/bge-reranker-base` or `cross-encoder/ms-marco-MiniLM-L-6-v2`) in `reranker.py` with GPU/CPU auto-detection and resilient lexical fallback.

### 🌐 HyDE (Hypothetical Document Embeddings)
HyDE query expansion generates a hypothetical answer via Gemini before embedding for dense vector search, coupled with BM25 keyword search and RRF fusion.

## Local development

The API allows browser requests from `http://localhost:3000` and
`http://127.0.0.1:3000` by default. For another frontend origin, set
`CORS_ALLOWED_ORIGINS` to a comma-separated list of exact origins, including
the scheme and port when applicable.

## Authentication

The website uses Cognito Managed Login with OAuth authorization code + PKCE.
Visitors can create their own accounts; Cognito verifies the required email
address for confirmation and password recovery, while sign-in uses the chosen
username. The frontend keeps access and refresh tokens in HttpOnly cookies and
proxies API requests server-side, so browser JavaScript does not handle tokens.
Access tokens last 60 minutes and refresh tokens last 30 days. Sign out clears
the app session and the Cognito hosted-login session.

For local development, copy `frontend/.env.example` to `frontend/.env.local` and
set the values from Terraform:

```text
AUTH_BASE_URL=http://localhost:3000
COGNITO_DOMAIN=<terraform output cognito_domain>
COGNITO_CLIENT_ID=<terraform output cognito_client_id>
RAG_API_URL=http://localhost:8000
```

`AUTH_BASE_URL` must exactly match the `frontend_origin` used by Terraform. For
production, set it to the HTTPS website origin, deploy/apply Terraform with
`-var="frontend_origin=https://your-site.example"`, and use the same origin in
the frontend environment. The Terraform Cognito client registers
`/api/auth/callback` and the site root as its callback and sign-out URLs.
API Gateway CORS is configured for this exact frontend origin.

The user pool accepts public self-registration and uses username as the
sign-in name. A Cognito pre-sign-up Lambda requires a valid email without
changing the existing pool's immutable schema; Cognito then sends its email
confirmation code. Existing accounts without an email will need an
administrator to add and verify one before they can use self-service password
recovery. Do not set `AUTH_DISABLED=true` in a deployed backend.
