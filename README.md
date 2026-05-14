# 🚀 Hybrid RAG Platform

A production-ready **Retrieval-Augmented Generation (RAG)** platform built with a fully serverless AWS architecture using:

- FastAPI
- AWS Lambda
- OpenSearch Serverless
- Bedrock
- Hybrid Search
- Reranking
- Semantic Caching

The platform allows users to upload documents, retrieve context-aware answers, summarize documents, and perform low-latency semantic search at scale.

---

# ✨ Features

## 📄 Intelligent Document Ingestion

Supports:

- PDF
- DOCX
- TXT

Pipeline includes:

- Automatic parsing
- Recursive chunking
- Metadata enrichment
- Embedding generation
- Vector indexing

---

## 🔍 Hybrid Search Pipeline

Combines:

- Semantic Vector Search (OpenSearch Serverless)
- BM25 Keyword Search
- Cross-Encoder Reranking

This improves:

- Accuracy
- Recall
- Context quality

---

## 🧠 Context-Aware Q&A

Uses:
- Amazon Bedrock models

Features:

- Context injection
- Prompt orchestration
- Hallucination reduction
- LLM fallback handling

---

## 📝 Document Summarization

Generate:

- Full summaries
- Context summaries
- Semantic document overviews

---

## ⚡ Cost Optimized RAG Pipeline

Implemented optimizations:

- Chunk-level embeddings
- Semantic caching
- Query rewriting
- Hybrid retrieval
- Reduced token usage
- Smaller embedding models
- Retrieval-first answering

### Estimated Cost Reduction

| Optimization | Savings |
|---|---|
| GPT-4 → GPT-4o-mini | ~70% |
| Semantic cache | ~20–40% |
| Hybrid retrieval | ~15–25% |
| Smaller embeddings | ~50% |
| Bedrock embeddings | Additional AWS-native savings |

Overall pipeline reduces LLM costs by approximately **60%+**.

---

# ☁️ AWS Serverless Architecture

## Request Flow

```text
User
↓
API Gateway
↓
Lambda (FastAPI RAG API)
↓
Hybrid Retrieval
├── OpenSearch Serverless (Vector Search)
└── BM25 Retriever
↓
Reranker
↓
Bedrock
↓
Response