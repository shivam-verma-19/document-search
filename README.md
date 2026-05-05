# 🚀 Hybrid RAG Platform

A **Retrieval-Augmented Generation (RAG) system** built with a hybrid search pipeline and deployed on AWS serverless infrastructure.

This platform allows users to upload documents, ask questions, and generate summaries with high accuracy and low latency.

## ✨ Features

- 📄 Multi-format document ingestion (PDF, DOCX, TXT)
- 🔍 Hybrid Retrieval (Semantic + BM25 Keyword Search)
- 🤖 Context-aware Q&A using LLM
- 📝 Document summarization
- ⚡ Semantic + embedding cache (reduces latency & cost)
- 📊 Real-time metrics tracking (latency, cache hits, fallback)
- 🔐 Secure authentication via AWS Cognito
- ☁️ Fully serverless architecture (Lambda + API Gateway)


## 🧠 Architecture
```
User
↓
API Gateway (HTTPS)
↓
Lambda (RAG Engine)
↓
Hybrid Retrieval
├── Pinecone (Semantic)
└── BM25 (Keyword)
↓
Reranker (Cross Encoder)
↓
LLM (OpenAI)
↓
Response
```

## Async Flow
```
S3 (Upload)
↓
SQS
↓
Lambda (Ingestion)
↓
Vector DB (Pinecone)
```

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI  
- **LLM:** OpenAI (GPT-4o-mini)  
- **Vector DB:** Pinecone  
- **Search:** Hybrid (Semantic + BM25)  
- **Infra:** AWS Lambda, API Gateway, S3, SQS, DynamoDB  
- **Auth:** AWS Cognito  
- **IaC:** Terraform  
- **Monitoring:** CloudWatch  


## ⚙️ Local Setup

### 1. Clone repo


git clone [repo](https://github.com/shivam-verma-19/document-search.git)

cd rag-platform

### 2. Environment Variables

```
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
AWS_REGION=us-east-1
```

### 3. Run with Docker

```
docker build -t rag .
docker run -p 8000:8000 rag
```