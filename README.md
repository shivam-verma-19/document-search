# Hybrid RAG Platform

## Features
- File upload (PDF, DOCX, TXT)
- RAG-based Q&A
- Document summarization
- Semantic caching
- Metrics tracking
- Serverless deployment

## Tech Stack
Python, FastAPI, AWS, Terraform, Pinecone, OpenAI

## Run
docker build -t rag .
docker run -p 8000:8000 rag