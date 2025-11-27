# RAG-System
RAG, AI Agent system

An intelligent Retrieval-Augmented Generation system that combines document retrieval with LLM generation for accurate, context-aware answers.

## About

This project implements a RAG-based chatbot that can ingest documents, retrieve relevant information, and generate intelligent responses using local LLMs. It features smart routing to decide between document-based answers and general knowledge.

## Features

- **PDF Ingestion** - Upload and process PDF documents into searchable knowledge base
- **RAG Query** - Retrieve relevant documents using vector similarity search
- **Generate** - Direct LLM generation for general questions
- **Smart Agent** - Automatically routes queries to RAG or LLM based on relevance
- **Docker Ready** - Containerized with Docker Compose for easy deployment
- **n8n Integration** - Webhook support for workflow automation

## Tech Stack

- FastAPI for REST API
- Ollama for local LLM (LLaMA 3.1)
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Docker & Docker Compose
- n8n for workflow automation

## Quick Start

```bash
# With Docker
docker compose up --build

# Pull LLM model
docker exec -it ollama ollama pull llama3.1:latest

# Initialize database
docker exec -it rag_api python create_docs.py

# Test
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"text": "Give your question"}'
```

## API Endpoints

- `POST /ingest_pdf` - Upload PDF documents
- `POST /rag_query` - Query knowledge base
- `POST /generate` - Generate with LLM
- `POST /agent` - Smart routing (recommended)