# RAG Application End-to-End

**Status:** GA

Build and deploy a retrieval-augmented generation (RAG) application on RHOAI 3.4. Covers the full stack from model deployment through document ingestion to interactive RAG queries: Model Deployment (KServe/vLLM) -> Document Ingestion (Docling + Embeddings) -> Vector Storage (Milvus/pgvector) -> RAG Queries.

## What's Covered

- Deploying an LLM (Granite) and embedding model (nomic-embed-text) via KServe with the vLLM runtime
- Parsing and chunking documents with Docling
- Generating embeddings and storing vectors in Milvus or pgvector
- Querying the RAG pipeline with retrieval-augmented prompts
- Streaming LLM responses
- Comparing RAG-augmented answers against vanilla (non-RAG) answers

## Prerequisites

- Python 3.10+
- Access to an RHOAI 3.4 cluster with KServe enabled
- `oc` CLI logged in to the cluster (`oc login ...`)
- GPU node(s) available for model serving
- A vector database deployed (Milvus or PostgreSQL with pgvector)
- Source documents (PDF, DOCX, HTML, Markdown, or plain text)

## Quick Start

### 1. Setup

```bash
cd examples/
cp .env.example .env
# Edit .env with your cluster endpoints and model URIs

pip install -r requirements.txt
```

### 2. Deploy Models

```bash
python 01_deploy_model.py --namespace my-rag-project
```

This creates two KServe InferenceService resources:

| Model | Runtime | Purpose |
|-------|---------|---------|
| `granite-llm` | vLLM | Chat completions for answering questions |
| `nomic-embed` | vLLM | Text embeddings for document retrieval |

The script waits for both models to reach a ready state before exiting.
Customize model names and storage URIs with `--llm-name`, `--llm-model-uri`,
`--embedding-name`, and `--embedding-model-uri`.

### 3. Ingest Documents

Place your source documents in the `documents/` directory, then run:

```bash
python 02_ingest_documents.py --document-dir ./documents
```

The ingestion pipeline:

1. **Parse** -- Docling extracts text from PDFs, DOCX, PPTX, HTML, Markdown, and plain text files
2. **Chunk** -- Text is split into overlapping chunks (default 512 chars, 64 overlap)
3. **Embed** -- Each chunk is embedded via the deployed nomic-embed-text model
4. **Store** -- Vectors are inserted into the configured vector database

Use pgvector instead of Milvus with `--vector-db pgvector`.

Tune chunking with `--chunk-size` and `--chunk-overlap`.

### 4. Query with RAG

```bash
# Interactive question
python 03_query_rag.py --question "What are the key findings in the report?"

# Multiple built-in demo questions
python 03_query_rag.py

# Adjust retrieval depth
python 03_query_rag.py --question "Summarize the main topics" --top-k 5

# Disable streaming
python 03_query_rag.py --question "What recommendations are made?" --no-stream
```

By default, the script runs in **compare mode** -- it shows both a
RAG-augmented answer (with retrieved context) and a vanilla answer
(without retrieval) side by side, so you can see the impact of
retrieval augmentation.  Disable comparison with `--no-compare`.

## Architecture

```text
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Documents  │────>│   Docling    │────>│  Text Chunking   │
│  (PDF, etc.) │     │   Parser     │     │  (512 chars)     │
└──────────────┘     └──────────────┘     └────────┬─────────┘
                                                   │
                                          ┌────────▼─────────┐
                                          │  nomic-embed-text │
                                          │  (KServe/vLLM)   │
                                          └────────┬─────────┘
                                                   │
                                          ┌────────▼─────────┐
                                          │  Vector Store    │
                                          │  (Milvus/pgvec)  │
                                          └────────┬─────────┘
                                                   │
┌──────────────┐     ┌──────────────┐     ┌────────▼─────────┐
│  User Query  │────>│  Embed Query │────>│   Retrieve Top-K │
└──────────────┘     └──────────────┘     └────────┬─────────┘
                                                   │
                                          ┌────────▼─────────┐
                                          │  Augmented Prompt │
                                          │  (context + query)│
                                          └────────┬─────────┘
                                                   │
                                          ┌────────▼─────────┐
                                          │  Granite LLM     │
                                          │  (KServe/vLLM)   │
                                          └────────┬─────────┘
                                                   │
                                          ┌────────▼─────────┐
                                          │  Answer          │
                                          └──────────────────┘
```

## File Reference

| File | Purpose |
|------|---------|
| `.env.example` | Template for environment variables |
| `requirements.txt` | Python dependencies |
| `01_deploy_model.py` | Deploy LLM + embedding model via KServe |
| `02_ingest_documents.py` | Parse, chunk, embed, and store documents |
| `03_query_rag.py` | RAG query with retrieval and comparison |

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NAMESPACE` | OpenShift project for model serving | -- |
| `LLM_ENDPOINT` | Granite LLM inference URL | -- |
| `LLM_API_KEY` | API key for the LLM endpoint | (empty) |
| `LLM_MODEL_URI` | S3 URI to Granite model weights | `s3://models/ibm-granite/granite-3.3-8b-instruct` |
| `EMBEDDING_ENDPOINT` | nomic-embed-text inference URL | -- |
| `EMBEDDING_API_KEY` | API key for embedding endpoint | (empty) |
| `EMBEDDING_MODEL_URI` | S3 URI to embedding model weights | `s3://models/nomic-ai/nomic-embed-text-v1.5` |
| `DOCUMENT_DIR` | Directory with source documents | `./documents` |
| `MILVUS_URI` | Milvus connection URI | `http://localhost:19530` |
| `MILVUS_TOKEN` | Milvus authentication token | (empty) |
| `PG_CONNECTION` | PostgreSQL connection string (pgvector) | `postgresql://user:password@localhost:5432/vectordb` |

### CLI Flags

Each script supports `--help` for full usage details.  Key flags:

- `01_deploy_model.py`: `--namespace`, `--llm-name`, `--llm-model-uri`, `--embedding-name`, `--embedding-model-uri`, `--timeout`
- `02_ingest_documents.py`: `--document-dir`, `--chunk-size`, `--chunk-overlap`, `--vector-db`, `--collection-name`, `--batch-size`
- `03_query_rag.py`: `--question`, `--top-k`, `--max-tokens`, `--no-stream`, `--no-compare`, `--vector-db`, `--collection-name`

## Official Documentation

- [Red Hat AI Examples](https://github.com/red-hat-data-services/red-hat-ai-examples)
