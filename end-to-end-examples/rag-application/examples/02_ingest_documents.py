"""Step 2: Document Ingestion Pipeline for RAG.

Parses PDF/document files with Docling, chunks the text with
configurable size and overlap, generates embeddings via the deployed
embedding model, and stores the resulting vectors in a vector database
(Milvus or pgvector).

Usage:
    cp .env.example .env   # fill in real values
    python 02_ingest_documents.py --document-dir ./documents
    python 02_ingest_documents.py --document-dir ./documents --vector-db pgvector
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".md", ".txt"}


def parse_documents(document_dir: str) -> list[dict]:
    """Parse documents in *document_dir* using Docling.

    Returns a list of dicts with ``source``, ``page``, and ``text`` keys.
    """
    from docling.document_converter import DocumentConverter

    doc_path = Path(document_dir)
    files = [
        f for f in sorted(doc_path.iterdir())
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        print(f"ERROR: No supported documents found in {document_dir}")
        print(f"  Supported extensions: {SUPPORTED_EXTENSIONS}")
        sys.exit(1)

    print(f"Found {len(files)} document(s) to parse")

    converter = DocumentConverter()
    parsed: list[dict] = []

    for filepath in files:
        print(f"  Parsing: {filepath.name}")
        try:
            result = converter.convert(str(filepath))
            text = result.document.export_to_markdown()
            parsed.append({
                "source": filepath.name,
                "page": 0,
                "text": text,
            })
        except Exception as exc:
            print(f"  WARNING: Failed to parse {filepath.name}: {exc}")

    print(f"Parsed {len(parsed)} document(s) successfully")
    return parsed


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    """Split parsed documents into overlapping text chunks.

    Each chunk retains the ``source`` metadata from its parent document.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for i, text in enumerate(splits):
            chunks.append({
                "source": doc["source"],
                "chunk_index": i,
                "text": text,
            })

    print(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def generate_embeddings(
    chunks: list[dict],
    embedding_endpoint: str,
    api_key: str,
    batch_size: int = 32,
) -> list[dict]:
    """Generate embeddings for each chunk via the deployed embedding model.

    Calls the OpenAI-compatible ``/v1/embeddings`` endpoint served by
    the KServe InferenceService.
    """
    url = f"{embedding_endpoint.rstrip('/')}/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"Generating embeddings ({len(chunks)} chunks, batch_size={batch_size}) ...")

    start = time.time()
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [c["text"] for c in batch]

        payload = {"input": texts, "model": "nomic-embed-text"}

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"ERROR: Embedding request failed: {exc}")
            sys.exit(1)

        data = resp.json()["data"]
        for item, chunk in zip(data, batch):
            chunk["embedding"] = item["embedding"]

        done = min(batch_start + batch_size, len(chunks))
        print(f"  Embedded {done}/{len(chunks)} chunks")

    elapsed = time.time() - start
    dim = len(chunks[0]["embedding"]) if chunks else 0
    print(f"Embeddings complete: dim={dim}, {elapsed:.1f}s")
    return chunks


def store_in_milvus(
    chunks: list[dict],
    collection_name: str,
    milvus_uri: str,
    milvus_token: str,
) -> None:
    """Insert embedded chunks into a Milvus collection."""
    from pymilvus import MilvusClient, DataType

    client = MilvusClient(uri=milvus_uri, token=milvus_token or "")
    dim = len(chunks[0]["embedding"])

    if client.has_collection(collection_name):
        print(f"  Dropping existing collection: {collection_name}")
        client.drop_collection(collection_name)

    from pymilvus import CollectionSchema, FieldSchema

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields=fields, description="RAG document chunks")
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )

    records = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            "embedding": c["embedding"],
        }
        for c in chunks
    ]

    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        client.insert(collection_name=collection_name, data=batch)

    print(f"  Inserted {len(records)} vectors into Milvus collection '{collection_name}'")

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        params={"nlist": 128},
    )
    client.create_index(collection_name=collection_name, index_params=index_params)
    print(f"  Created IVF_FLAT index on '{collection_name}'")


def store_in_pgvector(
    chunks: list[dict],
    collection_name: str,
    pg_connection: str,
) -> None:
    """Insert embedded chunks into a pgvector table."""
    import psycopg2
    from pgvector.psycopg2 import register_vector

    conn = psycopg2.connect(pg_connection)
    register_vector(conn)
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    dim = len(chunks[0]["embedding"])
    cur.execute(f"DROP TABLE IF EXISTS {collection_name};")
    cur.execute(f"""
        CREATE TABLE {collection_name} (
            id SERIAL PRIMARY KEY,
            source VARCHAR(512),
            chunk_index INTEGER,
            text TEXT,
            embedding vector({dim})
        );
    """)

    for c in chunks:
        cur.execute(
            f"INSERT INTO {collection_name} (source, chunk_index, text, embedding) "
            f"VALUES (%s, %s, %s, %s)",
            (c["source"], c["chunk_index"], c["text"], c["embedding"]),
        )

    cur.execute(f"""
        CREATE INDEX ON {collection_name}
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    """)

    conn.commit()
    cur.close()
    conn.close()
    print(f"  Inserted {len(chunks)} vectors into pgvector table '{collection_name}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents into a vector store for RAG."
    )
    parser.add_argument(
        "--document-dir",
        default=None,
        help="Directory with source documents (default: $DOCUMENT_DIR)",
    )
    parser.add_argument(
        "--embedding-endpoint",
        default=None,
        help="Embedding model endpoint URL (default: $EMBEDDING_ENDPOINT)",
    )
    parser.add_argument(
        "--embedding-api-key",
        default=None,
        help="API key for embedding endpoint (default: $EMBEDDING_API_KEY)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Maximum chunk size in characters (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        help="Overlap between chunks in characters (default: 64)",
    )
    parser.add_argument(
        "--collection-name",
        default="rag_documents",
        help="Vector store collection/table name (default: rag_documents)",
    )
    parser.add_argument(
        "--vector-db",
        choices=["milvus", "pgvector"],
        default="milvus",
        help="Vector database backend (default: milvus)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    document_dir = args.document_dir or os.getenv("DOCUMENT_DIR", "./documents")
    embedding_endpoint = args.embedding_endpoint or os.getenv("EMBEDDING_ENDPOINT")
    embedding_api_key = args.embedding_api_key or os.getenv("EMBEDDING_API_KEY", "")
    milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    milvus_token = os.getenv("MILVUS_TOKEN", "")
    pg_connection = os.getenv(
        "PG_CONNECTION", "postgresql://user:password@localhost:5432/vectordb"
    )

    if not embedding_endpoint:
        print("ERROR: Embedding endpoint is required.")
        print("  Set --embedding-endpoint or $EMBEDDING_ENDPOINT.")
        print("  Deploy the embedding model first with 01_deploy_model.py.")
        sys.exit(1)

    print(f"Document Ingestion Pipeline")
    print(f"  Document dir    : {document_dir}")
    print(f"  Embedding       : {embedding_endpoint}")
    print(f"  Chunk size      : {args.chunk_size}")
    print(f"  Chunk overlap   : {args.chunk_overlap}")
    print(f"  Vector DB       : {args.vector_db}")
    print(f"  Collection      : {args.collection_name}")
    print()

    documents = parse_documents(document_dir)

    chunks = chunk_documents(documents, args.chunk_size, args.chunk_overlap)

    chunks = generate_embeddings(
        chunks, embedding_endpoint, embedding_api_key, args.batch_size
    )

    print(f"\n{'=' * 60}")
    print(f"Storing vectors in {args.vector_db} ...")
    print(f"{'=' * 60}")

    if args.vector_db == "milvus":
        store_in_milvus(chunks, args.collection_name, milvus_uri, milvus_token)
    else:
        store_in_pgvector(chunks, args.collection_name, pg_connection)

    print(f"\n{'=' * 60}")
    print("Ingestion complete.")
    print(f"  Documents parsed : {len(documents)}")
    print(f"  Chunks created   : {len(chunks)}")
    print(f"  Vectors stored   : {len(chunks)}")
    print(f"  Collection       : {args.collection_name}")
    print(f"{'=' * 60}")
    print("\nProceed to 03_query_rag.py to test retrieval-augmented generation.")


if __name__ == "__main__":
    main()
