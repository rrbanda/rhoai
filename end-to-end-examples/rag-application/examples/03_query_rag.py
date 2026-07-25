"""Step 3: RAG Query Demonstration.

Retrieves relevant document chunks from the vector store, constructs
an augmented prompt with the retrieved context, and queries the deployed
LLM.  Demonstrates streaming responses and compares RAG-augmented
answers against vanilla (non-RAG) answers.

Usage:
    cp .env.example .env   # fill in real values
    python 03_query_rag.py --question "What are the key findings in the report?"
    python 03_query_rag.py --question "Summarize the main topics" --top-k 5 --no-stream
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

DEFAULT_QUESTIONS = [
    "What are the key findings discussed in the documents?",
    "Summarize the main topics covered in the knowledge base.",
    "What recommendations are made based on the analysis?",
]

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question accurately "
    "using the provided context. If the context does not contain enough "
    "information, say so clearly rather than guessing."
)

RAG_PROMPT_TEMPLATE = """{system}

### Retrieved Context
{context}

### User Question
{question}"""

VANILLA_PROMPT_TEMPLATE = """{system}

### User Question
{question}"""


def retrieve_from_milvus(
    query_embedding: list[float],
    collection_name: str,
    milvus_uri: str,
    milvus_token: str,
    top_k: int,
) -> list[dict]:
    """Search Milvus for the most similar chunks."""
    from pymilvus import MilvusClient

    client = MilvusClient(uri=milvus_uri, token=milvus_token or "")

    results = client.search(
        collection_name=collection_name,
        data=[query_embedding],
        limit=top_k,
        output_fields=["source", "chunk_index", "text"],
        search_params={"metric_type": "COSINE"},
    )

    hits = []
    for hit in results[0]:
        hits.append({
            "source": hit["entity"]["source"],
            "chunk_index": hit["entity"]["chunk_index"],
            "text": hit["entity"]["text"],
            "score": hit["distance"],
        })
    return hits


def retrieve_from_pgvector(
    query_embedding: list[float],
    collection_name: str,
    pg_connection: str,
    top_k: int,
) -> list[dict]:
    """Search pgvector for the most similar chunks."""
    import psycopg2
    from pgvector.psycopg2 import register_vector

    conn = psycopg2.connect(pg_connection)
    register_vector(conn)
    cur = conn.cursor()

    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    cur.execute(
        f"SELECT source, chunk_index, text, "
        f"1 - (embedding <=> %s::vector) AS score "
        f"FROM {collection_name} "
        f"ORDER BY embedding <=> %s::vector "
        f"LIMIT %s",
        (embedding_str, embedding_str, top_k),
    )

    hits = []
    for row in cur.fetchall():
        hits.append({
            "source": row[0],
            "chunk_index": row[1],
            "text": row[2],
            "score": float(row[3]),
        })

    cur.close()
    conn.close()
    return hits


def embed_query(
    text: str,
    embedding_endpoint: str,
    api_key: str,
) -> list[float]:
    """Generate an embedding for a single query string."""
    url = f"{embedding_endpoint.rstrip('/')}/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(
        url,
        json={"input": [text], "model": "nomic-embed-text"},
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def query_llm(
    prompt: str,
    llm_endpoint: str,
    api_key: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    stream: bool = False,
) -> str:
    """Send a chat completion request to the deployed LLM."""
    url = f"{llm_endpoint.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": "granite-3.3-8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    if not stream:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    resp = requests.post(url, json=payload, headers=headers, timeout=120, stream=True)
    resp.raise_for_status()

    full_response = []
    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        data_str = decoded[len("data: "):]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                print(content, end="", flush=True)
                full_response.append(content)
        except (json.JSONDecodeError, KeyError):
            continue

    print()
    return "".join(full_response)


def format_context(hits: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, hit in enumerate(hits, 1):
        score = f"{hit['score']:.4f}" if isinstance(hit["score"], float) else hit["score"]
        parts.append(
            f"[{i}] (source: {hit['source']}, score: {score})\n{hit['text']}"
        )
    return "\n\n".join(parts)


def run_comparison(
    question: str,
    hits: list[dict],
    llm_endpoint: str,
    api_key: str,
    stream: bool,
    max_tokens: int,
) -> None:
    """Run both RAG and vanilla queries and display side by side."""
    context = format_context(hits)

    rag_prompt = RAG_PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT, context=context, question=question
    )
    vanilla_prompt = VANILLA_PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT, question=question
    )

    print(f"\n{'=' * 60}")
    print("RAG-Augmented Answer")
    print(f"{'=' * 60}")
    start = time.time()
    rag_answer = query_llm(
        rag_prompt, llm_endpoint, api_key, max_tokens, stream=stream
    )
    rag_time = time.time() - start
    if not stream:
        print(rag_answer)
    print(f"\n  (generated in {rag_time:.1f}s)")

    print(f"\n{'=' * 60}")
    print("Vanilla Answer (no retrieval)")
    print(f"{'=' * 60}")
    start = time.time()
    vanilla_answer = query_llm(
        vanilla_prompt, llm_endpoint, api_key, max_tokens, stream=stream
    )
    vanilla_time = time.time() - start
    if not stream:
        print(vanilla_answer)
    print(f"\n  (generated in {vanilla_time:.1f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query a RAG pipeline with retrieval-augmented generation."
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Question to ask (default: runs built-in demo questions)",
    )
    parser.add_argument(
        "--llm-endpoint",
        default=None,
        help="LLM inference endpoint URL (default: $LLM_ENDPOINT)",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="API key for the LLM endpoint (default: $LLM_API_KEY)",
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
        "--vector-db",
        choices=["milvus", "pgvector"],
        default="milvus",
        help="Vector database backend (default: milvus)",
    )
    parser.add_argument(
        "--collection-name",
        default="rag_documents",
        help="Vector store collection/table name (default: rag_documents)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve (default: 3)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming responses",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=True,
        help="Compare RAG vs non-RAG answers (default: True)",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Skip the RAG vs non-RAG comparison",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    llm_endpoint = args.llm_endpoint or os.getenv("LLM_ENDPOINT")
    llm_api_key = args.llm_api_key or os.getenv("LLM_API_KEY", "")
    embedding_endpoint = args.embedding_endpoint or os.getenv("EMBEDDING_ENDPOINT")
    embedding_api_key = args.embedding_api_key or os.getenv("EMBEDDING_API_KEY", "")
    milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    milvus_token = os.getenv("MILVUS_TOKEN", "")
    pg_connection = os.getenv(
        "PG_CONNECTION", "postgresql://user:password@localhost:5432/vectordb"
    )

    if not llm_endpoint:
        print("ERROR: LLM endpoint is required.")
        print("  Set --llm-endpoint or $LLM_ENDPOINT.")
        sys.exit(1)

    if not embedding_endpoint:
        print("ERROR: Embedding endpoint is required.")
        print("  Set --embedding-endpoint or $EMBEDDING_ENDPOINT.")
        sys.exit(1)

    stream = not args.no_stream
    compare = args.compare and not args.no_compare
    questions = [args.question] if args.question else DEFAULT_QUESTIONS

    print(f"RAG Query Configuration")
    print(f"  LLM endpoint    : {llm_endpoint}")
    print(f"  Embedding       : {embedding_endpoint}")
    print(f"  Vector DB       : {args.vector_db}")
    print(f"  Collection      : {args.collection_name}")
    print(f"  Top-K           : {args.top_k}")
    print(f"  Streaming       : {stream}")
    print(f"  Compare mode    : {compare}")
    print()

    for q_idx, question in enumerate(questions, 1):
        print(f"\n{'#' * 60}")
        print(f"Question {q_idx}/{len(questions)}: {question}")
        print(f"{'#' * 60}")

        print("\nGenerating query embedding ...")
        try:
            query_embedding = embed_query(question, embedding_endpoint, embedding_api_key)
        except requests.RequestException as exc:
            print(f"ERROR: Failed to embed query: {exc}")
            continue

        print(f"Retrieving top-{args.top_k} chunks ...")
        try:
            if args.vector_db == "milvus":
                hits = retrieve_from_milvus(
                    query_embedding, args.collection_name,
                    milvus_uri, milvus_token, args.top_k,
                )
            else:
                hits = retrieve_from_pgvector(
                    query_embedding, args.collection_name,
                    pg_connection, args.top_k,
                )
        except Exception as exc:
            print(f"ERROR: Retrieval failed: {exc}")
            continue

        if not hits:
            print("WARNING: No matching chunks found. Is the collection populated?")
            continue

        print(f"\nRetrieved {len(hits)} chunks:")
        for i, hit in enumerate(hits, 1):
            score = f"{hit['score']:.4f}" if isinstance(hit["score"], float) else hit["score"]
            preview = hit["text"][:80].replace("\n", " ")
            print(f"  [{i}] score={score}  {hit['source']}  \"{preview}...\"")

        if compare:
            run_comparison(
                question, hits, llm_endpoint, llm_api_key,
                stream, args.max_tokens,
            )
        else:
            context = format_context(hits)
            prompt = RAG_PROMPT_TEMPLATE.format(
                system=SYSTEM_PROMPT, context=context, question=question
            )

            print(f"\n{'=' * 60}")
            print("Answer")
            print(f"{'=' * 60}")
            try:
                answer = query_llm(
                    prompt, llm_endpoint, llm_api_key,
                    args.max_tokens, stream=stream,
                )
                if not stream:
                    print(answer)
            except requests.RequestException as exc:
                print(f"ERROR: LLM query failed: {exc}")

    print(f"\n{'=' * 60}")
    print(f"RAG demonstration complete. {len(questions)} question(s) processed.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
