"""Generate a RAG evaluation dataset from source documents using SDG Hub.

This script takes text or PDF documents as input, runs the RAG Evaluation
Dataset Flow, and produces a JSONL file of question-answer pairs with ground
truth context suitable for evaluation frameworks like RAGAS.

Pipeline stages:
  1. Prepare input documents with outlines (chunking for long documents)
  2. Load the RAG Evaluation Dataset Flow from SDG Hub
  3. Run the flow (topic extraction → question generation → answer generation
     → groundedness scoring → ground truth extraction)
  4. Post-process output into RAGAS-compatible format
  5. Save results as JSONL

Usage:
    export INFERENCE_MODEL="openai/gpt-4o"
    export URL="https://your-endpoint/v1"
    export API_KEY="your-api-key"

    python generate_rag_eval_dataset.py \\
        --input-file /path/to/document.txt \\
        --document-title "My Document" \\
        --output rag_eval_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from sdg_hub import Flow, FlowRegistry


def prepare_dataset_from_text(
    text: str,
    document_outline: str,
    chunk_size: int = 3000,
    overlap: int = 500,
) -> Dataset:
    """Chunk a text document into a dataset for the RAG evaluation flow.

    Args:
        text: Full document text.
        document_outline: Title or summary of the document.
        chunk_size: Maximum characters per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        A Hugging Face Dataset with ``document`` and ``document_outline`` columns.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    step_size = chunk_size - overlap
    chunks = []
    for i in range(0, len(text), step_size):
        chunk = text[i : i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)

    return Dataset.from_dict(
        {
            "document": chunks,
            "document_outline": [document_outline] * len(chunks),
        }
    )


def prepare_dataset_from_pdf(
    pdf_path: str,
    document_outline: str,
    max_pages: int | None = None,
    chunk_size: int = 3000,
    chunk_overlap: int = 500,
) -> Dataset:
    """Extract text from a PDF and chunk it for the RAG evaluation flow."""
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("ERROR: PyPDF2 is required for PDF input. Install with: pip install PyPDF2", file=sys.stderr)
        sys.exit(1)

    reader = PdfReader(pdf_path)
    pages = reader.pages[:max_pages] if max_pages else reader.pages
    text = "\n".join(page.extract_text() for page in pages)
    return prepare_dataset_from_text(text, document_outline, chunk_size, chunk_overlap)


def configure_model(flow: Flow) -> Flow:
    """Apply model configuration from environment variables."""
    model = os.getenv("INFERENCE_MODEL", "")
    api_base = os.getenv("URL", "")
    api_key = os.getenv("API_KEY", "")

    if not model:
        print("ERROR: INFERENCE_MODEL environment variable must be set.", file=sys.stderr)
        sys.exit(1)

    if model and not model.startswith("openai/") and not model.startswith("ollama/"):
        model = "openai/" + model

    print(f"Model: {model}")
    flow.set_model_config(
        model=model,
        api_base=api_base or None,
        api_key=api_key or None,
    )
    return flow


def prepare_for_ragas(df: pd.DataFrame) -> list[dict]:
    """Convert generated output to RAGAS-compatible evaluation format.

    RAGAS expects: question, answer, contexts (list), ground_truth.
    """
    records = []
    for _, row in df.iterrows():
        question = str(row.get("question", ""))
        answer = str(row.get("response", ""))
        context = str(row.get("document", row.get("context", "")))
        ground_truth = str(row.get("ground_truth_context", answer))

        records.append(
            {
                "question": question,
                "answer": answer,
                "contexts": [context] if context else [""],
                "ground_truth": ground_truth,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a RAG evaluation dataset from source documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to input document (.txt or .pdf).",
    )
    parser.add_argument(
        "--document-title",
        required=True,
        help="Title or summary of the document (used as document_outline).",
    )
    parser.add_argument(
        "--output",
        default="rag_eval_dataset.jsonl",
        help="Output JSONL path (default: rag_eval_dataset.jsonl).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3000,
        help="Max characters per document chunk (default: 3000).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=500,
        help="Character overlap between chunks (default: 500).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Max PDF pages to process (default: all).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Max parallel LLM calls (default: from MAX_CONCURRENCY env var or 10).",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file for configuration (default: .env).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Step 1: Prepare input dataset
    print(f"\n--- Step 1: Preparing input dataset from {input_path.name} ---")
    if input_path.suffix.lower() == ".pdf":
        dataset = prepare_dataset_from_pdf(
            str(input_path),
            args.document_title,
            max_pages=args.max_pages,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    else:
        text = input_path.read_text(encoding="utf-8")
        dataset = prepare_dataset_from_text(
            text,
            args.document_title,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
        )
    print(f"Created {len(dataset)} document chunks")

    # Step 2: Load the RAG Evaluation flow
    print("\n--- Step 2: Loading RAG Evaluation Dataset Flow ---")
    flow_name = "RAG Evaluation Dataset Flow"
    flow_path = FlowRegistry.get_flow_path(flow_name)
    if flow_path is None:
        print(f"ERROR: Flow '{flow_name}' not found in registry.", file=sys.stderr)
        print("Ensure sdg_hub is installed: pip install sdg-hub[examples]", file=sys.stderr)
        sys.exit(1)
    flow = Flow.from_yaml(flow_path)
    print(f"Flow loaded: {flow_name}")

    # Step 3: Configure the model
    print("\n--- Step 3: Configuring model ---")
    flow = configure_model(flow)

    # Step 4: Generate
    print("\n--- Step 4: Generating RAG evaluation dataset ---")
    max_concurrency = args.max_concurrency or int(os.getenv("MAX_CONCURRENCY", "10"))
    print(f"Max concurrency: {max_concurrency}")
    print("This may take several minutes depending on dataset size and model speed...")

    generated_data = flow.generate(
        dataset,
        runtime_params={},
        max_concurrency=max_concurrency,
    )

    df = generated_data.to_pandas() if hasattr(generated_data, "to_pandas") else pd.DataFrame(generated_data)
    print(f"Generated {len(df)} records")
    print(f"Columns: {list(df.columns)}")

    # Step 5: Post-process and save
    print(f"\n--- Step 5: Post-processing for RAGAS format ---")
    ragas_data = prepare_for_ragas(df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in ragas_data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(ragas_data)} records to {output_path}")

    # Print sample
    if ragas_data:
        print("\n--- Sample Output ---")
        sample = ragas_data[0]
        for key in ["question", "answer", "ground_truth"]:
            val = sample.get(key, "")
            display = val[:200] + "..." if len(val) > 200 else val
            print(f"  {key}: {display}")

    print("\nDone. Use the output file with RAGAS or other evaluation frameworks.")


if __name__ == "__main__":
    main()
