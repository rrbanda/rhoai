"""Generate knowledge-tuning data for non-English languages (Japanese example).

Demonstrates the full multilingual knowledge data generation pipeline:
  1. Document preprocessing — convert PDFs via Docling, chunk into markdown,
     and create seed data with ICL examples from a qna.yaml config
  2. Flow discovery — find the language-specific knowledge generation flow
     in the SDG Hub FlowRegistry
  3. Data generation — run the flow against seed data with checkpointing
  4. Post-processing — deduplicate and convert into chat-message training
     format

This script uses Japanese as the reference language but the same approach
applies to any non-English language supported by SDG Hub flows.

Adapted from:
  sdg_hub/examples/knowledge_tuning/multilingual/japanese/
  document_pre_processing_ja.ipynb  &  knowledge_generation_ja.ipynb

Prerequisites:
  - Raw PDF documents in a local directory
  - A qna.yaml file with ICL examples (see SDG Hub docs for format)
  - pip install sdg-hub[examples] docling

Usage:
    # Step 1 + 2: preprocess documents then generate knowledge data
    python multilingual_knowledge_gen.py \\
        --docs-dir ./japanese_docs \\
        --qna-yaml ./japanese_docs/qna.yaml \\
        --model hosted_vllm/microsoft/phi-4 \\
        --api-base http://localhost:8000/v1 --api-key EMPTY

    # Preprocess only (creates seed_data.jsonl)
    python multilingual_knowledge_gen.py \\
        --docs-dir ./japanese_docs --preprocess-only

    # Generate only (from existing seed data)
    python multilingual_knowledge_gen.py \\
        --seed-data ./output/seed_data.jsonl \\
        --model hosted_vllm/microsoft/phi-4 \\
        --api-base http://localhost:8000/v1 --api-key EMPTY

    # With environment variables
    export MODEL=hosted_vllm/microsoft/phi-4
    export API_BASE=http://localhost:8000/v1
    export API_KEY=EMPTY
    python multilingual_knowledge_gen.py --docs-dir ./japanese_docs
"""

import argparse
import glob
import os
from pathlib import Path

import nest_asyncio
from datasets import Dataset, load_dataset
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

DEFAULT_FLOW_NAME = (
    "Advanced Japanese Document Grounded Question-Answer "
    "Generation Flow for Knowledge Tuning"
)


# ---------------------------------------------------------------------------
# Document preprocessing
# ---------------------------------------------------------------------------

def preprocess_documents(
    docs_dir: str,
    qna_yaml: str | None,
    output_dir: str,
) -> str:
    """Convert PDFs to markdown, chunk, and create seed data.

    Returns the path to the generated seed_data.jsonl.
    """
    from docling.document_converter import DocumentConverter

    docs_path = Path(docs_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Convert PDFs to markdown -------------------------------------------
    pdf_files = sorted(glob.glob(str(docs_path / "*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {docs_dir}")

    print(f"Converting {len(pdf_files)} PDF(s) via Docling ...")
    converter = DocumentConverter()
    md_files: list[str] = []
    for pdf in pdf_files:
        print(f"  {pdf}")
        result = converter.convert(pdf)
        md_text = result.document.export_to_markdown()
        md_path = docs_path / (Path(pdf).stem + ".md")
        md_path.write_text(md_text, encoding="utf-8")
        md_files.append(str(md_path))
    print(f"Produced {len(md_files)} markdown file(s)")

    # --- Chunk and create seed data -----------------------------------------
    if qna_yaml:
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from knowledge_utils import DocProcessor

            dp = DocProcessor(str(docs_path), user_config_path=qna_yaml)
            seed_data = dp.get_processed_markdown_dataset(md_files)
        except ImportError:
            print("knowledge_utils not available — falling back to simple chunking")
            seed_data = _simple_chunk_to_dataset(md_files)
    else:
        seed_data = _simple_chunk_to_dataset(md_files)

    seed_path = str(out_path / "seed_data.jsonl")
    seed_data.to_json(seed_path, orient="records", lines=True, force_ascii=False)
    print(f"Seed data written to {seed_path} ({len(seed_data)} rows)")
    return seed_path


def _simple_chunk_to_dataset(md_files: list[str], chunk_words: int = 5000) -> Dataset:
    """Chunk markdown files into a HuggingFace Dataset (fallback path)."""
    chunks: list[str] = []
    for md_file in md_files:
        text = Path(md_file).read_text(encoding="utf-8")
        words = text.split()
        for i in range(0, len(words), chunk_words):
            chunk = " ".join(words[i : i + chunk_words])
            if chunk.strip():
                chunks.append(chunk)
    return Dataset.from_dict({"document": chunks})


# ---------------------------------------------------------------------------
# Knowledge generation
# ---------------------------------------------------------------------------

def discover_flow(flow_name: str) -> str:
    """Find the flow path via FlowRegistry, raising on miss."""
    from sdg_hub import FlowRegistry

    FlowRegistry.discover_flows()
    flow_path = FlowRegistry.get_flow_path(flow_name)
    if flow_path is None:
        available = FlowRegistry.list_flows()
        raise SystemExit(
            f"Flow not found: {flow_name}\nAvailable flows: {available}"
        )
    return flow_path


def run_generation(
    seed_path: str,
    flow_name: str,
    model: str,
    api_base: str | None,
    api_key: str,
    output_dir: str,
    sample_size: int | None,
    max_concurrency: int,
    checkpoint_dir: str | None,
    save_freq: int,
    dry_run: bool,
) -> None:
    """Load the flow, configure the model, and generate data."""
    from sdg_hub import Flow

    flow_path = discover_flow(flow_name)
    flow = Flow.from_yaml(flow_path)
    print(f"Loaded flow: {flow_name}")

    # --- Model configuration ------------------------------------------------
    model_kwargs: dict = {"model": model, "api_key": api_key}
    if api_base:
        model_kwargs["api_base"] = api_base
    flow.set_model_config(**model_kwargs)

    # --- Load seed data -----------------------------------------------------
    ds = load_dataset("json", data_files=seed_path, split="train")
    ds = ds.add_column("seed_id", list(range(len(ds))))
    print(f"Loaded {len(ds)} seed rows from {seed_path}")

    # --- Dry run (optional) --------------------------------------------------
    if dry_run:
        print("Running dry run to validate pipeline ...")
        result = flow.dry_run(
            ds,
            sample_size=min(sample_size or 2, len(ds)),
            max_concurrency=max_concurrency,
            enable_time_estimation=True,
        )
        print(f"Dry run OK — output columns: {result['final_dataset']['columns']}")
        return

    # --- Generate ------------------------------------------------------------
    gen_kwargs: dict = {"max_concurrency": max_concurrency}
    if checkpoint_dir:
        gen_kwargs["checkpoint_dir"] = checkpoint_dir
        gen_kwargs["save_freq"] = save_freq

    if sample_size and sample_size < len(ds):
        ds = ds.select(range(sample_size))
        print(f"Using sample of {len(ds)} rows")

    print(f"Generating with model {model} ...")
    generated = flow.generate(ds, **gen_kwargs)
    print(f"Generated {len(generated)} rows")

    # --- Post-process into training format -----------------------------------
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    messages_data = _to_message_format(generated)
    messages_path = out_path / "messages_data.jsonl"
    messages_data.to_json(str(messages_path), force_ascii=False)
    print(f"Training data saved to {messages_path} ({len(messages_data)} rows)")

    raw_path = out_path / "generated_raw.jsonl"
    generated.to_json(str(raw_path), orient="records", lines=True, force_ascii=False)
    print(f"Raw output saved to {raw_path}")


def _to_message_format(generated_data: Dataset) -> Dataset:
    """Deduplicate and convert generated QA pairs into chat messages."""
    seen: set = set()
    messages_list: list[dict] = []
    for row in generated_data:
        user = row["question"]
        assistant = row["response"]
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        key = (user, assistant)
        if key not in seen:
            seen.add(key)
            messages_list.append({"messages": messages})
    return Dataset.from_list(messages_list)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multilingual knowledge data generation (Japanese example).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_argument_group("document source")
    src.add_argument(
        "--docs-dir", type=str, default=None,
        help="Directory of PDF documents to preprocess.",
    )
    src.add_argument(
        "--qna-yaml", type=str, default=None,
        help="Path to qna.yaml with ICL examples for seed data creation.",
    )
    src.add_argument(
        "--seed-data", type=str, default=None,
        help="Path to existing seed_data.jsonl (skips preprocessing).",
    )
    src.add_argument(
        "--preprocess-only", action="store_true",
        help="Only run preprocessing, do not generate data.",
    )

    mdl = p.add_argument_group("model configuration")
    mdl.add_argument(
        "--model", type=str,
        default=os.getenv("MODEL", "hosted_vllm/microsoft/phi-4"),
        help="LiteLLM model identifier.",
    )
    mdl.add_argument(
        "--api-base", type=str,
        default=os.getenv("API_BASE", None),
        help="API base URL (e.g. http://localhost:8000/v1).",
    )
    mdl.add_argument(
        "--api-key", type=str,
        default=os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")),
        help="API key (default: EMPTY for local vLLM).",
    )

    gen = p.add_argument_group("generation options")
    gen.add_argument(
        "--flow-name", type=str, default=DEFAULT_FLOW_NAME,
        help="Name of the SDG Hub flow to use.",
    )
    gen.add_argument(
        "--output-dir", type=str, default="./output",
        help="Output directory (default: ./output).",
    )
    gen.add_argument(
        "--sample-size", type=int, default=None,
        help="Limit seed data to N rows (useful for testing).",
    )
    gen.add_argument(
        "--max-concurrency", type=int, default=20,
        help="Max concurrent LLM requests in async mode (default: 20).",
    )
    gen.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Directory for generation checkpoints (enables resumable runs).",
    )
    gen.add_argument(
        "--save-freq", type=int, default=10,
        help="Checkpoint save frequency in requests (default: 10).",
    )
    gen.add_argument(
        "--dry-run", action="store_true",
        help="Validate the pipeline without running full generation.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_path = args.seed_data

    # --- Preprocessing -------------------------------------------------------
    if args.docs_dir and not seed_path:
        qna = args.qna_yaml
        if not qna:
            candidate = Path(args.docs_dir) / "qna.yaml"
            if candidate.exists():
                qna = str(candidate)
                print(f"Auto-detected qna.yaml: {qna}")

        seed_path = preprocess_documents(args.docs_dir, qna, args.output_dir)

    if args.preprocess_only:
        print("Preprocessing complete — exiting (--preprocess-only).")
        return

    if not seed_path:
        raise SystemExit(
            "Provide --docs-dir (to preprocess) or --seed-data (existing JSONL)."
        )

    # --- Generation ----------------------------------------------------------
    run_generation(
        seed_path=seed_path,
        flow_name=args.flow_name,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        max_concurrency=args.max_concurrency,
        checkpoint_dir=args.checkpoint_dir,
        save_freq=args.save_freq,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
