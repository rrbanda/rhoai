"""Step 1: Synthetic Data Generation for Knowledge Tuning.

Generates Q&A training pairs from source documents using SDG Hub's
enhanced_multi_summary_qa knowledge flows. Four augmentation strategies
are run in sequence -- extractive summary, detailed summary, key facts,
and document-based QA -- each producing a separate JSONL output.

Usage:
    cp .env.example .env   # fill in real values
    python 01_data_generation.py [--document-dir ./documents] [--output-dir ./generated_output_data]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import nest_asyncio
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

from sdg_hub import Flow, FlowRegistry

nest_asyncio.apply()

FLOW_VARIANT_NAMES = {
    "extractive_summary": "Extractive Summary Knowledge Tuning Dataset Generation Flow",
    "detailed_summary": "Detailed Summary Knowledge Tuning Dataset Generation Flow",
    "key_facts": "Key Facts Knowledge Tuning Dataset Generation Flow",
    "doc_direct_qa": "Document Based Knowledge Tuning Dataset Generation Flow",
}


def resolve_flow_variants() -> dict[str, str]:
    """Resolve flow names to YAML paths via the FlowRegistry."""
    FlowRegistry.discover_flows()
    variants = {}
    for key, flow_name in FLOW_VARIANT_NAMES.items():
        path = FlowRegistry.get_flow_path(flow_name)
        if path is None:
            print(f"WARNING: Flow '{flow_name}' not found in registry, skipping.")
            continue
        variants[key] = path
    return variants


def load_documents(document_dir: str) -> pd.DataFrame:
    """Load pre-processed JSONL documents from *document_dir*.

    Expects files with at least ``document`` and ``domain`` columns.
    Additional columns used by the flows (``document_outline``,
    ``icl_document``, ``icl_query_1``, ``icl_query_2``, ``icl_query_3``)
    should also be present where required by the flow variant.
    """
    jsonl_files = sorted(Path(document_dir).glob("*.jsonl"))
    if not jsonl_files:
        print(f"ERROR: No .jsonl files found in {document_dir}")
        sys.exit(1)

    frames = [pd.read_json(f, lines=True) for f in jsonl_files]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} documents from {len(jsonl_files)} file(s)")
    return df


def run_flow(
    variant_name: str,
    flow_yaml_path: str,
    dataset: pd.DataFrame,
    teacher_model: str,
    api_key: str,
    api_base: str | None,
    output_dir: str,
    checkpoint_dir: str | None,
) -> pd.DataFrame | None:
    """Load a flow variant, configure the model, and generate data."""
    print(f"\n{'=' * 60}")
    print(f"Running flow: {variant_name}")
    print(f"  YAML : {flow_yaml_path}")
    print(f"  Model: {teacher_model}")
    print(f"  Rows : {len(dataset)}")
    print(f"{'=' * 60}")

    try:
        flow = Flow.from_yaml(flow_yaml_path)
    except Exception as exc:
        print(f"ERROR loading flow YAML: {exc}")
        return None

    flow.set_model_config(
        model=teacher_model,
        api_key=api_key,
        api_base=api_base,
    )

    start = time.time()
    try:
        result = flow.generate(
            dataset,
            checkpoint_dir=os.path.join(checkpoint_dir, variant_name)
            if checkpoint_dir
            else None,
        )
    except Exception as exc:
        print(f"ERROR during generation: {exc}")
        return None
    elapsed = time.time() - start

    if isinstance(result, pd.DataFrame):
        result_df = result
    else:
        result_df = result.to_pandas()

    output_path = os.path.join(output_dir, f"{variant_name}.jsonl")
    result_df.to_json(output_path, orient="records", lines=True)
    print(f"Saved {len(result_df)} rows to {output_path}  ({elapsed:.1f}s)")
    return result_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate knowledge-tuning data with SDG Hub."
    )
    parser.add_argument(
        "--document-dir",
        default=None,
        help="Directory with source .jsonl documents (default: $DOCUMENT_DIR)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for generated data (default: $OUTPUT_DATA_FOLDER)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint directory for resumable runs (default: $CHECKPOINT_DIR)",
    )
    parser.add_argument(
        "--flows",
        nargs="+",
        choices=list(FLOW_VARIANT_NAMES.keys()),
        default=list(FLOW_VARIANT_NAMES.keys()),
        help="Which flow variants to run (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    teacher_model = os.getenv("TEACHER_MODEL", "openai/gpt-oss-120b")
    api_key = os.getenv("MODEL_API_KEY", "")
    api_base = os.getenv("MODEL_API_BASE")
    document_dir = args.document_dir or os.getenv("DOCUMENT_DIR", "./documents")
    output_dir = args.output_dir or os.getenv(
        "OUTPUT_DATA_FOLDER", "./generated_output_data"
    )
    checkpoint_dir = args.checkpoint_dir or os.getenv("CHECKPOINT_DIR")

    if not api_key:
        print("ERROR: MODEL_API_KEY is not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    dataset = load_documents(document_dir)

    flow_variants = resolve_flow_variants()
    if not flow_variants:
        print("ERROR: No knowledge flows found in the SDG Hub registry.")
        print("Ensure sdg-hub is installed: pip install sdg-hub[examples]")
        sys.exit(1)

    print(f"\nTeacher model : {teacher_model}")
    print(f"API base      : {api_base or '(default)'}")
    print(f"Output dir    : {output_dir}")
    print(f"Flow variants : {args.flows}")

    results: dict[str, pd.DataFrame] = {}
    for variant in args.flows:
        if variant not in flow_variants:
            print(f"Skipping {variant}: flow not found in registry")
            continue
        flow_path = flow_variants[variant]
        result = run_flow(
            variant_name=variant,
            flow_yaml_path=flow_path,
            dataset=dataset,
            teacher_model=teacher_model,
            api_key=api_key,
            api_base=api_base,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )
        if result is not None:
            results[variant] = result

    print(f"\n{'=' * 60}")
    print("Data generation complete.")
    for name, df in results.items():
        print(f"  {name}: {len(df)} Q&A pairs")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
