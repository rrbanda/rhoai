"""Generate synthetic knowledge-tuning data using SDG Hub.

Uses the built-in enhanced_multi_summary_qa flows to produce QA pairs from seed
documents. Four augmentation strategies are available:
  extractive_summary, detailed_summary, key_facts, document_based.

Usage:
    python generate_knowledge_data.py \\
        --flow-variant extractive_summary \\
        --input-data seed_data.jsonl \\
        --model openai/gpt-4o --api-key $OPENAI_API_KEY
"""

import argparse
import os
from pathlib import Path

import nest_asyncio
import pandas as pd
from sdg_hub import Flow

nest_asyncio.apply()

FLOW_VARIANTS = {
    "extractive_summary": (
        "src/sdg_hub/flows/knowledge_infusion/"
        "enhanced_multi_summary_qa/extractive_summary/flow.yaml"
    ),
    "detailed_summary": (
        "src/sdg_hub/flows/knowledge_infusion/"
        "enhanced_multi_summary_qa/detailed_summary/flow.yaml"
    ),
    "key_facts": (
        "src/sdg_hub/flows/knowledge_infusion/"
        "enhanced_multi_summary_qa/key_facts/flow.yaml"
    ),
    "document_based": (
        "src/sdg_hub/flows/knowledge_infusion/"
        "enhanced_multi_summary_qa/document_based/flow.yaml"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate knowledge-tuning data with SDG Hub."
    )
    parser.add_argument(
        "--flow-variant",
        choices=list(FLOW_VARIANTS),
        default="extractive_summary",
        help="Augmentation strategy (default: extractive_summary).",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to a JSONL seed file with 'document' and 'domain' columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory for generated output (default: ./output).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="LiteLLM model identifier (default: openai/gpt-4o).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key. Falls back to the provider's env var if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = pd.read_json(args.input_data, lines=True)
    print(f"Loaded {len(dataset)} seed documents from {args.input_data}")

    flow_path = FLOW_VARIANTS[args.flow_variant]
    flow = Flow.from_yaml(flow_path)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    flow.set_model_config(model=args.model, api_key=api_key)

    print(f"Running '{args.flow_variant}' flow with model {args.model} ...")
    result = flow.generate(dataset)
    print(f"Generated {len(result)} rows")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"knowledge_{args.flow_variant}.jsonl"
    result.to_json(output_path, orient="records", lines=True)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
