"""Generate synthetic skills-tuning data using SDG Hub.

This script loads a skill-specific flow (red_team/prompt_generation) and uses
it to produce training data for skills tuning. The flow generates diverse
prompt completions from a small seed dataset, which can then be used to
fine-tune a model's instruction-following capabilities.

Usage:
    python generate_skills_data.py \\
        --input-data seed_prompts.jsonl \\
        --output-dir ./output \\
        --model openai/gpt-4o \\
        --api-key $OPENAI_API_KEY
"""

import argparse
import os
import sys
from pathlib import Path

import nest_asyncio
import pandas as pd
from sdg_hub import Flow, FlowRegistry

nest_asyncio.apply()

FLOW_NAME = "Red Teaming Prompt Generation Flow"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate skills-tuning data with SDG Hub."
    )
    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to a JSONL seed file.",
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
    print(f"Loaded {len(dataset)} seed rows from {args.input_data}")

    FlowRegistry.discover_flows()
    flow_path = FlowRegistry.get_flow_path(FLOW_NAME)
    if flow_path is None:
        print(f"ERROR: Flow '{FLOW_NAME}' not found in registry.")
        print("Ensure sdg_hub is installed: pip install sdg-hub[examples]")
        sys.exit(1)
    flow = Flow.from_yaml(flow_path)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    flow.set_model_config(model=args.model, api_key=api_key)

    print(f"Running skills flow with model {args.model} ...")
    result = flow.generate(dataset)
    print(f"Generated {len(result)} rows")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "skills_data.jsonl"
    result.to_json(output_path, orient="records", lines=True)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
