"""Extract structured insights (summary, keywords, entities, sentiment) from text.

Uses SDG Hub's built-in structured text insights flow to perform four
analyses on every row of an input dataset:
  1. Summary   -- concise 2-3 sentence summary
  2. Keywords  -- top important terms
  3. Entities  -- named entities (people, organizations, locations)
  4. Sentiment -- emotional tone (positive / negative / neutral)

Results are combined into a single structured JSON column.

The script optionally loads a HuggingFace dataset (default: Bloomberg
Financial News) or accepts a local JSONL file with a ``text`` column.

Adapted from: sdg_hub/examples/text_analysis/structured_insights_demo.ipynb

Usage:
    # Using a HuggingFace dataset (default: Bloomberg Financial News)
    python text_analysis_insights.py \\
        --model openai/gpt-4o --api-key $OPENAI_API_KEY

    # Using a local JSONL file with a 'text' column
    python text_analysis_insights.py \\
        --input-data articles.jsonl --text-column text \\
        --model openai/gpt-4o

    # Using a local vLLM server
    python text_analysis_insights.py \\
        --model hosted_vllm/meta-llama/Llama-3.1-8B-Instruct \\
        --api-base http://localhost:8000/v1 --api-key EMPTY
"""

import argparse
import json
import os
from pathlib import Path

import nest_asyncio
import pandas as pd
from sdg_hub import Flow, FlowRegistry

nest_asyncio.apply()

FLOW_TAG = "text-analysis"
DEFAULT_HF_DATASET = "danidanou/Bloomberg_Financial_News"
DEFAULT_HF_SPLIT = "train"
DEFAULT_SAMPLE_SIZE = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured text insights with SDG Hub.",
    )

    src = parser.add_argument_group("data source (choose one)")
    src.add_argument(
        "--input-data",
        type=str,
        default=None,
        help="Path to a local JSONL file with a text column.",
    )
    src.add_argument(
        "--hf-dataset",
        type=str,
        default=DEFAULT_HF_DATASET,
        help=f"HuggingFace dataset name (default: {DEFAULT_HF_DATASET}).",
    )
    src.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column containing text to analyse (default: text).",
    )
    src.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of rows to sample when using a HF dataset (default: {DEFAULT_SAMPLE_SIZE}).",
    )

    model = parser.add_argument_group("model configuration")
    model.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="LiteLLM model identifier (default: openai/gpt-4o).",
    )
    model.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key. Falls back to the provider's standard env var if omitted.",
    )
    model.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Custom API base URL (e.g. for a local vLLM server).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory for generated output (default: ./output).",
    )

    return parser.parse_args()


def load_dataset_from_hf(name: str, split: str, sample_size: int, text_col: str) -> pd.DataFrame:
    """Load a HuggingFace dataset and prepare it for the flow."""
    from datasets import load_dataset

    print(f"Loading HuggingFace dataset: {name} (split={split})")
    ds = load_dataset(name, split=split)
    ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))

    df = ds.to_pandas()
    if text_col != "text" and text_col in df.columns:
        df = df.rename(columns={text_col: "text"})
    elif "Article" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"Article": "text"})

    print(f"Prepared {len(df)} rows (avg text length: "
          f"{df['text'].str.len().mean():.0f} chars)")
    return df


def main() -> None:
    args = parse_args()

    # --- Load data -----------------------------------------------------------
    if args.input_data:
        df = pd.read_json(args.input_data, lines=True)
        if args.text_column != "text" and args.text_column in df.columns:
            df = df.rename(columns={args.text_column: "text"})
        print(f"Loaded {len(df)} rows from {args.input_data}")
    else:
        df = load_dataset_from_hf(
            args.hf_dataset, DEFAULT_HF_SPLIT, args.sample_size, args.text_column,
        )

    # --- Discover and load the flow ------------------------------------------
    FlowRegistry.discover_flows()
    text_flows = FlowRegistry.search_flows(tag=FLOW_TAG)
    if not text_flows:
        raise SystemExit(
            f"No flow found with tag '{FLOW_TAG}'. "
            "Ensure sdg_hub is installed with: pip install sdg_hub"
        )
    flow_id = text_flows[0]
    flow_path = FlowRegistry.get_flow_path(flow_id)
    flow = Flow.from_yaml(flow_path)
    print(f"Loaded flow: {flow_id}")

    # --- Configure model -----------------------------------------------------
    model_kwargs: dict = {"model": args.model}
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        model_kwargs["api_key"] = api_key
    if args.api_base:
        model_kwargs["api_base"] = args.api_base
    flow.set_model_config(**model_kwargs)

    # --- Generate insights ---------------------------------------------------
    print(f"Running structured insights extraction with {args.model} ...")
    results = flow.generate(df)
    print(f"Generated insights for {len(results)} rows")
    print(f"Result columns: {list(results.columns)}")

    # --- Save results --------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "text_analysis_insights.jsonl"
    results.to_json(output_path, orient="records", lines=True)
    print(f"Output saved to {output_path}")

    # --- Print a sample result -----------------------------------------------
    if "structured_insights" in results.columns and len(results) > 0:
        sample = results.iloc[0]
        try:
            insights = json.loads(sample["structured_insights"])
            print("\n--- Sample Insight ---")
            print(json.dumps(insights, indent=2, ensure_ascii=False))
        except (json.JSONDecodeError, KeyError):
            pass


if __name__ == "__main__":
    main()
