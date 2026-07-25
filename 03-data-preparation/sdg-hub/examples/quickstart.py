"""SDG Hub quickstart -- the simplest possible data generation pipeline.

Core concept:  seed dataset -> Flow -> enriched dataset

Usage:
    export MODEL_API_KEY="your-key"
    python quickstart.py path/to/flow.yaml
"""

import os
import sys

import nest_asyncio
import pandas as pd
from sdg_hub import Flow

nest_asyncio.apply()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python quickstart.py <path/to/flow.yaml>")
        print("  Set MODEL_API_KEY env var or OPENAI_API_KEY before running.")
        sys.exit(1)

    flow_path = sys.argv[1]
    api_key = os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("TEACHER_MODEL", "openai/gpt-4o")

    if not api_key:
        print("ERROR: Set MODEL_API_KEY or OPENAI_API_KEY environment variable.")
        sys.exit(1)

    dataset = pd.DataFrame({
        "document": ["Your document text here..."],
        "domain": ["your-domain"],
    })

    flow = Flow.from_yaml(flow_path)
    flow.set_model_config(model=model, api_key=api_key)

    result = flow.generate(dataset)
    print(f"Generated {len(result)} rows")

    result.to_json("output.jsonl", orient="records", lines=True)
    print("Output saved to output.jsonl")


if __name__ == "__main__":
    main()
