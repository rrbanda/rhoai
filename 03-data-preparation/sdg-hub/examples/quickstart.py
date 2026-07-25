"""SDG Hub quickstart -- the simplest possible data generation pipeline.

Core concept:  seed dataset -> Flow -> enriched dataset

Usage:
    export OPENAI_API_KEY="your-key"
    python quickstart.py path/to/flow.yaml
"""

import sys

import nest_asyncio
import pandas as pd
from sdg_hub import Flow

nest_asyncio.apply()


def main() -> None:
    flow_path = sys.argv[1] if len(sys.argv) > 1 else "path/to/flow.yaml"

    dataset = pd.DataFrame({
        "document": ["Your document text here..."],
        "domain": ["your-domain"],
    })

    flow = Flow.from_yaml(flow_path)
    flow.set_model_config(model="openai/gpt-4o", api_key="your-key")

    result = flow.generate(dataset)
    print(f"Generated {len(result)} rows")

    result.to_json("output.jsonl", orient="records", lines=True)
    print("Output saved to output.jsonl")


if __name__ == "__main__":
    main()
