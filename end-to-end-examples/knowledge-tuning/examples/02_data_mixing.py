"""Step 2: Data Mixing -- Convert SDG output to Training Hub format.

Reads the JSONL files produced by 01_data_generation.py, converts each
Q&A pair into the ``messages`` chat format required by Training Hub,
marks every sample with ``"unmask": true`` for knowledge training, and
writes a combined training-mix file.

Usage:
    python 02_data_mixing.py [--input-dir ./generated_output_data] [--output-dir ./generated_output_data/training_mix]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the user's question "
    "accurately and thoroughly based on your training data."
)


def load_generated_data(input_dir: str) -> dict[str, pd.DataFrame]:
    """Load all JSONL files produced by the data generation step."""
    jsonl_files = sorted(Path(input_dir).glob("*.jsonl"))
    if not jsonl_files:
        print(f"ERROR: No .jsonl files found in {input_dir}")
        sys.exit(1)

    datasets: dict[str, pd.DataFrame] = {}
    for f in jsonl_files:
        name = f.stem
        df = pd.read_json(f, lines=True)
        datasets[name] = df
        print(f"  Loaded {name}: {len(df)} rows")
    return datasets


def row_to_messages(row: pd.Series) -> dict:
    """Convert a single Q&A row into the Training Hub messages format.

    Knowledge samples use ``"unmask": true`` so Training Hub unmasks all
    messages except the system prompt during loss computation.
    """
    question = str(row.get("question", ""))
    answer = str(row.get("response", ""))

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "unmask": True,
    }


def tokenize_and_stats(
    samples: list[dict], tokenizer_name: str
) -> list[dict]:
    """Tokenize each sample and attach a ``num_tokens`` field.

    Returns the augmented sample list and prints summary statistics.
    """
    print(f"\nTokenizing with {tokenizer_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    token_counts: list[int] = []
    for sample in samples:
        text = tokenizer.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=False
        )
        n_tokens = len(tokenizer.encode(text))
        sample["num_tokens"] = n_tokens
        token_counts.append(n_tokens)

    s = pd.Series(token_counts)
    print(f"  Samples       : {len(s)}")
    print(f"  Total tokens  : {s.sum():,}")
    print(f"  Mean tokens   : {s.mean():.0f}")
    print(f"  Median tokens : {s.median():.0f}")
    print(f"  Min tokens    : {s.min()}")
    print(f"  Max tokens    : {s.max()}")
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mix SDG output into Training Hub format."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory with generated JSONL files (default: $OUTPUT_DATA_FOLDER)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the training mix (default: $OUTPUT_DATA_FOLDER/training_mix)",
    )
    parser.add_argument(
        "--student-model",
        default=None,
        help="Student model for tokenization stats (default: $STUDENT_MODEL)",
    )
    parser.add_argument(
        "--skip-tokenize",
        action="store_true",
        help="Skip tokenization statistics (useful if model is not downloaded)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    input_dir = args.input_dir or os.getenv(
        "OUTPUT_DATA_FOLDER", "./generated_output_data"
    )
    default_output = os.path.join(
        os.getenv("OUTPUT_DATA_FOLDER", "./generated_output_data"), "training_mix"
    )
    output_dir = args.output_dir or default_output
    student_model = args.student_model or os.getenv(
        "STUDENT_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
    )

    os.makedirs(output_dir, exist_ok=True)

    print("Loading generated datasets ...")
    datasets = load_generated_data(input_dir)

    all_samples: list[dict] = []
    for name, df in datasets.items():
        if "question" not in df.columns or "response" not in df.columns:
            print(f"  Skipping {name}: missing 'question' or 'response' column")
            continue
        samples = [row_to_messages(row) for _, row in df.iterrows()]
        print(f"  {name}: {len(samples)} samples converted")
        all_samples.extend(samples)

    if not all_samples:
        print("ERROR: No valid Q&A samples found.")
        sys.exit(1)

    print(f"\nTotal training samples: {len(all_samples)}")

    if not args.skip_tokenize:
        try:
            all_samples = tokenize_and_stats(all_samples, student_model)
        except Exception as exc:
            print(f"WARNING: Tokenization failed ({exc}). Skipping stats.")

    output_path = os.path.join(output_dir, "knowledge_train.jsonl")
    with open(output_path, "w") as f:
        for sample in all_samples:
            record = {
                "messages": sample["messages"],
                "unmask": sample["unmask"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"\nTraining mix written to {output_path}")
    print(f"  Total samples: {len(all_samples)}")


if __name__ == "__main__":
    main()
