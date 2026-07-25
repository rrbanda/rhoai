#!/usr/bin/env python3
"""GEPA prompt optimisation -- improve prompts without changing model weights.

Uses evolutionary search with Pareto-based selection and LLM-driven
reflection to evolve a seed prompt into one that maximises task accuracy.
Supports two backends:

- **GEPA** (default): calls ``gepa.optimize()`` directly, saves best
  candidate to ``best_candidate.json``.
- **MLflow**: wraps GEPA with MLflow's prompt registry and experiment
  tracking via ``mlflow.genai.optimize_prompts()``.

Adapted from the Training Hub ``gepa_prompt_optimization`` notebook.

Prerequisites:
    pip install 'training-hub[gepa]'
    # Start a vLLM server with a >= 3B model (see README.md)

Usage:
    # GEPA backend (default)
    python gepa_prompt_optimization.py \\
        --data-path ./qa_data.jsonl \\
        --output-dir ./gepa_output

    # MLflow backend
    python gepa_prompt_optimization.py \\
        --data-path ./qa_data.jsonl \\
        --output-dir ./gepa_output \\
        --backend mlflow

    # Generate sample data, then optimise
    python gepa_prompt_optimization.py \\
        --generate-sample-data \\
        --output-dir ./gepa_output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from training_hub import gepa

SAMPLE_DATA = [
    {
        "input": "A store sells apples for $3 each and oranges for $5 each. "
        "If you buy 4 apples and 3 oranges, how much do you spend in total?",
        "answer": "27",
    },
    {
        "input": "A train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours. "
        "What is the total distance traveled in miles?",
        "answer": "270",
    },
    {
        "input": "A recipe needs 2/3 cup of sugar. If you want to make 4 batches, "
        "how many cups of sugar do you need? Express as a fraction.",
        "answer": "8/3",
    },
    {
        "input": "A shirt originally costs $80. It is on sale for 25% off, and you have "
        "an additional 10% off coupon applied after the sale discount. What do you pay?",
        "answer": "54",
    },
    {
        "input": "A rectangular garden is 12 meters long and 8 meters wide. "
        "What is its area in square meters?",
        "answer": "96",
    },
    {
        "input": "In a class of 40 students, 60% passed the exam. Of those who passed, "
        "75% scored above 80. How many students scored above 80?",
        "answer": "18",
    },
    {
        "input": "A car's fuel tank holds 50 liters. The car uses 8 liters per 100 km. "
        "How many kilometers can the car travel on a full tank?",
        "answer": "625",
    },
    {
        "input": "If 3 workers can paint a house in 12 days, how many days would it take "
        "5 workers to paint the same house?",
        "answer": "7.2",
    },
    {
        "input": "What is the sum of the first 10 positive odd numbers?",
        "answer": "100",
    },
    {
        "input": "A ball is dropped from 100 meters. Each bounce reaches half the previous "
        "height. What is the maximum height after the third bounce, in meters?",
        "answer": "12.5",
    },
    {
        "input": "A factory produces 150 widgets per hour. It operates 8 hours a day, "
        "5 days a week. How many widgets does it produce in 4 weeks?",
        "answer": "24000",
    },
    {
        "input": "If a 2-liter bottle of juice is shared equally among 8 people, "
        "how many milliliters does each person get?",
        "answer": "250",
    },
    {
        "input": "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
        "answer": "9",
    },
    {
        "input": "How many times does the digit 3 appear in all numbers from 1 to 50?",
        "answer": "15",
    },
    {
        "input": "You have a 3-gallon jug and a 5-gallon jug. You fill the 5-gallon jug "
        "completely and pour into the 3-gallon jug until full. How many gallons "
        "are left in the 5-gallon jug?",
        "answer": "2",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GEPA prompt optimisation -- evolve prompts via evolutionary search.",
    )

    parser.add_argument("--data-path", help="JSONL with 'input' and 'answer' fields")
    parser.add_argument(
        "--output-dir", default="./gepa_output", help="Output directory (default: ./gepa_output)"
    )
    parser.add_argument(
        "--backend",
        choices=["gepa", "mlflow"],
        default="gepa",
        help="Optimisation backend (default: gepa)",
    )
    parser.add_argument(
        "--seed-prompt",
        default="Answer the question.",
        help="Starting prompt to optimise (default: 'Answer the question.')",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GEPA_MODEL", "openai/Qwen/Qwen2.5-3B-Instruct"),
        help="Task/reflection model in litellm format (default: openai/Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("GEPA_API_BASE", "http://localhost:8000/v1"),
        help="vLLM or OpenAI-compatible endpoint (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=300,
        help="Evaluation budget (default: 300, use >= 15 * dataset_size)",
    )
    parser.add_argument(
        "--reflection-minibatch-size",
        type=int,
        default=5,
        help="Examples per reflection step (default: 5)",
    )
    parser.add_argument(
        "--generate-sample-data",
        action="store_true",
        help="Generate sample math word problems and use them as training data",
    )

    return parser.parse_args()


def run_gepa_backend(args: argparse.Namespace, data_path: str) -> None:
    """Run optimisation with the native GEPA backend."""
    seed = {"system_prompt": args.seed_prompt}
    result = gepa(
        seed_candidate=seed,
        task_lm=args.model,
        api_base=args.api_base,
        data_path=data_path,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        output_dir=os.path.join(args.output_dir, "gepa_backend"),
    )

    print()
    print("=" * 60)
    print("Optimisation complete (GEPA backend)")
    print("=" * 60)
    print("Best prompt:")
    for field, text in result.best_candidate.items():
        print(f"  {field}: {text}")
    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"\nBest score: {best_score:.2f}")
    print(f"Candidates explored: {len(result.candidates)}")
    print(f"Starting prompt: {seed}")


def run_mlflow_backend(args: argparse.Namespace, data_path: str) -> None:
    """Run optimisation with the MLflow backend."""
    import litellm
    import mlflow
    from mlflow.genai.scorers import scorer

    # Set up MLflow tracking
    mlflow_dir = Path(args.output_dir) / "mlflow_tracking"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir.resolve()}")
    mlflow.set_experiment("gepa-prompt-optimization")

    prompt = mlflow.genai.register_prompt(
        name="qa_system_prompt",
        template="{{ system_prompt }}\n\nQuestion: {{ input }}\nAnswer:",
    )

    def predict_fn(input: str) -> str:  # noqa: A002
        response = litellm.completion(
            model=args.model,
            messages=[{"role": "user", "content": input}],
            api_base=args.api_base,
            max_tokens=64,
        )
        return response.choices[0].message.content

    @scorer
    def answer_contains(inputs, outputs, expectations):  # noqa: ANN001, ANN201
        import re

        expected = (expectations or {}).get("expected_response", "")
        clean = re.sub(r"<think>.*?</think>", "", outputs or "", flags=re.DOTALL).strip()
        return 1.0 if expected.lower() in clean.lower() else 0.0

    # Load and convert data
    with open(data_path) as f:
        raw_data = [json.loads(line) for line in f]

    mlflow_data = [
        {
            "inputs": {"input": entry["input"]},
            "expectations": {"expected_response": entry["answer"]},
        }
        for entry in raw_data
    ]

    result = gepa(
        seed_candidate={"system_prompt": args.seed_prompt},
        task_lm=args.model,
        api_base=args.api_base,
        backend="mlflow",
        predict_fn=predict_fn,
        prompt_uris=[prompt.uri],
        scorers=[answer_contains],
        trainset=mlflow_data,
        enable_tracking=False,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        output_dir=os.path.join(args.output_dir, "mlflow_backend"),
    )

    print()
    print("=" * 60)
    print("Optimisation complete (MLflow backend)")
    print("=" * 60)
    print(f"Initial score: {result.initial_eval_score:.2f}")
    print(f"Final score:   {result.final_eval_score:.2f}")
    for p in result.optimized_prompts:
        print(f"Optimised prompt URI: {p.uri}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Ensure API key is set (vLLM ignores it but litellm requires one)
    os.environ.setdefault("OPENAI_API_KEY", "dummy")

    # Resolve data path
    if args.generate_sample_data or not args.data_path:
        data_path = os.path.join(args.output_dir, "sample_data.jsonl")
        with open(data_path, "w") as f:
            for entry in SAMPLE_DATA:
                f.write(json.dumps(entry) + "\n")
        print(f"Generated sample data: {data_path} ({len(SAMPLE_DATA)} examples)")
    else:
        data_path = args.data_path
        if not os.path.exists(data_path):
            print(f"Error: data file not found: {data_path}", file=sys.stderr)
            sys.exit(1)

    print(f"Model:    {args.model}")
    print(f"Endpoint: {args.api_base}")
    print(f"Backend:  {args.backend}")
    print(f"Budget:   {args.max_metric_calls} metric calls")
    print()

    if args.backend == "gepa":
        run_gepa_backend(args, data_path)
    else:
        run_mlflow_backend(args, data_path)


if __name__ == "__main__":
    main()
