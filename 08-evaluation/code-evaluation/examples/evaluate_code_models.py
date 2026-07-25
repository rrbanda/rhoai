"""Evaluate coding models on a generated benchmark using pass@1 scoring.

Loads a benchmark produced by generate_code_benchmark.py, evaluates one or more
coding models by having each generate function implementations, then grades
them by running verified test suites in a sandboxed Python interpreter.

Pipeline stages:
  1. Load benchmark (problem_description, test_code, function_code)
  2. For each model:
     a. Generate solutions from problem descriptions via LLMChatBlock
     b. Extract code from model responses
     c. Combine model code + test suite and execute via PythonInterpreterBlock
  3. Compute and report pass@1 scores per model, domain, and difficulty

Usage:
    export API_KEY="your-api-key"

    python evaluate_code_models.py \\
        --benchmark domain_code_benchmark.jsonl \\
        --models gpt-4o gpt-4o-mini \\
        --output evaluation_results.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from sdg_hub.core.blocks.code import PythonInterpreterBlock
from sdg_hub.core.blocks.llm import LLMChatBlock, LLMResponseExtractorBlock


SYSTEM_PROMPT = (
    "You are a Python programmer. Write the requested function. "
    "Output ONLY the Python function code. No explanations, no markdown fences, "
    "no import statements. The function must use only built-in Python features."
)


def evaluate_model(model_name: str, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate a single model on the benchmark.

    Returns a DataFrame with columns: model, domain, difficulty, function_spec,
    passed, error, model_code, problem_description, function_code, test_code.
    """
    df = benchmark_df.copy()

    # Build prompts
    df["solve_prompt"] = df["problem_description"].apply(
        lambda p: [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p},
        ]
    )

    # Generate solutions
    safe_name = model_name.replace(".", "_").replace("-", "_").replace("/", "_")
    solver = LLMChatBlock(
        block_name=f"solve_{safe_name}",
        input_cols="solve_prompt",
        output_cols="raw_solution",
        model=model_name,
        max_tokens=2048,
        temperature=0.0,
        async_mode=True,
    )
    df = solver(df)

    # Extract code from response
    extractor = LLMResponseExtractorBlock(
        block_name="extract_code",
        input_cols="raw_solution",
        extract_content=True,
        expand_lists=True,
    )
    df = extractor(df)

    # Combine model code + test suite and execute
    df["candidate_code"] = df.apply(
        lambda row: str(row.get("extract_code_content", "")) + "\n\n" + row["test_code"],
        axis=1,
    )

    grader = PythonInterpreterBlock(
        block_name="grade",
        input_cols=["candidate_code"],
        output_cols=["grade_result"],
        interpreter_framework="monty",
        timeout=10.0,
    )
    df = grader(df)

    # Extract results
    df["model"] = model_name
    df["model_code"] = df.get("extract_code_content", "")
    df["passed"] = df["grade_result"].apply(lambda r: r.get("success", False))
    df["error"] = df["grade_result"].apply(
        lambda r: r.get("error") if not r.get("success") else None
    )

    result_cols = [
        "model", "domain", "difficulty", "function_spec", "passed", "error",
        "model_code", "problem_description", "function_code", "test_code",
    ]
    return df[[c for c in result_cols if c in df.columns]]


def print_results(results_df: pd.DataFrame) -> None:
    """Print a summary of evaluation results for one model."""
    model = results_df["model"].iloc[0]
    total = len(results_df)
    passed = results_df["passed"].sum()

    print(f"\n{'=' * 70}")
    print(f"  {model}:  pass@1 = {passed}/{total} ({passed / total * 100:.1f}%)")
    print(f"{'=' * 70}")

    print("\n  By domain:")
    for domain, group in results_df.groupby("domain"):
        p = group["passed"].sum()
        t = len(group)
        bar = "#" * p + "." * (t - p)
        print(f"    {domain:<28s}  {p:2d}/{t:2d}  [{bar}]")

    print("\n  By difficulty:")
    for diff in ["beginner", "intermediate", "advanced"]:
        group = results_df[results_df["difficulty"] == diff]
        if len(group) > 0:
            p = group["passed"].sum()
            t = len(group)
            print(f"    {diff:<15s}  {p:2d}/{t:2d}  ({p / t * 100:.0f}%)")


def print_comparison(results_df: pd.DataFrame) -> None:
    """Print a comparison summary across all models."""
    print(f"\n{'=' * 70}")
    print("  Overall pass@1 Comparison")
    print(f"{'=' * 70}")

    summary = results_df.groupby("model")["passed"].agg(["sum", "count", "mean"])
    summary.columns = ["passed", "total", "pass@1"]
    summary = summary.sort_values("pass@1", ascending=False)

    for model, row in summary.iterrows():
        bar = "#" * int(row["pass@1"] * 30) + "." * (30 - int(row["pass@1"] * 30))
        print(f"  {model:<30s}  {int(row['passed']):2d}/{int(row['total']):2d}  ({row['pass@1'] * 100:5.1f}%)  [{bar}]")

    # By domain
    print("\n  pass@1 by Domain:")
    pivot_domain = results_df.groupby(["domain", "model"])["passed"].mean().unstack(fill_value=0)
    print(f"  {pivot_domain.round(2).to_string()}")

    # By difficulty
    print("\n  pass@1 by Difficulty:")
    pivot_diff = results_df.groupby(["difficulty", "model"])["passed"].mean().unstack(fill_value=0)
    diff_order = ["beginner", "intermediate", "advanced"]
    pivot_diff = pivot_diff.reindex([d for d in diff_order if d in pivot_diff.index])
    print(f"  {pivot_diff.round(2).to_string()}")

    # Discriminating problems
    problem_pass = results_df.pivot_table(
        index="function_spec", columns="model", values="passed", aggfunc="first"
    )
    discriminating = problem_pass[problem_pass.nunique(axis=1) > 1]
    if len(discriminating) > 0:
        print(f"\n  Discriminating problems ({len(discriminating)} problems where models disagree):")
        for spec, row in discriminating.iterrows():
            tags = "  ".join(f"{m}: {'PASS' if v else 'FAIL'}" for m, v in row.items())
            print(f"    {str(spec)[:55]:<57s} {tags}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate coding models on a generated benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSONL file (from generate_code_benchmark.py).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names to evaluate (e.g., gpt-4o gpt-4o-mini).",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.jsonl",
        help="Output JSONL path for detailed results (default: evaluation_results.jsonl).",
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

    # Step 1: Load benchmark
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        print(f"ERROR: Benchmark file not found: {benchmark_path}", file=sys.stderr)
        sys.exit(1)

    benchmark = pd.read_json(benchmark_path, lines=True)
    print(f"Benchmark: {len(benchmark)} verified problems")
    print(f"Domains: {sorted(benchmark['domain'].unique())}")
    print(f"Difficulties: {sorted(benchmark['difficulty'].unique())}")

    # Step 2: Evaluate each model
    all_results = []
    for model in args.models:
        print(f"\nEvaluating {model}...")
        try:
            result = evaluate_model(model, benchmark)
            print_results(result)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR evaluating {model}: {e}", file=sys.stderr)

    if not all_results:
        print("ERROR: No models were successfully evaluated.", file=sys.stderr)
        sys.exit(1)

    results_df = pd.concat(all_results, ignore_index=True)

    # Step 3: Compare models
    if len(args.models) > 1:
        print_comparison(results_df)

    # Step 4: Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_json(output_path, orient="records", lines=True)
    print(f"\nSaved {len(results_df)} evaluation records to {output_path}")

    # Final leaderboard
    summary = results_df.groupby("model").agg(
        problems=("passed", "count"),
        passed=("passed", "sum"),
        pass_at_1=("passed", "mean"),
    ).sort_values("pass_at_1", ascending=False)
    summary["pass_at_1"] = (summary["pass_at_1"] * 100).round(1)
    summary = summary.rename(columns={"pass_at_1": "pass@1 (%)"})
    print(f"\nFinal Leaderboard:\n{summary.to_string()}")


if __name__ == "__main__":
    main()
