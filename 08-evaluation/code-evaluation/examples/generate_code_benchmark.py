"""Generate an execution-verified coding benchmark using SDG Hub.

Creates domain-specific coding problems where every problem is backed by a
reference solution and test suite verified via sandboxed execution
(PythonInterpreterBlock).

Pipeline stages:
  1. Define domain and function specifications (seed data)
  2. Load the domain-code-eval flow from SDG Hub
  3. Run the pipeline: LLM generates function → LLM generates tests
     → PythonInterpreterBlock verifies → LLM generates problem description
  4. Filter to execution-verified problems only
  5. Save the benchmark as JSONL

Usage:
    export INFERENCE_MODEL="openai/gpt-4o"
    export API_KEY="your-api-key"

    python generate_code_benchmark.py --output domain_code_benchmark.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from sdg_hub import Flow, FlowRegistry


def build_seed_data() -> pd.DataFrame:
    """Build the seed dataset of function specifications across six domains.

    Each entry specifies what the function should do and its expected time
    complexity. The LLM generates the implementation, tests, and problem
    description from these specs.
    """
    specs: dict[str, dict[str, list[tuple[str, str]]]] = {
        "financial calculations": {
            "beginner": [
                ("simple interest calculator given principal, rate, and time", "O(1)"),
                ("percentage change between two values", "O(1)"),
                ("discount price given original price and discount percentage", "O(1)"),
                ("tip calculator that splits bill among people", "O(1)"),
            ],
            "intermediate": [
                ("compound interest with variable compounding periods", "O(1)"),
                ("loan monthly payment calculator using amortization formula", "O(1)"),
                ("tax bracket calculator with progressive brackets given as list of (threshold, rate) tuples", "O(n)"),
                ("weighted average calculator given values and weights", "O(n)"),
                ("moving average of a list of prices with configurable window size", "O(n)"),
                ("present value of future cash flows with given discount rate", "O(n)"),
                ("profit margin calculator (gross and net) from revenue and costs", "O(1)"),
            ],
            "advanced": [
                ("bond price calculator from coupon rate, yield, face value, and periods", "O(n)"),
                ("internal rate of return (IRR) approximation using bisection method", "O(n log(1/eps))"),
                ("depreciation schedule using declining balance method", "O(n)"),
                ("currency converter with cross-rate calculation through a base currency", "O(n)"),
            ],
        },
        "text processing": {
            "beginner": [
                ("word frequency counter returning dict of word counts", "O(n)"),
                ("title case converter that keeps small words (a, an, the, of) lowercase", "O(n)"),
                ("vowel and consonant counter returning a dict with counts", "O(n)"),
                ("string truncator that adds ellipsis if string exceeds max length", "O(1)"),
            ],
            "intermediate": [
                ("run-length encoder that compresses 'aaabbc' to '3a2b1c'", "O(n)"),
                ("run-length decoder that expands '3a2b1c' to 'aaabbc'", "O(n)"),
                ("Caesar cipher encoder/decoder with configurable shift", "O(n)"),
                ("bracket validator that checks matching parentheses, brackets, and braces", "O(n)"),
                ("slug generator that converts titles to URL-safe lowercase strings", "O(n)"),
                ("sentence splitter that handles abbreviations like 'Dr.' and 'U.S.A.'", "O(n)"),
                ("text wrapper that breaks text into lines of max width without splitting words", "O(n)"),
            ],
            "advanced": [
                ("regex-free pattern matcher supporting * (any chars) and ? (single char) wildcards", "O(n*m)"),
                ("Levenshtein edit distance calculator between two strings", "O(n*m)"),
                ("longest common subsequence of two strings", "O(n*m)"),
                ("template string renderer that replaces {{variable}} placeholders from a dict", "O(n*k)"),
            ],
        },
        "data transformation": {
            "beginner": [
                ("flatten a nested list of arbitrary depth into a single flat list", "O(n)"),
                ("deduplicate a list while preserving original order", "O(n)"),
                ("chunk a list into groups of n elements", "O(n)"),
                ("zip two dictionaries by shared keys into a dict of tuples", "O(n)"),
            ],
            "intermediate": [
                ("group a list of dicts by a given key field", "O(n)"),
                ("pivot a list of records: given rows with (category, metric, value), produce {category: {metric: value}}", "O(n)"),
                ("transpose a matrix (list of lists)", "O(n*m)"),
                ("merge two sorted lists into one sorted list", "O(n+m)"),
                ("invert a dictionary (swap keys and values, handling duplicate values as lists)", "O(n)"),
                ("running sum of a list of numbers", "O(n)"),
                ("interleave two lists element by element, handling unequal lengths", "O(n+m)"),
            ],
            "advanced": [
                ("deep merge two nested dicts recursively (lists concatenate, dicts merge, scalars overwrite)", "O(n)"),
                ("topological sort of a dependency graph given as adjacency dict", "O(V+E)"),
                ("JSON path extractor: get nested value from dict using dot notation like 'a.b.0.c'", "O(d)"),
                ("diff two flat dicts returning added, removed, and changed keys with old/new values", "O(n)"),
            ],
        },
        "mathematical functions": {
            "beginner": [
                ("greatest common divisor (GCD) of two numbers using Euclidean algorithm", "O(log(min(a,b)))"),
                ("check if a number is prime", "O(sqrt(n))"),
                ("factorial of a non-negative integer", "O(n)"),
                ("Fibonacci number at position n (iterative)", "O(n)"),
            ],
            "intermediate": [
                ("prime factorization returning list of (prime, exponent) tuples", "O(sqrt(n))"),
                ("combination calculator nCr without using factorial for large numbers", "O(min(r, n-r))"),
                ("mean, median, and mode of a list of numbers returned as a dict", "O(n log n)"),
                ("standard deviation of a list of numbers (population)", "O(n)"),
                ("Euclidean distance between two points in n-dimensional space", "O(n)"),
                ("base converter from decimal to any base (2-36) returning string", "O(log n)"),
                ("matrix multiplication for two 2D matrices (lists of lists)", "O(n^3)"),
            ],
            "advanced": [
                ("Newton's method for finding square root to specified precision", "O(log(1/eps))"),
                ("polynomial evaluator from list of coefficients using Horner's method", "O(n)"),
                ("numerical integration using Simpson's rule", "O(n)"),
                ("solve a system of 2 linear equations (2x2) returning (x, y) or None if no solution", "O(1)"),
            ],
        },
        "algorithms": {
            "beginner": [
                ("binary search returning index or -1 if not found", "O(log n)"),
                ("bubble sort implementation", "O(n^2)"),
                ("selection sort implementation", "O(n^2)"),
                ("reverse a singly linked list represented as list of (value, next_index) tuples", "O(n)"),
            ],
            "intermediate": [
                ("merge sort implementation", "O(n log n)"),
                ("stack implementation using a list with push, pop, peek, is_empty methods (as a class)", "O(1) per operation"),
                ("queue implementation using a list with enqueue, dequeue, peek, is_empty methods (as a class)", "O(1) per operation"),
                ("breadth-first traversal of a tree represented as adjacency dict returning list of values", "O(V+E)"),
                ("depth-first traversal (pre-order) of a tree represented as adjacency dict", "O(V+E)"),
                ("binary search tree: insert, search, and in-order traversal (as a class)", "O(n) worst case"),
                ("0/1 knapsack problem returning maximum value given weights, values, and capacity", "O(n*W)"),
            ],
            "advanced": [
                ("Dijkstra's shortest path algorithm on weighted adjacency dict", "O(V^2)"),
                ("longest increasing subsequence length", "O(n log n)"),
                ("minimum number of coins for a given amount (coin change problem)", "O(n*amount)"),
                ("detect cycle in a directed graph given as adjacency dict", "O(V+E)"),
            ],
        },
        "string utilities": {
            "beginner": [
                ("palindrome checker (ignoring case and non-alphanumeric characters)", "O(n)"),
                ("anagram checker for two strings", "O(n)"),
                ("count occurrences of each character returning sorted dict", "O(n log n)"),
                ("camelCase to snake_case converter", "O(n)"),
            ],
            "intermediate": [
                ("snake_case to camelCase converter", "O(n)"),
                ("version string comparator: return -1, 0, or 1 for two version strings like '1.2.3'", "O(n)"),
                ("Roman numeral to integer converter", "O(n)"),
                ("integer to Roman numeral converter", "O(1)"),
                ("IP address validator (IPv4) returning True/False", "O(1)"),
                ("credit card number masker showing only last 4 digits", "O(n)"),
            ],
            "advanced": [
                ("URL query string parser returning nested dict from 'a=1&b=2&c[0]=x&c[1]=y'", "O(n)"),
                ("simple expression evaluator for +, -, *, / with parentheses (no eval)", "O(n)"),
                ("password strength scorer (0-100) based on length, diversity, patterns", "O(n)"),
            ],
        },
    }

    rows = []
    for domain, by_difficulty in specs.items():
        for difficulty, func_specs in by_difficulty.items():
            for spec, complexity in func_specs:
                rows.append(
                    {
                        "domain": domain,
                        "function_spec": spec,
                        "difficulty": difficulty,
                        "time_complexity": complexity,
                    }
                )
    return pd.DataFrame(rows)


def configure_model(flow: Flow) -> Flow:
    """Apply model configuration from environment variables."""
    model = os.getenv("INFERENCE_MODEL", "")
    api_base = os.getenv("URL", "")
    api_key = os.getenv("API_KEY", "")

    if not model:
        print("ERROR: INFERENCE_MODEL environment variable must be set.", file=sys.stderr)
        sys.exit(1)

    if not model.startswith(("openai/", "ollama/")):
        model = "openai/" + model

    print(f"Model: {model}")
    flow.set_model_config(
        model=model,
        api_base=api_base or None,
        api_key=api_key or None,
    )
    return flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an execution-verified coding benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        default="domain_code_benchmark.jsonl",
        help="Output JSONL path (default: domain_code_benchmark.jsonl).",
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

    # Step 1: Build seed data
    print("\n--- Step 1: Building seed data ---")
    seed_data = build_seed_data()
    print(f"Seed dataset: {len(seed_data)} specs across {seed_data['domain'].nunique()} domains")
    print(f"Distribution by domain:\n{seed_data['domain'].value_counts().to_string()}")
    print(f"Distribution by difficulty:\n{seed_data['difficulty'].value_counts().to_string()}")

    # Step 2: Load the domain-code-eval flow
    print("\n--- Step 2: Loading domain-code-eval flow ---")
    FlowRegistry.discover_flows()
    flow = Flow.from_yaml(FlowRegistry.get_flow_path("domain-code-eval"))
    print("Flow loaded: domain-code-eval")

    # Step 3: Configure the model
    print("\n--- Step 3: Configuring model ---")
    flow = configure_model(flow)

    # Step 4: Generate the benchmark
    print("\n--- Step 4: Generating benchmark ---")
    print("The flow makes 3 LLM calls + 1 execution call per row.")
    print(f"Processing {len(seed_data)} specs...")

    results = flow.generate(seed_data)
    results_df = results.to_pandas() if hasattr(results, "to_pandas") else pd.DataFrame(results)
    print(f"Generated {len(results_df)} benchmark problems")

    # Step 5: Filter to verified problems
    print("\n--- Step 5: Filtering to execution-verified problems ---")
    results_df["verified"] = results_df["execution_result"].apply(
        lambda r: r.get("success", False) if isinstance(r, dict) else False
    )

    passed = results_df["verified"].sum()
    total = len(results_df)
    print(f"Verification: {passed}/{total} passed ({passed / total * 100:.1f}%)")

    print("\nPass rate by domain:")
    for domain, group in results_df.groupby("domain"):
        p = group["verified"].sum()
        t = len(group)
        print(f"  {domain:25s}  {p:2d}/{t:2d}  ({p / t * 100:.0f}%)")

    print("\nPass rate by difficulty:")
    for diff in ["beginner", "intermediate", "advanced"]:
        group = results_df[results_df["difficulty"] == diff]
        if len(group) > 0:
            p = group["verified"].sum()
            t = len(group)
            print(f"  {diff:15s}  {p:2d}/{t:2d}  ({p / t * 100:.0f}%)")

    # Step 6: Export
    benchmark_cols = [
        "domain",
        "difficulty",
        "time_complexity",
        "function_spec",
        "problem_description",
        "function_code",
        "test_code",
    ]
    available_cols = [c for c in benchmark_cols if c in results_df.columns]
    if "input_generator" in results_df.columns:
        available_cols.append("input_generator")

    benchmark = results_df[results_df["verified"]][available_cols].reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_json(output_path, orient="records", lines=True)

    print(f"\nSaved {len(benchmark)} verified problems to {output_path}")
    print(f"Domains: {sorted(benchmark['domain'].unique())}")

    if len(benchmark) > 0:
        ex = benchmark.iloc[0]
        print(f"\n--- Sample Problem ---")
        print(f"Domain: {ex['domain']} | Difficulty: {ex['difficulty']}")
        print(f"Spec: {ex['function_spec']}")
        desc = str(ex.get("problem_description", ""))
        print(f"Problem (first 200 chars): {desc[:200]}...")


if __name__ == "__main__":
    main()
