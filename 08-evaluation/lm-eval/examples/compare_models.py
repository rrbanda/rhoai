"""Compare base vs fine-tuned model scores using LM-Eval benchmarks.

Runs the same set of benchmarks against two model endpoints (base and
fine-tuned), collects scores from completed LMEvalJob results, and prints
a side-by-side comparison table with improvement percentages.

Usage:
    python compare_models.py \\
        --base-model-url https://granite-base-eval-lm.apps.cluster.example.com/v1/completions \\
        --base-model-name granite-3-8b-instruct \\
        --tuned-model-url https://granite-tuned-eval-lm.apps.cluster.example.com/v1/completions \\
        --tuned-model-name granite-3-8b-instruct-ft \\
        --tasks mmlu hellaswag arc_challenge

    python compare_models.py \\
        --base-model-url https://base.example.com/v1/completions \\
        --base-model-name llama-3-8b \\
        --tuned-model-url https://tuned.example.com/v1/completions \\
        --tuned-model-name llama-3-8b-ft \\
        --tasks mmlu gsm8k truthfulqa_mc2 \\
        --num-fewshot 5
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from kubernetes import client, config
from kubernetes.client.rest import ApiException


API_GROUP = "trustyai.opendatahub.io"
API_VERSION = "v1alpha1"
PLURAL = "lmevaljobs"

SUPPORTED_TASKS = [
    "mmlu",
    "hellaswag",
    "arc_challenge",
    "gsm8k",
    "truthfulqa_mc2",
]

DEFAULT_BATCH_SIZE = "8"
DEFAULT_NUM_FEWSHOT = 5
DEFAULT_NAMESPACE = "eval-lm"
POLL_INTERVAL_SECONDS = 30


def create_eval_job(
    custom_api: client.CustomObjectsApi,
    model_name: str,
    model_url: str,
    tasks: list[str],
    num_fewshot: int,
    batch_size: str,
    namespace: str,
    suffix: str,
) -> str:
    """Create an LMEvalJob for a model and return the job name."""
    task_slug = "-".join(tasks)[:20]
    job_name = f"cmp-{suffix}-{task_slug}-{num_fewshot}shot"
    job_name = job_name[:63].rstrip("-")

    manifest = {
        "apiVersion": f"{API_GROUP}/{API_VERSION}",
        "kind": "LMEvalJob",
        "metadata": {
            "name": job_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/part-of": "lm-eval",
                "evaluation/model": model_name,
                "evaluation/comparison": "true",
            },
        },
        "spec": {
            "model": "local-completions",
            "modelArgs": [
                {"name": "model", "value": model_name},
                {"name": "base_url", "value": model_url},
                {"name": "tokenized_requests", "value": "False"},
            ],
            "taskList": {"taskNames": tasks},
            "numFewShot": num_fewshot,
            "batchSize": batch_size,
            "logSamples": True,
            "pod": {
                "container": {
                    "resources": {
                        "requests": {
                            "cpu": "4",
                            "memory": "8Gi",
                            "nvidia.com/gpu": "1",
                        },
                        "limits": {
                            "cpu": "8",
                            "memory": "16Gi",
                            "nvidia.com/gpu": "1",
                        },
                    }
                }
            },
            "outputs": {"pvcManaged": {"size": "5Gi"}},
        },
    }

    try:
        custom_api.create_namespaced_custom_object(
            group=API_GROUP,
            version=API_VERSION,
            namespace=namespace,
            plural=PLURAL,
            body=manifest,
        )
    except ApiException as e:
        if e.status == 409:
            print(f"  Job '{job_name}' already exists, reusing.")
        else:
            print(f"ERROR: Failed to create LMEvalJob '{job_name}': {e.reason}", file=sys.stderr)
            sys.exit(1)

    return job_name


def wait_for_job(
    custom_api: client.CustomObjectsApi,
    name: str,
    namespace: str,
    timeout: int = 3600,
) -> dict[str, Any]:
    """Poll until the LMEvalJob reaches a terminal state."""
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"ERROR: Timeout waiting for '{name}'.", file=sys.stderr)
            sys.exit(1)

        try:
            job = custom_api.get_namespaced_custom_object(
                group=API_GROUP,
                version=API_VERSION,
                namespace=namespace,
                plural=PLURAL,
                name=name,
            )
        except ApiException as e:
            print(f"ERROR: Failed to get status for '{name}': {e.reason}", file=sys.stderr)
            sys.exit(1)

        state = job.get("status", {}).get("state", "Unknown")
        if state == "Complete":
            return job
        if state in ("Failed", "Error"):
            reason = job.get("status", {}).get("reason", "Unknown")
            print(f"ERROR: Job '{name}' failed: {reason}", file=sys.stderr)
            sys.exit(1)

        time.sleep(POLL_INTERVAL_SECONDS)


def collect_scores(job: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract per-task scores from a completed LMEvalJob.

    Returns:
        Mapping of task_name -> {metric_name: score}.
    """
    results = job.get("status", {}).get("results", {})
    scores: dict[str, dict[str, float]] = {}

    for task_name, task_data in results.items():
        if isinstance(task_data, dict):
            scores[task_name] = {
                k: v for k, v in task_data.items() if isinstance(v, (int, float))
            }
        elif isinstance(task_data, (int, float)):
            scores[task_name] = {"score": task_data}

    return scores


def print_comparison_table(
    base_scores: dict[str, dict[str, float]],
    tuned_scores: dict[str, dict[str, float]],
    base_name: str,
    tuned_name: str,
) -> None:
    """Print a formatted side-by-side comparison with improvement percentages."""
    all_tasks = sorted(set(list(base_scores.keys()) + list(tuned_scores.keys())))

    header_base = base_name[:20]
    header_tuned = tuned_name[:20]

    print(f"\n{'=' * 80}")
    print(f"  Model Comparison: {base_name} vs {tuned_name}")
    print(f"{'=' * 80}")
    print(
        f"  {'Task':<20s} {'Metric':<15s} "
        f"{header_base:>12s} {header_tuned:>12s} {'Improvement':>12s}"
    )
    print(f"  {'-' * 20} {'-' * 15} {'-' * 12} {'-' * 12} {'-' * 12}")

    improvements: list[float] = []

    for task in all_tasks:
        base_task = base_scores.get(task, {})
        tuned_task = tuned_scores.get(task, {})
        all_metrics = sorted(set(list(base_task.keys()) + list(tuned_task.keys())))

        for metric in all_metrics:
            base_val = base_task.get(metric)
            tuned_val = tuned_task.get(metric)

            base_str = f"{base_val:.4f}" if base_val is not None else "N/A"
            tuned_str = f"{tuned_val:.4f}" if tuned_val is not None else "N/A"

            if base_val is not None and tuned_val is not None and base_val > 0:
                improvement = ((tuned_val - base_val) / base_val) * 100
                improvements.append(improvement)
                sign = "+" if improvement >= 0 else ""
                imp_str = f"{sign}{improvement:.1f}%"
            else:
                imp_str = "---"

            print(
                f"  {task:<20s} {metric:<15s} "
                f"{base_str:>12s} {tuned_str:>12s} {imp_str:>12s}"
            )

    print(f"  {'-' * 20} {'-' * 15} {'-' * 12} {'-' * 12} {'-' * 12}")

    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        sign = "+" if avg_improvement >= 0 else ""
        print(f"  {'Average improvement:':<37s} {'':>12s} {'':>12s} {sign}{avg_improvement:.1f}%")

    print(f"{'=' * 80}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base vs fine-tuned model using LM-Eval benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-model-url",
        required=True,
        help="OpenAI-compatible endpoint URL for the base model.",
    )
    parser.add_argument(
        "--base-model-name",
        required=True,
        help="Identifier for the base model.",
    )
    parser.add_argument(
        "--tuned-model-url",
        required=True,
        help="OpenAI-compatible endpoint URL for the fine-tuned model.",
    )
    parser.add_argument(
        "--tuned-model-name",
        required=True,
        help="Identifier for the fine-tuned model.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["mmlu", "hellaswag", "arc_challenge"],
        choices=SUPPORTED_TASKS,
        help=f"Benchmark tasks to run (default: mmlu hellaswag arc_challenge).",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=DEFAULT_NUM_FEWSHOT,
        help=f"Number of few-shot examples (default: {DEFAULT_NUM_FEWSHOT}).",
    )
    parser.add_argument(
        "--batch-size",
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for evaluation (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--namespace",
        default=DEFAULT_NAMESPACE,
        help=f"Kubernetes namespace (default: {DEFAULT_NAMESPACE}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Max seconds to wait for each job (default: 7200).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    custom_api = client.CustomObjectsApi()

    # Step 1: Create evaluation jobs for both models
    print(f"\n--- Step 1: Creating evaluation jobs ---")
    print(f"  Tasks: {args.tasks}")
    print(f"  Few-shot: {args.num_fewshot}")

    base_job_name = create_eval_job(
        custom_api,
        model_name=args.base_model_name,
        model_url=args.base_model_url,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        namespace=args.namespace,
        suffix="base",
    )
    print(f"  Base model job: {base_job_name}")

    tuned_job_name = create_eval_job(
        custom_api,
        model_name=args.tuned_model_name,
        model_url=args.tuned_model_url,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        namespace=args.namespace,
        suffix="tuned",
    )
    print(f"  Tuned model job: {tuned_job_name}")

    # Step 2: Wait for both jobs to complete
    print(f"\n--- Step 2: Waiting for evaluations to complete ---")
    print(f"  Monitoring base model job...")
    base_job = wait_for_job(custom_api, base_job_name, args.namespace, args.timeout)
    print(f"  Base model evaluation complete.")

    print(f"  Monitoring tuned model job...")
    tuned_job = wait_for_job(custom_api, tuned_job_name, args.namespace, args.timeout)
    print(f"  Tuned model evaluation complete.")

    # Step 3: Collect scores
    print(f"\n--- Step 3: Collecting scores ---")
    base_scores = collect_scores(base_job)
    tuned_scores = collect_scores(tuned_job)

    if not base_scores and not tuned_scores:
        print("ERROR: No scores available from either job.", file=sys.stderr)
        sys.exit(1)

    # Step 4: Print comparison
    print(f"\n--- Step 4: Comparison ---")
    print_comparison_table(base_scores, tuned_scores, args.base_model_name, args.tuned_model_name)

    print("\nComparison complete.")


if __name__ == "__main__":
    main()
