"""Run LM-Eval benchmarks on RHOAI using LMEvalJob custom resources.

Creates LMEvalJob resources programmatically via the Kubernetes API, monitors
their execution until completion, and retrieves evaluation metrics. Supports
standard benchmarks (MMLU, HellaSwag, ARC-Challenge, GSM8K, TruthfulQA) against
either a served model endpoint or a local model path.

Usage:
    python run_lm_eval.py \\
        --model-url https://granite-3-8b-instruct-eval-lm.apps.cluster.example.com/v1/completions \\
        --model-name granite-3-8b-instruct \\
        --tasks mmlu hellaswag \\
        --num-fewshot 5

    python run_lm_eval.py \\
        --model-path /mnt/models/granite-3-8b-instruct \\
        --model-name granite-3-8b-instruct \\
        --tasks arc_challenge \\
        --num-fewshot 25 \\
        --namespace my-eval-ns
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


def build_lmevaljob_manifest(
    name: str,
    model_name: str,
    tasks: list[str],
    namespace: str,
    model_url: str | None = None,
    model_path: str | None = None,
    num_fewshot: int = DEFAULT_NUM_FEWSHOT,
    batch_size: str = DEFAULT_BATCH_SIZE,
    log_samples: bool = True,
) -> dict[str, Any]:
    """Build an LMEvalJob manifest dictionary.

    Args:
        name: Name for the LMEvalJob resource.
        model_name: Identifier of the model being evaluated.
        tasks: List of benchmark task names.
        namespace: Kubernetes namespace to deploy in.
        model_url: URL of the OpenAI-compatible model endpoint.
        model_path: Filesystem path to a local model (mutually exclusive with model_url).
        num_fewshot: Number of few-shot examples for evaluation.
        batch_size: Batch size for inference requests.
        log_samples: Whether to log individual sample results.

    Returns:
        A dictionary representing the LMEvalJob manifest.
    """
    model_args = [
        {"name": "model", "value": model_name},
    ]

    if model_url:
        model_type = "local-completions"
        model_args.append({"name": "base_url", "value": model_url})
        model_args.append({"name": "tokenized_requests", "value": "False"})
    elif model_path:
        model_type = "hf"
        model_args.append({"name": "pretrained", "value": model_path})
    else:
        raise ValueError("Either --model-url or --model-path must be provided.")

    return {
        "apiVersion": f"{API_GROUP}/{API_VERSION}",
        "kind": "LMEvalJob",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/part-of": "lm-eval",
                "evaluation/model": model_name,
                "evaluation/benchmark": "-".join(tasks),
            },
        },
        "spec": {
            "model": model_type,
            "modelArgs": model_args,
            "taskList": {"taskNames": tasks},
            "numFewShot": num_fewshot,
            "batchSize": batch_size,
            "logSamples": log_samples,
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


def create_lmevaljob(
    custom_api: client.CustomObjectsApi,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Create an LMEvalJob resource in the cluster."""
    namespace = manifest["metadata"]["namespace"]
    try:
        result = custom_api.create_namespaced_custom_object(
            group=API_GROUP,
            version=API_VERSION,
            namespace=namespace,
            plural=PLURAL,
            body=manifest,
        )
        return result
    except ApiException as e:
        print(f"ERROR: Failed to create LMEvalJob: {e.reason}", file=sys.stderr)
        if e.status == 409:
            print("  Resource already exists. Delete it first or use a different name.", file=sys.stderr)
        sys.exit(1)


def wait_for_completion(
    custom_api: client.CustomObjectsApi,
    name: str,
    namespace: str,
    timeout: int = 3600,
) -> dict[str, Any]:
    """Poll the LMEvalJob until it reaches a terminal state.

    Args:
        custom_api: Kubernetes custom objects API client.
        name: Name of the LMEvalJob.
        namespace: Namespace of the LMEvalJob.
        timeout: Maximum seconds to wait before giving up.

    Returns:
        The final LMEvalJob resource.
    """
    start = time.time()
    print(f"Waiting for LMEvalJob '{name}' to complete (timeout: {timeout}s)...")

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"ERROR: Timeout after {timeout}s waiting for completion.", file=sys.stderr)
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
            print(f"ERROR: Failed to get LMEvalJob status: {e.reason}", file=sys.stderr)
            sys.exit(1)

        status = job.get("status", {})
        state = status.get("state", "Unknown")
        print(f"  [{int(elapsed)}s] State: {state}")

        if state == "Complete":
            return job
        if state in ("Failed", "Error"):
            reason = status.get("reason", "Unknown")
            print(f"ERROR: LMEvalJob failed: {reason}", file=sys.stderr)
            sys.exit(1)

        time.sleep(POLL_INTERVAL_SECONDS)


def extract_results(job: dict[str, Any]) -> dict[str, Any]:
    """Extract evaluation results from a completed LMEvalJob."""
    status = job.get("status", {})
    results = status.get("results", {})
    return results


def print_results(results: dict[str, Any], model_name: str) -> None:
    """Print evaluation results in a formatted table."""
    print(f"\n{'=' * 70}")
    print(f"  Evaluation Results: {model_name}")
    print(f"{'=' * 70}")

    if not results:
        print("  No results available.")
        return

    print(f"  {'Task':<25s} {'Metric':<20s} {'Score':<10s}")
    print(f"  {'-' * 25} {'-' * 20} {'-' * 10}")

    for task_name, task_results in results.items():
        if isinstance(task_results, dict):
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    print(f"  {task_name:<25s} {metric:<20s} {value:<10.4f}")
        elif isinstance(task_results, (int, float)):
            print(f"  {task_name:<25s} {'score':<20s} {task_results:<10.4f}")

    print(f"{'=' * 70}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LM-Eval benchmarks on RHOAI via LMEvalJob CRDs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-url",
        help="URL of the OpenAI-compatible model endpoint (e.g., https://host/v1/completions).",
    )
    model_group.add_argument(
        "--model-path",
        help="Local filesystem path to the model (for on-cluster evaluation).",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Identifier for the model being evaluated.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["mmlu"],
        choices=SUPPORTED_TASKS,
        help=f"Benchmark tasks to run (default: mmlu). Choices: {SUPPORTED_TASKS}",
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
        default=3600,
        help="Max seconds to wait for job completion (default: 3600).",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Custom name for the LMEvalJob resource (auto-generated if not set).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Create the job and exit without waiting for results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load kube config
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    custom_api = client.CustomObjectsApi()

    # Build job name
    task_slug = "-".join(args.tasks)[:30]
    job_name = args.job_name or f"eval-{args.model_name}-{task_slug}-{args.num_fewshot}shot"
    job_name = job_name[:63].rstrip("-")

    # Step 1: Build manifest
    print(f"\n--- Step 1: Building LMEvalJob manifest ---")
    manifest = build_lmevaljob_manifest(
        name=job_name,
        model_name=args.model_name,
        tasks=args.tasks,
        namespace=args.namespace,
        model_url=args.model_url,
        model_path=args.model_path,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
    )
    print(f"  Job name:   {job_name}")
    print(f"  Model:      {args.model_name}")
    print(f"  Tasks:      {args.tasks}")
    print(f"  Few-shot:   {args.num_fewshot}")
    print(f"  Namespace:  {args.namespace}")

    # Step 2: Create the resource
    print(f"\n--- Step 2: Creating LMEvalJob ---")
    create_lmevaljob(custom_api, manifest)
    print(f"  LMEvalJob '{job_name}' created successfully.")

    if args.no_wait:
        print(f"\n  --no-wait specified. Monitor with:")
        print(f"    kubectl get lmevaljob {job_name} -n {args.namespace} -o yaml")
        return

    # Step 3: Wait for completion
    print(f"\n--- Step 3: Monitoring job status ---")
    completed_job = wait_for_completion(
        custom_api, job_name, args.namespace, timeout=args.timeout
    )

    # Step 4: Display results
    print(f"\n--- Step 4: Retrieving results ---")
    results = extract_results(completed_job)
    print_results(results, args.model_name)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
