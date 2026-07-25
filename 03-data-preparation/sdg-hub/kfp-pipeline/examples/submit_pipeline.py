"""Submit the SDG Hub KFP pipeline to Red Hat OpenShift AI.

Compiles the pipeline (or uses a pre-compiled YAML), connects to the
RHOAI Data Science Pipelines endpoint, submits a run with the given
parameters, and optionally monitors its status until completion.

Usage:
    cp .env.example .env   # fill in real values

    # Submit and monitor
    python submit_pipeline.py \
        --endpoint https://ds-pipeline-dspa.apps.cluster.example.com \
        --model-endpoint https://your-model/v1 \
        --flow-variant extractive_summary \
        --document-jsonl s3://my-bucket/docs/input.jsonl

    # Submit without monitoring
    python submit_pipeline.py \
        --endpoint https://ds-pipeline-dspa.apps.cluster.example.com \
        --no-wait
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv


def compile_pipeline(output_path: str) -> str:
    """Compile the pipeline YAML if it does not already exist."""
    if os.path.isfile(output_path):
        print(f"Using existing pipeline YAML: {output_path}")
        return output_path

    from kfp import compiler

    from sdg_kfp_pipeline import sdg_knowledge_pipeline

    compiler.Compiler().compile(
        pipeline_func=sdg_knowledge_pipeline,
        package_path=output_path,
    )
    print(f"Compiled pipeline to {output_path}")
    return output_path


def create_kfp_client(endpoint: str, token: str | None) -> "kfp.Client":
    """Create a KFP client configured for the RHOAI endpoint."""
    import kfp

    kwargs: dict = {"host": endpoint}
    if token:
        kwargs["existing_token"] = token

    try:
        client = kfp.Client(**kwargs)
    except Exception as exc:
        print(f"ERROR: Failed to connect to KFP endpoint: {exc}")
        print(f"  Endpoint: {endpoint}")
        print("  Ensure the Data Science Pipelines server is running and")
        print("  your token is valid (oc whoami -t).")
        sys.exit(1)

    print(f"Connected to KFP endpoint: {endpoint}")
    return client


def submit_run(
    client: "kfp.Client",
    pipeline_path: str,
    experiment_name: str,
    run_name: str,
    params: dict,
) -> "kfp.client.RunPipelineResult":
    """Upload the pipeline and create a run."""
    print(f"\nSubmitting pipeline run: {run_name}")
    print(f"  Experiment : {experiment_name}")
    print(f"  Parameters:")
    for key, value in params.items():
        display = "***" if "key" in key.lower() else value
        print(f"    {key}: {display}")

    try:
        run = client.create_run_from_pipeline_package(
            pipeline_file=pipeline_path,
            arguments=params,
            run_name=run_name,
            experiment_name=experiment_name,
        )
    except Exception as exc:
        print(f"ERROR: Failed to submit pipeline run: {exc}")
        sys.exit(1)

    print(f"\nRun submitted successfully.")
    print(f"  Run ID: {run.run_id}")
    return run


def monitor_run(
    client: "kfp.Client",
    run_id: str,
    poll_interval: int = 30,
    timeout: int = 3600,
) -> str:
    """Poll run status until completion or timeout.

    Returns the final status string.
    """
    print(f"\nMonitoring run {run_id} (poll every {poll_interval}s, timeout {timeout}s)")
    print(f"{'=' * 60}")

    start = time.time()
    last_status = ""

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"\nERROR: Run timed out after {timeout}s")
            return "TIMEOUT"

        try:
            run_detail = client.get_run(run_id)
            status = run_detail.state or "Unknown"
        except Exception as exc:
            print(f"  WARNING: Failed to fetch run status: {exc}")
            time.sleep(poll_interval)
            continue

        if status != last_status:
            elapsed_min = elapsed / 60
            print(f"  [{elapsed_min:5.1f}m] Status: {status}")
            last_status = status

        terminal_states = {"SUCCEEDED", "FAILED", "SKIPPED", "ERROR", "CANCELED"}
        if status in terminal_states:
            print(f"\nRun completed with status: {status}")
            return status

        time.sleep(poll_interval)


def retrieve_artifacts(client: "kfp.Client", run_id: str) -> None:
    """Print information about output artifacts from the completed run."""
    print(f"\nRetrieving artifacts for run {run_id} ...")
    try:
        run_detail = client.get_run(run_id)
        print(f"  Run name   : {run_detail.display_name}")
        print(f"  Status     : {run_detail.state}")
        print(f"  Created    : {run_detail.created_at}")
        print(f"  Finished   : {run_detail.finished_at}")
    except Exception as exc:
        print(f"  WARNING: Could not retrieve run details: {exc}")

    print("\nTo download output artifacts, use the RHOAI dashboard or:")
    print(f"  kfp run get {run_id} --output-artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit the SDG Hub KFP pipeline to RHOAI."
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="KFP / Data Science Pipelines endpoint URL (default: $KFP_ENDPOINT)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="OpenShift bearer token (default: $KFP_TOKEN or 'oc whoami -t')",
    )
    parser.add_argument(
        "--pipeline-yaml",
        default="sdg_knowledge_pipeline.yaml",
        help="Path to compiled pipeline YAML (default: sdg_knowledge_pipeline.yaml)",
    )
    parser.add_argument(
        "--experiment-name",
        default="sdg-hub-knowledge",
        help="KFP experiment name (default: sdg-hub-knowledge)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="KFP run name (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--model-endpoint",
        default=None,
        help="Teacher model endpoint URL (default: $MODEL_API_BASE)",
    )
    parser.add_argument(
        "--model-api-key",
        default=None,
        help="Teacher model API key (default: $MODEL_API_KEY)",
    )
    parser.add_argument(
        "--flow-variant",
        default="extractive_summary",
        choices=[
            "extractive_summary",
            "detailed_summary",
            "key_facts",
            "doc_direct_qa",
        ],
        help="SDG Hub flow variant to run (default: extractive_summary)",
    )
    parser.add_argument(
        "--document-jsonl",
        default=None,
        help="URI to input document JSONL (default: $DOCUMENT_JSONL_URI)",
    )
    parser.add_argument(
        "--output-path",
        default="/data/sdg-output/generated_data.jsonl",
        help="Output path on PVC or S3 key (default: /data/sdg-output/generated_data.jsonl)",
    )
    parser.add_argument(
        "--s3-endpoint",
        default=None,
        help="S3-compatible endpoint for upload (default: $S3_ENDPOINT)",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="S3 bucket name (default: $S3_BUCKET)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Status poll interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Maximum wait time in seconds (default: 3600)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit the run and exit without monitoring",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    endpoint = args.endpoint or os.getenv("KFP_ENDPOINT")
    token = args.token or os.getenv("KFP_TOKEN")
    model_endpoint = args.model_endpoint or os.getenv("MODEL_API_BASE")
    model_api_key = args.model_api_key or os.getenv("MODEL_API_KEY", "")
    document_jsonl = args.document_jsonl or os.getenv("DOCUMENT_JSONL_URI")
    s3_endpoint = args.s3_endpoint or os.getenv("S3_ENDPOINT", "")
    s3_bucket = args.s3_bucket or os.getenv("S3_BUCKET", "")
    s3_access_key = os.getenv("S3_ACCESS_KEY", "")
    s3_secret_key = os.getenv("S3_SECRET_KEY", "")

    if not endpoint:
        print("ERROR: KFP endpoint is required.")
        print("  Set --endpoint or $KFP_ENDPOINT.")
        sys.exit(1)

    if not model_endpoint:
        print("ERROR: Model endpoint is required.")
        print("  Set --model-endpoint or $MODEL_API_BASE.")
        sys.exit(1)

    if not document_jsonl:
        print("ERROR: Document JSONL URI is required.")
        print("  Set --document-jsonl or $DOCUMENT_JSONL_URI.")
        sys.exit(1)

    pipeline_path = compile_pipeline(args.pipeline_yaml)

    run_name = args.run_name
    if not run_name:
        import datetime

        ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_name = f"sdg-{args.flow_variant}-{ts}"

    params = {
        "model_endpoint": model_endpoint,
        "model_api_key": model_api_key,
        "flow_variant": args.flow_variant,
        "document_jsonl": document_jsonl,
        "output_path": args.output_path,
        "min_rows": 10,
        "s3_endpoint": s3_endpoint,
        "s3_bucket": s3_bucket,
        "s3_access_key": s3_access_key,
        "s3_secret_key": s3_secret_key,
    }

    client = create_kfp_client(endpoint, token)
    run = submit_run(client, pipeline_path, args.experiment_name, run_name, params)

    if args.no_wait:
        print("\n--no-wait specified. Exiting.")
        print(f"Monitor the run in the RHOAI dashboard or with:")
        print(f"  kfp run get {run.run_id}")
        return

    status = monitor_run(client, run.run_id, args.poll_interval, args.timeout)

    if status == "SUCCEEDED":
        retrieve_artifacts(client, run.run_id)
    elif status == "TIMEOUT":
        print("The run is still in progress. Check the RHOAI dashboard.")
        sys.exit(1)
    else:
        print(f"Run ended with status: {status}")
        sys.exit(1)


if __name__ == "__main__":
    main()
