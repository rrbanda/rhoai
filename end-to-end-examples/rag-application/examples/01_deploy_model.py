"""Step 1: Deploy an LLM and Embedding Model for RAG on RHOAI.

Creates KServe InferenceService resources for a vLLM-served LLM
(Granite) and an embedding model (nomic-embed-text), then waits for
both to reach a ready state.  Uses the Kubernetes Python client to
apply custom resources directly.

Usage:
    cp .env.example .env   # fill in real values
    python 01_deploy_model.py [--namespace my-project]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv


VLLM_RUNTIME = "vllm-runtime"

LLM_ISVC_TEMPLATE = {
    "apiVersion": "serving.kserve.io/v1beta1",
    "kind": "InferenceService",
    "metadata": {},
    "spec": {
        "predictor": {
            "model": {
                "modelFormat": {"name": "vLLM"},
                "runtime": VLLM_RUNTIME,
                "storageUri": "",
            },
            "resources": {
                "limits": {"nvidia.com/gpu": "1"},
                "requests": {"cpu": "4", "memory": "16Gi"},
            },
        }
    },
}

EMBEDDING_ISVC_TEMPLATE = {
    "apiVersion": "serving.kserve.io/v1beta1",
    "kind": "InferenceService",
    "metadata": {},
    "spec": {
        "predictor": {
            "model": {
                "modelFormat": {"name": "vLLM"},
                "runtime": VLLM_RUNTIME,
                "storageUri": "",
                "args": ["--task", "embedding"],
            },
            "resources": {
                "limits": {"nvidia.com/gpu": "1"},
                "requests": {"cpu": "2", "memory": "8Gi"},
            },
        }
    },
}


def get_k8s_client():
    """Load Kubernetes config and return the CustomObjects API client."""
    from kubernetes import client, config

    try:
        config.load_incluster_config()
        print("Using in-cluster Kubernetes config")
    except config.ConfigException:
        config.load_kube_config()
        print("Using local kubeconfig")

    return client.CustomObjectsApi()


def create_inference_service(
    api: "kubernetes.client.CustomObjectsApi",
    namespace: str,
    name: str,
    template: dict,
    storage_uri: str,
) -> dict:
    """Create or update a KServe InferenceService."""
    import copy

    body = copy.deepcopy(template)
    body["metadata"]["name"] = name
    body["metadata"]["namespace"] = namespace
    body["spec"]["predictor"]["model"]["storageUri"] = storage_uri

    group = "serving.kserve.io"
    version = "v1beta1"
    plural = "inferenceservices"

    try:
        existing = api.get_namespaced_custom_object(
            group, version, namespace, plural, name
        )
        body["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
        result = api.replace_namespaced_custom_object(
            group, version, namespace, plural, name, body
        )
        print(f"  Updated InferenceService: {name}")
    except Exception:
        result = api.create_namespaced_custom_object(
            group, version, namespace, plural, body
        )
        print(f"  Created InferenceService: {name}")

    return result


def wait_for_ready(
    api: "kubernetes.client.CustomObjectsApi",
    namespace: str,
    name: str,
    timeout: int = 600,
    poll_interval: int = 15,
) -> bool:
    """Poll an InferenceService until it is ready or timeout is reached."""
    print(f"  Waiting for {name} to be ready (timeout {timeout}s) ...")

    group = "serving.kserve.io"
    version = "v1beta1"
    plural = "inferenceservices"

    start = time.time()
    while time.time() - start < timeout:
        try:
            obj = api.get_namespaced_custom_object(
                group, version, namespace, plural, name
            )
            conditions = obj.get("status", {}).get("conditions", [])
            ready = any(
                c.get("type") == "Ready" and c.get("status") == "True"
                for c in conditions
            )
            if ready:
                url = (
                    obj.get("status", {})
                    .get("url", "(URL not yet assigned)")
                )
                elapsed = time.time() - start
                print(f"  {name} is READY ({elapsed:.0f}s)")
                print(f"    URL: {url}")
                return True
        except Exception as exc:
            print(f"  WARNING: Error checking status for {name}: {exc}")

        time.sleep(poll_interval)

    print(f"  ERROR: {name} did not become ready within {timeout}s")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy LLM and embedding model on RHOAI via KServe."
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="OpenShift namespace / project (default: $NAMESPACE)",
    )
    parser.add_argument(
        "--llm-name",
        default="granite-llm",
        help="InferenceService name for the LLM (default: granite-llm)",
    )
    parser.add_argument(
        "--llm-model-uri",
        default=None,
        help="Storage URI for the LLM model (default: $LLM_MODEL_URI)",
    )
    parser.add_argument(
        "--embedding-name",
        default="nomic-embed",
        help="InferenceService name for embeddings (default: nomic-embed)",
    )
    parser.add_argument(
        "--embedding-model-uri",
        default=None,
        help="Storage URI for the embedding model (default: $EMBEDDING_MODEL_URI)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Readiness timeout in seconds per model (default: 600)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    namespace = args.namespace or os.getenv("NAMESPACE")
    llm_uri = args.llm_model_uri or os.getenv(
        "LLM_MODEL_URI",
        "s3://models/ibm-granite/granite-3.3-8b-instruct",
    )
    embedding_uri = args.embedding_model_uri or os.getenv(
        "EMBEDDING_MODEL_URI",
        "s3://models/nomic-ai/nomic-embed-text-v1.5",
    )

    if not namespace:
        print("ERROR: Namespace is required.")
        print("  Set --namespace or $NAMESPACE.")
        sys.exit(1)

    print(f"Deploying models to namespace: {namespace}")
    print(f"  LLM           : {args.llm_name} ({llm_uri})")
    print(f"  Embedding     : {args.embedding_name} ({embedding_uri})")
    print()

    api = get_k8s_client()

    print("Creating LLM InferenceService ...")
    create_inference_service(
        api, namespace, args.llm_name, LLM_ISVC_TEMPLATE, llm_uri
    )

    print("Creating Embedding InferenceService ...")
    create_inference_service(
        api, namespace, args.embedding_name, EMBEDDING_ISVC_TEMPLATE, embedding_uri
    )

    print(f"\n{'=' * 60}")
    print("Waiting for models to become ready ...")
    print(f"{'=' * 60}")

    llm_ready = wait_for_ready(api, namespace, args.llm_name, args.timeout)
    embed_ready = wait_for_ready(api, namespace, args.embedding_name, args.timeout)

    print(f"\n{'=' * 60}")
    if llm_ready and embed_ready:
        print("Both models are ready. Proceed to 02_ingest_documents.py.")
    else:
        failed = []
        if not llm_ready:
            failed.append(args.llm_name)
        if not embed_ready:
            failed.append(args.embedding_name)
        print(f"ERROR: The following models failed to become ready: {failed}")
        print("Check the RHOAI dashboard for pod events and logs.")
        sys.exit(1)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
