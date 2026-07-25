"""Deploy a fine-tuned model on RHOAI 3.4 via KServe RawDeployment.

Creates an InferenceService that loads a model from S3 or PVC storage and
serves it using the vLLM runtime with GPU acceleration.  Supports both
RawDeployment (default in RHOAI 3.4) and Knative serving modes.

After the endpoint becomes ready the script sends a test inference request
to verify the deployment is serving correctly.

Requirements:
    pip install kubernetes requests

Usage:
    python deploy_finetuned_model.py --model-path s3://my-bucket/models/granite-8b-finetuned
    python deploy_finetuned_model.py --model-path /mnt/models/granite-8b --storage-type pvc
    python deploy_finetuned_model.py --model-path s3://bucket/model --mode knative --gpu-count 4

Environment variables:
    KUBECONFIG          Path to kubeconfig file (default: ~/.kube/config)
    INFERENCE_NS        Target namespace (overridden by --namespace)
    S3_ACCESS_KEY_ID    S3 credential for model storage
    S3_SECRET_ACCESS_KEY  S3 credential for model storage
    S3_ENDPOINT_URL     S3 endpoint (e.g. https://s3.amazonaws.com)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException


_DEFAULT_NAMESPACE = "rhoai-serving"
_DEFAULT_RUNTIME = "vllm-runtime"
_READY_TIMEOUT = 600
_POLL_INTERVAL = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a fine-tuned model via KServe RawDeployment on RHOAI 3.4.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="S3 URI (s3://bucket/path) or PVC mount path for the model artifacts",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="InferenceService name (default: derived from model path)",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help=f"Kubernetes namespace (default: $INFERENCE_NS or {_DEFAULT_NAMESPACE})",
    )
    parser.add_argument(
        "--runtime",
        default=_DEFAULT_RUNTIME,
        help=f"ServingRuntime name (default: {_DEFAULT_RUNTIME})",
    )
    parser.add_argument(
        "--mode",
        choices=["rawdeployment", "knative"],
        default="rawdeployment",
        help="Serving mode (default: rawdeployment)",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="Number of GPUs to allocate (default: 1)",
    )
    parser.add_argument(
        "--gpu-type",
        default="nvidia.com/gpu",
        help="GPU resource name (default: nvidia.com/gpu)",
    )
    parser.add_argument(
        "--memory",
        default="48Gi",
        help="Memory request/limit (default: 48Gi)",
    )
    parser.add_argument(
        "--cpu",
        default="8",
        help="CPU request/limit (default: 8)",
    )
    parser.add_argument(
        "--storage-type",
        choices=["s3", "pvc"],
        default="s3",
        help="Model storage backend (default: s3)",
    )
    parser.add_argument(
        "--storage-secret",
        default="aws-connection-models",
        help="Secret name for S3 credentials (default: aws-connection-models)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length for vLLM (default: 4096)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=_READY_TIMEOUT,
        help=f"Seconds to wait for endpoint readiness (default: {_READY_TIMEOUT})",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip the inference verification request",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the manifest without applying it",
    )
    return parser.parse_args()


def derive_service_name(model_path: str) -> str:
    """Generate a valid K8s name from the model path."""
    base = model_path.rstrip("/").split("/")[-1]
    name = base.lower().replace("_", "-").replace(".", "-")
    return name[:63]


def build_inferenceservice(args: argparse.Namespace, namespace: str) -> dict:
    """Construct the InferenceService manifest."""
    name = args.name or derive_service_name(args.model_path)

    annotations = {
        "serving.kserve.io/deploymentMode": (
            "RawDeployment" if args.mode == "rawdeployment" else "Knative"
        ),
    }

    storage_uri = args.model_path
    if args.storage_type == "s3":
        annotations["serving.kserve.io/secretName"] = args.storage_secret

    container_env = [
        {"name": "VLLM_MAX_MODEL_LEN", "value": str(args.max_model_len)},
    ]
    if args.gpu_count > 1:
        container_env.append(
            {"name": "VLLM_TENSOR_PARALLEL_SIZE", "value": str(args.gpu_count)}
        )

    resources = {
        "requests": {
            "cpu": args.cpu,
            "memory": args.memory,
            args.gpu_type: str(args.gpu_count),
        },
        "limits": {
            "cpu": args.cpu,
            "memory": args.memory,
            args.gpu_type: str(args.gpu_count),
        },
    }

    manifest = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/part-of": "rhoai-serving",
                "app.kubernetes.io/managed-by": "rhoai-examples",
            },
            "annotations": annotations,
        },
        "spec": {
            "predictor": {
                "model": {
                    "modelFormat": {"name": "vLLM"},
                    "runtime": args.runtime,
                    "storageUri": storage_uri,
                    "resources": resources,
                    "env": container_env,
                },
            },
        },
    }

    if args.storage_type == "pvc":
        manifest["spec"]["predictor"]["model"]["storageUri"] = f"pvc://{storage_uri}"

    return manifest


def apply_inferenceservice(manifest: dict, namespace: str) -> None:
    """Create or update the InferenceService via the Kubernetes API."""
    api = client.CustomObjectsApi()
    name = manifest["metadata"]["name"]

    try:
        api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=name,
        )
        api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=name,
            body=manifest,
        )
        print(f"  Updated existing InferenceService '{name}'")
    except ApiException as e:
        if e.status == 404:
            api.create_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                body=manifest,
            )
            print(f"  Created InferenceService '{name}'")
        else:
            raise


def wait_for_ready(name: str, namespace: str, timeout: int) -> str:
    """Poll until the InferenceService reports a ready URL."""
    api = client.CustomObjectsApi()
    deadline = time.time() + timeout
    print(f"  Waiting up to {timeout}s for endpoint to become ready...")

    while time.time() < deadline:
        obj = api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=name,
        )
        conditions = obj.get("status", {}).get("conditions", [])
        for cond in conditions:
            if cond.get("type") == "Ready" and cond.get("status") == "True":
                url = obj["status"].get("url", "")
                print(f"  Endpoint ready: {url}")
                return url

        time.sleep(_POLL_INTERVAL)

    print("  ERROR: Timed out waiting for endpoint readiness", file=sys.stderr)
    sys.exit(1)


def test_inference(url: str) -> None:
    """Send a test chat completion to verify the deployment."""
    endpoint = f"{url}/v1/chat/completions"
    payload = {
        "model": "default",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "max_tokens": 64,
        "temperature": 0.1,
    }

    print(f"  Sending test request to {endpoint}")
    start = time.time()
    resp = requests.post(endpoint, json=payload, timeout=60)
    latency_ms = (time.time() - start) * 1000

    if resp.status_code != 200:
        print(f"  ERROR: Received status {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("completion_tokens", "?")
    print(f"  Response ({latency_ms:.0f}ms, {tokens} tokens): {content[:120]}")


def main() -> None:
    args = parse_args()
    namespace = args.namespace or os.getenv("INFERENCE_NS", _DEFAULT_NAMESPACE)

    print("=" * 60)
    print("Deploy Fine-Tuned Model — KServe RawDeployment")
    print("=" * 60)
    print(f"  Model path:         {args.model_path}")
    print(f"  Namespace:          {namespace}")
    print(f"  Serving mode:       {args.mode}")
    print(f"  Runtime:            {args.runtime}")
    print(f"  GPUs:               {args.gpu_count} x {args.gpu_type}")
    print(f"  Memory:             {args.memory}")
    print(f"  Storage type:       {args.storage_type}")
    print(f"  Max model length:   {args.max_model_len}")
    print("=" * 60)

    manifest = build_inferenceservice(args, namespace)

    if args.dry_run:
        import json

        print("\nGenerated manifest (dry-run):")
        print(json.dumps(manifest, indent=2))
        return

    config.load_kube_config()
    apply_inferenceservice(manifest, namespace)

    name = manifest["metadata"]["name"]
    url = wait_for_ready(name, namespace, args.timeout)

    if not args.skip_test:
        test_inference(url)

    print(f"\nDeployment complete: {name}")


if __name__ == "__main__":
    main()
