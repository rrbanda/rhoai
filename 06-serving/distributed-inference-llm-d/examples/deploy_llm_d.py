"""Deploy a large model with llm-d distributed inference on RHOAI 3.4.

Creates an LLMInferenceService custom resource that configures distributed
serving with prefix-cache-aware routing and optional disaggregated
prefill/decode phases.  This enables efficient multi-node deployment of
models that exceed single-node GPU memory.

Requirements:
    pip install kubernetes requests

Usage:
    python deploy_llm_d.py --model ibm-granite/granite-3.3-8b-instruct
    python deploy_llm_d.py --model meta-llama/Llama-3.1-70B-Instruct --replicas 4 --gpu-per-replica 4
    python deploy_llm_d.py --model Qwen/Qwen3-235B-A22B --replicas 8 --disaggregated

Environment variables:
    KUBECONFIG          Path to kubeconfig file (default: ~/.kube/config)
    INFERENCE_NS        Target namespace (overridden by --namespace)
    MODEL_REGISTRY      Model registry URL for pulling model artifacts
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
_READY_TIMEOUT = 900
_POLL_INTERVAL = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a model with llm-d distributed inference on RHOAI 3.4.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name or HuggingFace ID (e.g. ibm-granite/granite-3.3-8b-instruct)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="LLMInferenceService name (default: derived from model)",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help=f"Kubernetes namespace (default: $INFERENCE_NS or {_DEFAULT_NAMESPACE})",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=2,
        help="Number of serving replicas (default: 2)",
    )
    parser.add_argument(
        "--gpu-per-replica",
        type=int,
        default=4,
        help="GPUs allocated per replica (default: 4)",
    )
    parser.add_argument(
        "--gpu-type",
        default="nvidia.com/gpu",
        help="GPU resource name (default: nvidia.com/gpu)",
    )
    parser.add_argument(
        "--memory-per-replica",
        default="96Gi",
        help="Memory per replica (default: 96Gi)",
    )
    parser.add_argument(
        "--prefix-caching",
        action="store_true",
        default=True,
        help="Enable prefix-cache-aware routing (default: enabled)",
    )
    parser.add_argument(
        "--no-prefix-caching",
        action="store_true",
        help="Disable prefix-cache-aware routing",
    )
    parser.add_argument(
        "--disaggregated",
        action="store_true",
        help="Enable disaggregated serving (separate prefill/decode phases)",
    )
    parser.add_argument(
        "--prefill-replicas",
        type=int,
        default=1,
        help="Number of prefill-phase replicas when disaggregated (default: 1)",
    )
    parser.add_argument(
        "--decode-replicas",
        type=int,
        default=1,
        help="Number of decode-phase replicas when disaggregated (default: 1)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length (default: 8192)",
    )
    parser.add_argument(
        "--storage-secret",
        default="aws-connection-models",
        help="Secret name for model storage credentials (default: aws-connection-models)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=_READY_TIMEOUT,
        help=f"Seconds to wait for readiness (default: {_READY_TIMEOUT})",
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


def derive_service_name(model: str) -> str:
    """Generate a valid K8s name from the model identifier."""
    base = model.rstrip("/").split("/")[-1]
    name = base.lower().replace("_", "-").replace(".", "-")
    return name[:63]


def build_llminferenceservice(args: argparse.Namespace, namespace: str) -> dict:
    """Construct the LLMInferenceService manifest."""
    name = args.name or derive_service_name(args.model)
    enable_prefix_caching = args.prefix_caching and not args.no_prefix_caching

    manifest = {
        "apiVersion": "inference.llm-d.ai/v1alpha1",
        "kind": "LLMInferenceService",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/part-of": "rhoai-serving",
                "app.kubernetes.io/managed-by": "rhoai-examples",
                "llm-d.ai/model-family": name.split("-")[0],
            },
            "annotations": {
                "llm-d.ai/description": f"Distributed deployment of {args.model}",
            },
        },
        "spec": {
            "modelRef": {
                "name": args.model,
            },
            "storageSecretRef": {
                "name": args.storage_secret,
            },
            "routing": {
                "prefixCaching": {
                    "enabled": enable_prefix_caching,
                },
            },
            "engine": {
                "name": "vllm",
                "args": [
                    f"--max-model-len={args.max_model_len}",
                    f"--tensor-parallel-size={args.gpu_per_replica}",
                    "--enable-prefix-caching" if enable_prefix_caching else "",
                    "--dtype=bfloat16",
                    "--gpu-memory-utilization=0.90",
                ],
                "resources": {
                    "requests": {
                        "cpu": "8",
                        "memory": args.memory_per_replica,
                        args.gpu_type: str(args.gpu_per_replica),
                    },
                    "limits": {
                        "cpu": "8",
                        "memory": args.memory_per_replica,
                        args.gpu_type: str(args.gpu_per_replica),
                    },
                },
            },
            "replicas": args.replicas,
        },
    }

    # Filter empty args
    manifest["spec"]["engine"]["args"] = [
        a for a in manifest["spec"]["engine"]["args"] if a
    ]

    if args.disaggregated:
        manifest["spec"]["disaggregated"] = {
            "enabled": True,
            "prefill": {
                "replicas": args.prefill_replicas,
                "resources": {
                    "requests": {
                        args.gpu_type: str(args.gpu_per_replica),
                        "memory": args.memory_per_replica,
                    },
                    "limits": {
                        args.gpu_type: str(args.gpu_per_replica),
                        "memory": args.memory_per_replica,
                    },
                },
            },
            "decode": {
                "replicas": args.decode_replicas,
                "resources": {
                    "requests": {
                        args.gpu_type: str(args.gpu_per_replica),
                        "memory": args.memory_per_replica,
                    },
                    "limits": {
                        args.gpu_type: str(args.gpu_per_replica),
                        "memory": args.memory_per_replica,
                    },
                },
            },
        }

    return manifest


def apply_llminferenceservice(manifest: dict, namespace: str) -> None:
    """Create or update the LLMInferenceService via the Kubernetes API."""
    api = client.CustomObjectsApi()
    name = manifest["metadata"]["name"]

    try:
        api.get_namespaced_custom_object(
            group="inference.llm-d.ai",
            version="v1alpha1",
            namespace=namespace,
            plural="llminferenceservices",
            name=name,
        )
        api.patch_namespaced_custom_object(
            group="inference.llm-d.ai",
            version="v1alpha1",
            namespace=namespace,
            plural="llminferenceservices",
            name=name,
            body=manifest,
        )
        print(f"  Updated existing LLMInferenceService '{name}'")
    except ApiException as e:
        if e.status == 404:
            api.create_namespaced_custom_object(
                group="inference.llm-d.ai",
                version="v1alpha1",
                namespace=namespace,
                plural="llminferenceservices",
                body=manifest,
            )
            print(f"  Created LLMInferenceService '{name}'")
        else:
            raise


def wait_for_ready(name: str, namespace: str, timeout: int) -> str:
    """Poll until the LLMInferenceService reports ready."""
    api = client.CustomObjectsApi()
    deadline = time.time() + timeout
    print(f"  Waiting up to {timeout}s for distributed endpoint to become ready...")

    while time.time() < deadline:
        obj = api.get_namespaced_custom_object(
            group="inference.llm-d.ai",
            version="v1alpha1",
            namespace=namespace,
            plural="llminferenceservices",
            name=name,
        )
        conditions = obj.get("status", {}).get("conditions", [])
        for cond in conditions:
            if cond.get("type") == "Ready" and cond.get("status") == "True":
                url = obj["status"].get("url", "")
                print(f"  Endpoint ready: {url}")
                return url

        ready_replicas = obj.get("status", {}).get("readyReplicas", 0)
        total_replicas = obj.get("status", {}).get("replicas", "?")
        print(f"    {ready_replicas}/{total_replicas} replicas ready...")
        time.sleep(_POLL_INTERVAL)

    print("  ERROR: Timed out waiting for endpoint readiness", file=sys.stderr)
    sys.exit(1)


def test_inference(url: str, model: str) -> None:
    """Send a test chat completion to verify the distributed deployment."""
    endpoint = f"{url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Explain distributed inference in one sentence."},
        ],
        "max_tokens": 128,
        "temperature": 0.1,
    }

    print(f"  Sending test request to {endpoint}")
    start = time.time()
    resp = requests.post(endpoint, json=payload, timeout=120)
    latency_ms = (time.time() - start) * 1000

    if resp.status_code != 200:
        print(f"  ERROR: Received status {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("completion_tokens", "?")
    print(f"  Response ({latency_ms:.0f}ms, {tokens} tokens): {content[:200]}")


def main() -> None:
    args = parse_args()
    namespace = args.namespace or os.getenv("INFERENCE_NS", _DEFAULT_NAMESPACE)

    print("=" * 60)
    print("Deploy Model — llm-d Distributed Inference")
    print("=" * 60)
    print(f"  Model:              {args.model}")
    print(f"  Namespace:          {namespace}")
    print(f"  Replicas:           {args.replicas}")
    print(f"  GPUs / replica:     {args.gpu_per_replica} x {args.gpu_type}")
    print(f"  Memory / replica:   {args.memory_per_replica}")
    print(f"  Prefix caching:     {args.prefix_caching and not args.no_prefix_caching}")
    print(f"  Disaggregated:      {args.disaggregated}")
    if args.disaggregated:
        print(f"    Prefill replicas: {args.prefill_replicas}")
        print(f"    Decode replicas:  {args.decode_replicas}")
    print(f"  Max model length:   {args.max_model_len}")
    print("=" * 60)

    manifest = build_llminferenceservice(args, namespace)

    if args.dry_run:
        import json

        print("\nGenerated manifest (dry-run):")
        print(json.dumps(manifest, indent=2))
        return

    config.load_kube_config()
    apply_llminferenceservice(manifest, namespace)

    name = manifest["metadata"]["name"]
    url = wait_for_ready(name, namespace, args.timeout)

    if not args.skip_test:
        test_inference(url, args.model)

    print(f"\nDistributed deployment complete: {name}")


if __name__ == "__main__":
    main()
