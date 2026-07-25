"""Submit and monitor KubeRay RayJob resources for distributed training.

Creates RayJob custom resources on an RHOAI 3.4 cluster to run
Training Hub algorithms (GRPO, LoRA) across a managed Ray cluster.
The script handles cluster provisioning, job submission, status
polling, Ray dashboard URL retrieval, and result collection.

Requirements:
    pip install kubernetes

Usage:
    python submit_rayjob.py --algorithm grpo
    python submit_rayjob.py --algorithm lora --num-workers 4 --gpus-per-worker 4
    python submit_rayjob.py --algorithm grpo --backend verl --group-size 16
    python submit_rayjob.py --algorithm lora --model-path /mnt/storage/models/Llama-3.1-70B
"""

from __future__ import annotations

import argparse
import sys
import time

from kubernetes import client, config


API_GROUP = "ray.io"
API_VERSION = "v1"
PLURAL = "rayjobs"
TRAINING_IMAGE = "quay.io/modh/training-hub:latest-ray"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a distributed RayJob to an RHOAI 3.4 cluster.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["grpo", "lora"],
        required=True,
        help="Training algorithm to run",
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/storage/models/meta-llama/Llama-3.1-8B-Instruct",
        help="Path to model weights inside the PVC",
    )
    parser.add_argument(
        "--data-path",
        default="/mnt/storage/data/training_data.jsonl",
        help="Path to training data inside the PVC",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/mnt/storage/checkpoints",
        help="PVC directory for model checkpoints",
    )
    parser.add_argument(
        "--namespace",
        default="distributed-training",
        help="Kubernetes namespace for the RayJob",
    )
    parser.add_argument(
        "--job-name",
        help="RayJob name (default: <algorithm>-ray-<timestamp>)",
    )
    parser.add_argument(
        "--pvc-name",
        default="training-storage",
        help="PVC containing model weights and training data",
    )

    # Cluster shape
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of Ray GPU worker pods",
    )
    parser.add_argument(
        "--gpus-per-worker",
        type=int,
        default=2,
        help="GPUs per worker pod",
    )
    parser.add_argument(
        "--cpu-per-worker",
        type=int,
        default=8,
        help="CPU cores requested per worker",
    )
    parser.add_argument(
        "--memory-per-worker",
        default="64Gi",
        help="Memory requested per worker",
    )
    parser.add_argument(
        "--head-cpu",
        type=int,
        default=8,
        help="CPU cores for the Ray head node",
    )
    parser.add_argument(
        "--head-memory",
        default="32Gi",
        help="Memory for the Ray head node",
    )

    # GRPO-specific
    parser.add_argument(
        "--backend",
        choices=["art", "verl"],
        default="verl",
        help="GRPO backend: 'art' or 'verl'",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha scaling factor",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="GRPO: number of outer iterations",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="GRPO: rollout candidates per prompt",
    )
    parser.add_argument(
        "--prompt-batch",
        type=int,
        default=100,
        help="GRPO: prompts sampled per iteration",
    )

    # LoRA-specific
    parser.add_argument("--epochs", type=int, default=3, help="LoRA: number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length in tokens",
    )

    # Monitoring
    parser.add_argument(
        "--wait",
        action="store_true",
        default=True,
        help="Wait for job completion (default: True)",
    )
    parser.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="Submit and exit without waiting",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between status polls",
    )
    return parser.parse_args()


def build_entrypoint(args: argparse.Namespace) -> str:
    """Build the Ray job entrypoint command."""
    if args.algorithm == "grpo":
        return (
            f"python -m training_hub.run lora_grpo"
            f" --model-path {args.model_path}"
            f" --data-path {args.data_path}"
            f" --ckpt-output-dir {args.checkpoint_dir}/grpo"
            f" --backend {args.backend}"
            f" --lora-r {args.lora_r}"
            f" --lora-alpha {args.lora_alpha}"
            f" --num-iterations {args.iterations}"
            f" --group-size {args.group_size}"
            f" --prompt-batch-size {args.prompt_batch}"
            f" --learning-rate {args.lr}"
        )
    return (
        f"python -m training_hub.run lora_sft"
        f" --model-path {args.model_path}"
        f" --data-path {args.data_path}"
        f" --ckpt-output-dir {args.checkpoint_dir}/lora"
        f" --lora-r {args.lora_r}"
        f" --lora-alpha {args.lora_alpha}"
        f" --num-epochs {args.epochs}"
        f" --learning-rate {args.lr}"
        f" --max-seq-len {args.max_seq_len}"
    )


def build_rayjob_manifest(args: argparse.Namespace) -> dict:
    """Construct the RayJob custom resource dict."""
    job_name = args.job_name or f"{args.algorithm}-ray-{int(time.time())}"

    return {
        "apiVersion": f"{API_GROUP}/{API_VERSION}",
        "kind": "RayJob",
        "metadata": {
            "name": job_name,
            "namespace": args.namespace,
            "labels": {
                "app.kubernetes.io/name": job_name,
                "app.kubernetes.io/component": "training",
                "app.kubernetes.io/part-of": "training-hub",
                "training-hub.rhoai.io/algorithm": args.algorithm,
            },
        },
        "spec": {
            "shutdownAfterJobFinishes": True,
            "ttlSecondsAfterFinished": 3600,
            "activeDeadlineSeconds": 86400,
            "entrypoint": build_entrypoint(args),
            "runtimeEnvYAML": (
                "env_vars:\n"
                "  NCCL_DEBUG: INFO\n"
                "  NCCL_IB_DISABLE: '1'\n"
                "  TRANSFORMERS_CACHE: /mnt/storage/cache\n"
                "  HF_HOME: /mnt/storage/cache\n"
            ),
            "managedClusterConfig": {
                "headGroupSpec": {
                    "rayStartParams": {
                        "dashboard-host": "0.0.0.0",
                        "num-cpus": "0",
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app.kubernetes.io/name": job_name,
                                "ray.io/node-type": "head",
                            },
                        },
                        "spec": {
                            "containers": [{
                                "name": "ray-head",
                                "image": TRAINING_IMAGE,
                                "ports": [
                                    {"containerPort": 6379, "name": "gcs-server"},
                                    {"containerPort": 8265, "name": "dashboard"},
                                    {"containerPort": 10001, "name": "client"},
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": str(args.head_cpu),
                                        "memory": args.head_memory,
                                    },
                                    "limits": {
                                        "cpu": str(args.head_cpu * 2),
                                        "memory": args.head_memory,
                                    },
                                },
                                "volumeMounts": [
                                    {"name": "training-storage", "mountPath": "/mnt/storage"},
                                ],
                            }],
                            "volumes": [{
                                "name": "training-storage",
                                "persistentVolumeClaim": {"claimName": args.pvc_name},
                            }],
                            "tolerations": [{
                                "key": "nvidia.com/gpu",
                                "operator": "Exists",
                                "effect": "NoSchedule",
                            }],
                        },
                    },
                },
                "workerGroupSpecs": [{
                    "groupName": "gpu-workers",
                    "replicas": args.num_workers,
                    "minReplicas": args.num_workers,
                    "maxReplicas": args.num_workers,
                    "rayStartParams": {
                        "num-gpus": str(args.gpus_per_worker),
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app.kubernetes.io/name": job_name,
                                "ray.io/node-type": "worker",
                            },
                        },
                        "spec": {
                            "containers": [{
                                "name": "ray-worker",
                                "image": TRAINING_IMAGE,
                                "resources": {
                                    "requests": {
                                        "cpu": str(args.cpu_per_worker),
                                        "memory": args.memory_per_worker,
                                        "nvidia.com/gpu": str(args.gpus_per_worker),
                                    },
                                    "limits": {
                                        "cpu": str(args.cpu_per_worker * 2),
                                        "memory": args.memory_per_worker,
                                        "nvidia.com/gpu": str(args.gpus_per_worker),
                                    },
                                },
                                "volumeMounts": [
                                    {"name": "training-storage", "mountPath": "/mnt/storage"},
                                    {"name": "dshm", "mountPath": "/dev/shm"},
                                ],
                            }],
                            "volumes": [
                                {
                                    "name": "training-storage",
                                    "persistentVolumeClaim": {"claimName": args.pvc_name},
                                },
                                {
                                    "name": "dshm",
                                    "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"},
                                },
                            ],
                            "tolerations": [{
                                "key": "nvidia.com/gpu",
                                "operator": "Exists",
                                "effect": "NoSchedule",
                            }],
                        },
                    },
                }],
            },
        },
    }


def create_rayjob(crd_api: client.CustomObjectsApi, manifest: dict) -> dict:
    """Submit the RayJob to the cluster."""
    ns = manifest["metadata"]["namespace"]
    return crd_api.create_namespaced_custom_object(
        group=API_GROUP,
        version=API_VERSION,
        namespace=ns,
        plural=PLURAL,
        body=manifest,
    )


def get_rayjob_status(
    crd_api: client.CustomObjectsApi,
    name: str,
    namespace: str,
) -> dict:
    """Fetch current RayJob status."""
    obj = crd_api.get_namespaced_custom_object(
        group=API_GROUP,
        version=API_VERSION,
        namespace=namespace,
        plural=PLURAL,
        name=name,
    )
    return obj.get("status", {})


def get_dashboard_url(status: dict) -> str | None:
    """Extract Ray dashboard URL from job status."""
    return status.get("dashboardURL")


def wait_for_completion(
    crd_api: client.CustomObjectsApi,
    name: str,
    namespace: str,
    poll_interval: int,
) -> bool:
    """Poll until the RayJob reaches a terminal state."""
    print(f"\nMonitoring RayJob '{name}' in namespace '{namespace}'...")
    print(f"  Poll interval: {poll_interval}s\n")

    dashboard_shown = False

    while True:
        status = get_rayjob_status(crd_api, name, namespace)
        job_status = status.get("jobStatus", "PENDING")
        deployment_status = status.get("jobDeploymentStatus", "Initializing")

        if not dashboard_shown:
            dashboard_url = get_dashboard_url(status)
            if dashboard_url:
                print(f"  Ray Dashboard: {dashboard_url}")
                dashboard_shown = True

        print(
            f"  [{time.strftime('%H:%M:%S')}] "
            f"Job: {job_status} | Cluster: {deployment_status}"
        )

        if job_status == "SUCCEEDED":
            print("\nRayJob completed successfully.")
            return True
        if job_status in ("FAILED", "STOPPED"):
            message = status.get("message", "unknown error")
            print(f"\nRayJob {job_status.lower()}: {message}", file=sys.stderr)
            return False

        time.sleep(poll_interval)


def print_summary(manifest: dict) -> None:
    """Print job submission summary."""
    meta = manifest["metadata"]
    spec = manifest["spec"]
    cluster = spec["managedClusterConfig"]
    workers = cluster["workerGroupSpecs"][0]
    gpus_per_worker = int(workers["rayStartParams"]["num-gpus"])
    num_workers = workers["replicas"]
    total_gpus = num_workers * gpus_per_worker

    print("=" * 60)
    print("RayJob Submitted")
    print("=" * 60)
    print(f"  Name:           {meta['name']}")
    print(f"  Namespace:      {meta['namespace']}")
    print(f"  Algorithm:      {meta['labels']['training-hub.rhoai.io/algorithm']}")
    print(f"  Worker pods:    {num_workers}")
    print(f"  GPUs / worker:  {gpus_per_worker}")
    print(f"  Total GPUs:     {total_gpus}")
    print(f"  Image:          {TRAINING_IMAGE}")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    crd_api = client.CustomObjectsApi()

    manifest = build_rayjob_manifest(args)

    print(f"\nCreating RayJob for {args.algorithm.upper()} training...")
    create_rayjob(crd_api, manifest)
    print_summary(manifest)

    if not args.wait:
        print(f"\nJob submitted. Monitor with:")
        print(f"  oc get rayjob {manifest['metadata']['name']} -n {args.namespace} -w")
        return

    success = wait_for_completion(
        crd_api,
        manifest["metadata"]["name"],
        args.namespace,
        args.poll_interval,
    )

    status = get_rayjob_status(
        crd_api, manifest["metadata"]["name"], args.namespace,
    )
    dashboard_url = get_dashboard_url(status)
    if dashboard_url:
        print(f"\nRay Dashboard: {dashboard_url}")

    if not success:
        sys.exit(1)

    ckpt_dir = f"{args.checkpoint_dir}/{args.algorithm}"
    print(f"\nCheckpoints saved to PVC at: {ckpt_dir}")


if __name__ == "__main__":
    main()
