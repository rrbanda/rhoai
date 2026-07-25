"""Submit and monitor Kubeflow TrainJob resources for distributed training.

Creates TrainJob custom resources on an RHOAI 3.4 cluster to run
Training Hub algorithms (SFT, OSFT) across multiple nodes and GPUs.
The script handles resource creation, status polling, log streaming,
and final metrics retrieval.

Requirements:
    pip install kubernetes

Usage:
    python submit_trainjob.py --algorithm sft
    python submit_trainjob.py --algorithm osft --unfreeze-ratio 0.3 --use-liger
    python submit_trainjob.py --algorithm sft --num-workers 4 --gpus-per-worker 8
    python submit_trainjob.py --algorithm sft --model-path /mnt/storage/models/Llama-3.1-70B
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from kubernetes import client, config, watch


API_GROUP = "kubeflow.org"
API_VERSION = "v2alpha1"
PLURAL = "trainjobs"
TRAINING_IMAGE = "quay.io/modh/training-hub:latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a distributed TrainJob to an RHOAI 3.4 cluster.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["sft", "osft"],
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
        help="Kubernetes namespace for the TrainJob",
    )
    parser.add_argument(
        "--job-name",
        help="TrainJob name (default: <algorithm>-distributed-<timestamp>)",
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
        default=2,
        help="Number of worker nodes",
    )
    parser.add_argument(
        "--gpus-per-worker",
        type=int,
        default=4,
        help="GPUs per worker node",
    )
    parser.add_argument(
        "--cpu-per-worker",
        type=int,
        default=16,
        help="CPU cores requested per worker",
    )
    parser.add_argument(
        "--memory-per-worker",
        default="128Gi",
        help="Memory requested per worker",
    )

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Effective batch size across all GPUs",
    )
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length in tokens",
    )

    # OSFT-specific
    parser.add_argument(
        "--unfreeze-ratio",
        type=float,
        default=0.3,
        help="OSFT: fraction of weight-matrix directions to unfreeze",
    )
    parser.add_argument(
        "--use-liger",
        action="store_true",
        help="OSFT: enable Liger fused kernels for lower peak memory",
    )

    # Monitoring
    parser.add_argument(
        "--wait",
        action="store_true",
        default=True,
        help="Wait for job completion and stream logs (default: True)",
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


def build_trainjob_manifest(args: argparse.Namespace) -> dict:
    """Construct the TrainJob custom resource dict."""
    job_name = args.job_name or f"{args.algorithm}-distributed-{int(time.time())}"
    ckpt_dir = f"{args.checkpoint_dir}/{args.algorithm}"

    trainer_args = [
        f"--nnodes={args.num_workers}",
        f"--nproc-per-node={args.gpus_per_worker}",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=$(MASTER_ADDR):$(MASTER_PORT)",
        "-m", "training_hub.run",
        args.algorithm,
        f"--model-path=$(MODEL_PATH)",
        f"--data-path=$(DATA_PATH)",
        f"--ckpt-output-dir=$(CHECKPOINT_DIR)",
        f"--num-epochs={args.epochs}",
        f"--effective-batch-size={args.batch_size}",
        f"--learning-rate={args.lr}",
        f"--max-seq-len={args.max_seq_len}",
        "--checkpoint-at-epoch",
    ]

    if args.algorithm == "osft":
        trainer_args.extend([
            f"--unfreeze-rank-ratio={args.unfreeze_ratio}",
            "--unmask-messages",
        ])
        if args.use_liger:
            trainer_args.append("--use-liger")

    return {
        "apiVersion": f"{API_GROUP}/{API_VERSION}",
        "kind": "TrainJob",
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
            "suspend": False,
            "trainer": {
                "image": TRAINING_IMAGE,
                "command": ["torchrun"],
                "args": trainer_args,
                "env": [
                    {"name": "MODEL_PATH", "value": args.model_path},
                    {"name": "DATA_PATH", "value": args.data_path},
                    {"name": "CHECKPOINT_DIR", "value": ckpt_dir},
                    {"name": "NCCL_DEBUG", "value": "INFO"},
                    {"name": "NCCL_IB_DISABLE", "value": "1"},
                    {"name": "TRANSFORMERS_CACHE", "value": "/mnt/storage/cache"},
                    {"name": "HF_HOME", "value": "/mnt/storage/cache"},
                ],
                "numNodes": args.num_workers,
                "resourcesPerNode": {
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
            },
            "podSpecOverrides": {
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
                "tolerations": [
                    {
                        "key": "nvidia.com/gpu",
                        "operator": "Exists",
                        "effect": "NoSchedule",
                    },
                ],
            },
        },
    }


def create_trainjob(crd_api: client.CustomObjectsApi, manifest: dict) -> dict:
    """Submit the TrainJob to the cluster."""
    ns = manifest["metadata"]["namespace"]
    return crd_api.create_namespaced_custom_object(
        group=API_GROUP,
        version=API_VERSION,
        namespace=ns,
        plural=PLURAL,
        body=manifest,
    )


def get_trainjob_status(
    crd_api: client.CustomObjectsApi,
    name: str,
    namespace: str,
) -> dict:
    """Fetch current TrainJob status."""
    obj = crd_api.get_namespaced_custom_object(
        group=API_GROUP,
        version=API_VERSION,
        namespace=namespace,
        plural=PLURAL,
        name=name,
    )
    return obj.get("status", {})


def list_worker_pods(
    core_api: client.CoreV1Api,
    job_name: str,
    namespace: str,
) -> list:
    """List pods belonging to this TrainJob."""
    label_selector = f"app.kubernetes.io/name={job_name}"
    pods = core_api.list_namespaced_pod(namespace, label_selector=label_selector)
    return pods.items


def stream_pod_logs(
    core_api: client.CoreV1Api,
    pod_name: str,
    namespace: str,
    lines: int = 50,
) -> str:
    """Retrieve recent log lines from a pod."""
    try:
        return core_api.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            tail_lines=lines,
        )
    except client.ApiException:
        return "(logs not yet available)"


def wait_for_completion(
    crd_api: client.CustomObjectsApi,
    core_api: client.CoreV1Api,
    name: str,
    namespace: str,
    poll_interval: int,
) -> bool:
    """Poll until the TrainJob reaches a terminal state."""
    print(f"\nMonitoring TrainJob '{name}' in namespace '{namespace}'...")
    print(f"  Poll interval: {poll_interval}s\n")

    while True:
        status = get_trainjob_status(crd_api, name, namespace)
        conditions = status.get("conditions", [])

        phase = "Unknown"
        for cond in conditions:
            if cond.get("type") == "Complete" and cond.get("status") == "True":
                phase = "Complete"
                break
            if cond.get("type") == "Failed" and cond.get("status") == "True":
                phase = "Failed"
                break
            if cond.get("type") == "Running" and cond.get("status") == "True":
                phase = "Running"

        pods = list_worker_pods(core_api, name, namespace)
        running = sum(1 for p in pods if p.status.phase == "Running")
        total = len(pods)

        print(f"  [{time.strftime('%H:%M:%S')}] Phase: {phase} | Pods: {running}/{total} running")

        if phase == "Complete":
            print("\nTrainJob completed successfully.")
            return True
        if phase == "Failed":
            reason = next(
                (c.get("message", "") for c in conditions if c.get("type") == "Failed"),
                "unknown",
            )
            print(f"\nTrainJob failed: {reason}", file=sys.stderr)
            return False

        time.sleep(poll_interval)


def print_summary(manifest: dict, response: dict) -> None:
    """Print job submission summary."""
    meta = manifest["metadata"]
    spec = manifest["spec"]["trainer"]
    total_gpus = spec["numNodes"] * int(spec["resourcesPerNode"]["requests"]["nvidia.com/gpu"])

    print("=" * 60)
    print("TrainJob Submitted")
    print("=" * 60)
    print(f"  Name:           {meta['name']}")
    print(f"  Namespace:      {meta['namespace']}")
    print(f"  Algorithm:      {meta['labels']['training-hub.rhoai.io/algorithm']}")
    print(f"  Worker nodes:   {spec['numNodes']}")
    print(f"  GPUs / worker:  {spec['resourcesPerNode']['requests']['nvidia.com/gpu']}")
    print(f"  Total GPUs:     {total_gpus}")
    print(f"  Image:          {spec['image']}")
    print("=" * 60)


def main() -> None:
    args = parse_args()

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    crd_api = client.CustomObjectsApi()
    core_api = client.CoreV1Api()

    manifest = build_trainjob_manifest(args)

    print(f"\nCreating TrainJob for {args.algorithm.upper()} training...")
    response = create_trainjob(crd_api, manifest)
    print_summary(manifest, response)

    if not args.wait:
        print(f"\nJob submitted. Monitor with:")
        print(f"  oc get trainjob {manifest['metadata']['name']} -n {args.namespace} -w")
        return

    success = wait_for_completion(
        crd_api, core_api,
        manifest["metadata"]["name"],
        args.namespace,
        args.poll_interval,
    )

    pods = list_worker_pods(core_api, manifest["metadata"]["name"], args.namespace)
    if pods:
        print(f"\nLogs from worker-0 (last 20 lines):")
        print("-" * 40)
        logs = stream_pod_logs(core_api, pods[0].metadata.name, args.namespace, lines=20)
        print(logs)

    if not success:
        sys.exit(1)

    ckpt_dir = f"{args.checkpoint_dir}/{args.algorithm}"
    print(f"\nCheckpoints saved to PVC at: {ckpt_dir}")


if __name__ == "__main__":
    main()
