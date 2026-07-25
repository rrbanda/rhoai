#!/usr/bin/env python3
"""SFT Training with MLflow Experiment Tracking.

Demonstrates Supervised Fine-Tuning (SFT) with MLflow integration using
Training Hub on Red Hat OpenShift AI. MLflow automatically logs training
parameters, metrics (loss, learning rate), and model checkpoints.

Prerequisites:
    - MLflow server deployed via the MLflow Operator (or local instance)
    - Training data in JSONL format (messages-style)
    - GPU node available for training

Example usage:
    # Basic (uses env vars for MLflow URI):
    python sft_with_mlflow.py --data-path ./train_data.jsonl

    # Specify model and MLflow server:
    python sft_with_mlflow.py \\
        --data-path ./train_data.jsonl \\
        --model-path meta-llama/Llama-3.1-8B-Instruct \\
        --mlflow-uri http://mlflow.apps.cluster.example.com:5000

    # Multi-GPU with custom experiment:
    python sft_with_mlflow.py \\
        --data-path ./train_data.jsonl \\
        --nproc-per-node 4 \\
        --mlflow-experiment my-sft-experiment
"""

import argparse
import os
import sys
from datetime import datetime

from training_hub import sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT Training with MLflow Experiment Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    model_group = parser.add_argument_group("Model and Data")
    model_group.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"),
        help="Model path or HuggingFace name (env: MODEL_PATH)",
    )
    model_group.add_argument(
        "--data-path",
        required=True,
        help="Path to training data in JSONL format",
    )
    model_group.add_argument(
        "--ckpt-output-dir",
        default=os.getenv("CKPT_OUTPUT_DIR", "./sft-mlflow-checkpoints"),
        help="Directory for model checkpoints (env: CKPT_OUTPUT_DIR)",
    )

    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument(
        "--num-epochs", type=int, default=1, help="Number of training epochs"
    )
    train_group.add_argument(
        "--effective-batch-size", type=int, default=32, help="Effective batch size"
    )
    train_group.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Learning rate"
    )
    train_group.add_argument(
        "--max-seq-len", type=int, default=4096, help="Maximum sequence length"
    )
    train_group.add_argument(
        "--max-tokens-per-gpu", type=int, default=4096, help="Max tokens per GPU"
    )
    train_group.add_argument(
        "--warmup-steps", type=int, default=10, help="Learning rate warmup steps"
    )
    train_group.add_argument(
        "--nproc-per-node", type=int, default=1, help="Number of GPUs per node"
    )

    mlflow_group = parser.add_argument_group("MLflow Configuration")
    mlflow_group.add_argument(
        "--mlflow-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking URI (env: MLFLOW_TRACKING_URI)",
    )
    mlflow_group.add_argument(
        "--mlflow-experiment",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "sft-training"),
        help="MLflow experiment name (env: MLFLOW_EXPERIMENT_NAME)",
    )
    mlflow_group.add_argument(
        "--mlflow-run-name",
        default=None,
        help="MLflow run name (default: auto-generated with timestamp)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_name = args.mlflow_run_name or f"sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(args.ckpt_output_dir, exist_ok=True)

    print("=" * 60)
    print("SFT Training with MLflow Experiment Tracking")
    print("=" * 60)
    print(f"Model:             {args.model_path}")
    print(f"Data:              {args.data_path}")
    print(f"Output:            {args.ckpt_output_dir}")
    print(f"GPUs:              {args.nproc_per_node}")
    print(f"Epochs:            {args.num_epochs}")
    print(f"Batch Size:        {args.effective_batch_size}")
    print(f"Learning Rate:     {args.learning_rate}")
    print(f"Max Seq Len:       {args.max_seq_len}")
    print("-" * 60)
    print(f"MLflow URI:        {args.mlflow_uri}")
    print(f"MLflow Experiment: {args.mlflow_experiment}")
    print(f"Run Name:          {run_name}")
    print("=" * 60)
    print()

    try:
        sft(
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,
            data_output_dir=args.ckpt_output_dir,
            num_epochs=args.num_epochs,
            effective_batch_size=args.effective_batch_size,
            learning_rate=args.learning_rate,
            max_seq_len=args.max_seq_len,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            warmup_steps=args.warmup_steps,
            save_samples=0,
            checkpoint_at_epoch=True,
            # MLflow integration — Training Hub handles logging internally
            mlflow_tracking_uri=args.mlflow_uri,
            mlflow_experiment_name=args.mlflow_experiment,
            mlflow_run_name=run_name,
            # Distributed training
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
            node_rank=0,
            rdzv_id=100,
            rdzv_endpoint="127.0.0.1:29500",
        )

        print()
        print("=" * 60)
        print("Training completed successfully!")
        print(f"Checkpoints saved to: {args.ckpt_output_dir}")
        print(f"View results in MLflow UI: {args.mlflow_uri}")
        print("=" * 60)

    except Exception as e:  # noqa: BLE001
        print(f"\nTraining failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
