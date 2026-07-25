"""Step 3: Fine-tune a student model with Training Hub.

Supports both standard Supervised Fine-Tuning (SFT) and
Orthogonal Subspace Fine-Tuning (OSFT) via the --algorithm flag.
OSFT is recommended for knowledge tuning as it preserves the base
model's existing capabilities while absorbing new knowledge.

Usage:
    python 03_model_training.py --algorithm osft
    python 03_model_training.py --algorithm sft --num-epochs 5 --learning-rate 1e-5
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a model with Training Hub (SFT or OSFT)."
    )
    parser.add_argument(
        "--algorithm",
        choices=["sft", "osft"],
        default="osft",
        help="Training algorithm (default: osft)",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to training JSONL (default: $OUTPUT_DATA_FOLDER/training_mix/knowledge_train.jsonl)",
    )
    parser.add_argument(
        "--ckpt-output-dir",
        default=None,
        help="Checkpoint output directory (default: $CHECKPOINT_DIR)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Student model name or path (default: $STUDENT_MODEL)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Number of GPUs per node (default: 1)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate (default: 5e-6)",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=128,
        help="Effective batch size across all GPUs (default: 128)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)",
    )
    return parser.parse_args()


def run_sft(
    model_path: str,
    data_path: str,
    ckpt_output_dir: str,
    num_epochs: int,
    effective_batch_size: int,
    learning_rate: float,
    max_seq_len: int,
    nproc_per_node: int,
) -> None:
    """Run standard Supervised Fine-Tuning."""
    from training_hub import sft

    sft(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=ckpt_output_dir,
        num_epochs=num_epochs,
        effective_batch_size=effective_batch_size,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
        max_tokens_per_gpu=25000,
        nproc_per_node=nproc_per_node,
    )


def run_osft(
    model_path: str,
    data_path: str,
    ckpt_output_dir: str,
    num_epochs: int,
    effective_batch_size: int,
    learning_rate: float,
    max_seq_len: int,
    nproc_per_node: int,
) -> None:
    """Run Orthogonal Subspace Fine-Tuning (preserves base capabilities)."""
    from training_hub import osft

    osft(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=ckpt_output_dir,
        unfreeze_rank_ratio=0.3,
        num_epochs=num_epochs,
        effective_batch_size=effective_batch_size,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
        max_tokens_per_gpu=10000,
        unmask_messages=True,
        use_liger=True,
        nproc_per_node=nproc_per_node,
    )


def main() -> None:
    load_dotenv()
    args = parse_args()

    model_path = args.model_path or os.getenv(
        "STUDENT_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
    )
    default_data_path = os.path.join(
        os.getenv("OUTPUT_DATA_FOLDER", "./generated_output_data"),
        "training_mix",
        "knowledge_train.jsonl",
    )
    data_path = args.data_path or default_data_path
    ckpt_output_dir = args.ckpt_output_dir or os.getenv(
        "CHECKPOINT_DIR", "./checkpoints"
    )

    if not os.path.isfile(data_path):
        print(f"ERROR: Training data not found at {data_path}")
        print("Run 01_data_generation.py and 02_data_mixing.py first.")
        sys.exit(1)

    os.makedirs(ckpt_output_dir, exist_ok=True)

    print(f"Training Configuration")
    print(f"  Algorithm          : {args.algorithm.upper()}")
    print(f"  Model              : {model_path}")
    print(f"  Data               : {data_path}")
    print(f"  Checkpoint dir     : {ckpt_output_dir}")
    print(f"  Epochs             : {args.num_epochs}")
    print(f"  Effective batch    : {args.effective_batch_size}")
    print(f"  Learning rate      : {args.learning_rate}")
    print(f"  Max sequence length: {args.max_seq_len}")
    print(f"  GPUs per node      : {args.nproc_per_node}")
    print()

    train_fn = run_sft if args.algorithm == "sft" else run_osft

    try:
        train_fn(
            model_path=model_path,
            data_path=data_path,
            ckpt_output_dir=ckpt_output_dir,
            num_epochs=args.num_epochs,
            effective_batch_size=args.effective_batch_size,
            learning_rate=args.learning_rate,
            max_seq_len=args.max_seq_len,
            nproc_per_node=args.nproc_per_node,
        )
    except Exception as exc:
        print(f"ERROR: Training failed: {exc}")
        sys.exit(1)

    print(f"\nTraining complete. Checkpoints saved to {ckpt_output_dir}")


if __name__ == "__main__":
    main()
