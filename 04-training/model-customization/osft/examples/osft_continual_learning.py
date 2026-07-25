#!/usr/bin/env python3
"""OSFT continual learning -- add new capabilities without catastrophic forgetting.

Demonstrates Orthogonal Subspace Fine-Tuning (OSFT) for continual learning
with Llama 3 8B.  OSFT constrains weight updates to a subspace orthogonal to
the model's critical knowledge directions, so new skills (e.g. structured JSON
output) are absorbed without degrading existing abilities.

Adapted from the Training Hub ``osft_continual_learning`` example.

Hardware requirements:
    8x A100 40 GB (or equivalent).  Reduce ``--max-tokens-per-gpu`` and
    ``--nproc-per-node`` for smaller setups.

Usage:
    python osft_continual_learning.py \\
        --data-path /path/to/json_training_data.jsonl \\
        --ckpt-output-dir /path/to/checkpoints

    python osft_continual_learning.py \\
        --data-path data.jsonl \\
        --ckpt-output-dir ./ckpts \\
        --unfreeze-rank-ratio 0.35 \\
        --nproc-per-node 4
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

from training_hub import osft


def find_most_recent_checkpoint(output_dir: str) -> str:
    """Return the path to the most recently created checkpoint under *output_dir*.

    Raises ``ValueError`` when no checkpoints exist.
    """
    pattern = os.path.join(output_dir, "hf_format", "samples_*.0")
    checkpoint_dirs = glob.glob(pattern)
    if not checkpoint_dirs:
        raise ValueError(
            f"No checkpoints found in {os.path.join(output_dir, 'hf_format')}"
        )
    return max(checkpoint_dirs, key=os.path.getctime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OSFT continual learning: add new capabilities without forgetting.",
    )

    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to training data in JSONL format",
    )
    parser.add_argument(
        "--ckpt-output-dir",
        required=True,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--model-path",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model ID or local path (default: Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--unfreeze-rank-ratio",
        type=float,
        default=0.28,
        help="Fraction of singular-value directions to unfreeze, 0.0-1.0 (default: 0.28)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Effective batch size across all GPUs (default: 128)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length in tokens (default: 2048)",
    )
    parser.add_argument(
        "--max-tokens-per-gpu",
        type=int,
        default=8192,
        help="Token budget per GPU per step (default: 8192)",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=8,
        help="Number of GPUs (default: 8)",
    )
    parser.add_argument(
        "--data-output-dir",
        default="/dev/shm",
        help="Directory for processed data (default: /dev/shm)",
    )

    # Distributed training
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes (default: 1)")
    parser.add_argument("--node-rank", type=int, default=0, help="Node rank (default: 0)")
    parser.add_argument("--rdzv-id", type=int, default=100, help="Rendezvous ID (default: 100)")
    parser.add_argument(
        "--rdzv-endpoint",
        default="127.0.0.1:29500",
        help="Rendezvous endpoint (default: 127.0.0.1:29500)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("OSFT Continual Learning")
    print("=" * 60)
    print(f"  Model:              {args.model_path}")
    print(f"  Data:               {args.data_path}")
    print(f"  Output:             {args.ckpt_output_dir}")
    print(f"  Unfreeze ratio:     {args.unfreeze_rank_ratio}")
    print(f"  Epochs:             {args.num_epochs}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Learning rate:      {args.learning_rate}")
    print(f"  Max seq length:     {args.max_seq_len:,}")
    print(f"  Tokens / GPU:       {args.max_tokens_per_gpu:,}")
    print(f"  GPUs:               {args.nproc_per_node}")
    print("=" * 60)
    print()

    start = time.time()

    try:
        osft(
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,
            unfreeze_rank_ratio=args.unfreeze_rank_ratio,
            num_epochs=args.num_epochs,
            effective_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_len=args.max_seq_len,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            data_output_dir=args.data_output_dir,
            warmup_steps=0,
            use_liger=True,
            seed=42,
            lr_scheduler="cosine",
            checkpoint_at_epoch=True,
            save_final_checkpoint=True,
            nproc_per_node=args.nproc_per_node,
            nnodes=args.nnodes,
            node_rank=args.node_rank,
            rdzv_id=args.rdzv_id,
            rdzv_endpoint=args.rdzv_endpoint,
        )

        elapsed = time.time() - start
        checkpoint = find_most_recent_checkpoint(args.ckpt_output_dir)

        print()
        print("=" * 60)
        print(f"Training completed in {elapsed / 3600:.2f} hours")
        print(f"Final checkpoint: {checkpoint}")
        print("=" * 60)

    except Exception as exc:
        elapsed = time.time() - start
        print(f"\nTraining failed after {elapsed / 60:.1f} minutes: {exc}", file=sys.stderr)
        print("\nTroubleshooting:", file=sys.stderr)
        print("  - Reduce --max-tokens-per-gpu for OOM errors", file=sys.stderr)
        print("  - Verify data is valid JSONL", file=sys.stderr)
        print(
            "  - For continual learning, --unfreeze-rank-ratio 0.25-0.35 works well",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
