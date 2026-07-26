#!/usr/bin/env python3
"""Train Phi 4 Mini Instruct with SFT or OSFT.

Phi 4 Mini is a compact but dense model well-suited for efficient fine-tuning.
Supports both SFT and OSFT for continual learning without forgetting.

Adapted from the Training Hub ``sft_phi_example`` and ``osft_phi_example``.

Hardware requirements:
    8x A100 40 GB recommended (25k tokens/GPU for both algorithms).

Usage:
    python train_phi.py --algorithm sft \\
        --data-path /path/to/data.jsonl \\
        --ckpt-output-dir /path/to/checkpoints

    python train_phi.py --algorithm osft \\
        --data-path /path/to/data.jsonl \\
        --ckpt-output-dir /path/to/checkpoints
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

from training_hub import osft, sft


def find_most_recent_checkpoint(output_dir: str) -> str | None:
    dirs = glob.glob(os.path.join(output_dir, "hf_format", "samples_*.0"))
    return max(dirs, key=os.path.getctime) if dirs else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phi 4 Mini Instruct")

    parser.add_argument(
        "--algorithm", choices=["sft", "osft"], default="sft", help="Algorithm (default: sft)"
    )
    parser.add_argument("--data-path", required=True, help="JSONL training data")
    parser.add_argument("--ckpt-output-dir", required=True, help="Checkpoint directory")
    parser.add_argument(
        "--model-path",
        default="microsoft/Phi-4-mini-instruct",
        help="Model ID (default: microsoft/Phi-4-mini-instruct)",
    )
    parser.add_argument("--num-epochs", type=int, default=3, help="Epochs (default: 3)")
    parser.add_argument("--max-tokens-per-gpu", type=int, default=25000, help="Tokens/GPU (default: 25000)")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="GPUs (default: 1)")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="LR (default: 5e-6)")

    # OSFT-specific
    parser.add_argument("--unfreeze-rank-ratio", type=float, default=0.2, help="OSFT ratio (default: 0.2)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    algo = args.algorithm.upper()

    print("=" * 60)
    print(f"{algo} Training: Phi 4 Mini Instruct")
    print("=" * 60)
    print(f"  Model:          {args.model_path}")
    print(f"  Data:           {args.data_path}")
    print(f"  Output:         {args.ckpt_output_dir}")
    print(f"  GPUs:           {args.nproc_per_node}")
    print(f"  Tokens / GPU:   {args.max_tokens_per_gpu:,}")
    print(f"  Learning rate:  {args.learning_rate}")
    if args.algorithm == "osft":
        print(f"  Unfreeze ratio: {args.unfreeze_rank_ratio}")
    print("=" * 60)

    start = time.time()

    try:
        if args.algorithm == "sft":
            sft(
                model_path=args.model_path,
                data_path=args.data_path,
                ckpt_output_dir=args.ckpt_output_dir,
                num_epochs=args.num_epochs,
                effective_batch_size=64,
                learning_rate=args.learning_rate,
                max_seq_len=8192,
                max_tokens_per_gpu=args.max_tokens_per_gpu,
                data_output_dir="/dev/shm",
                warmup_steps=100,
                save_samples=0,
                checkpoint_at_epoch=True,
                accelerate_full_state_at_epoch=True,
                nproc_per_node=args.nproc_per_node,
                nnodes=1,
                node_rank=0,
                rdzv_id=102,
                rdzv_endpoint="127.0.0.1:29500",
            )
        else:
            osft(
                model_path=args.model_path,
                data_path=args.data_path,
                ckpt_output_dir=args.ckpt_output_dir,
                unfreeze_rank_ratio=args.unfreeze_rank_ratio,
                num_epochs=args.num_epochs,
                effective_batch_size=64,
                learning_rate=args.learning_rate,
                max_seq_len=8192,
                max_tokens_per_gpu=args.max_tokens_per_gpu,
                data_output_dir="/dev/shm",
                warmup_steps=0,
                use_liger=True,
                seed=42,
                lr_scheduler="cosine",
                checkpoint_at_epoch=True,
                save_final_checkpoint=True,
                nproc_per_node=args.nproc_per_node,
                nnodes=1,
                node_rank=0,
                rdzv_id=102,
                rdzv_endpoint="127.0.0.1:29500",
            )

        elapsed = time.time() - start
        print("=" * 60)
        print(f"{algo} training completed in {elapsed / 3600:.2f} hours")
        print(f"Checkpoints: {args.ckpt_output_dir}/hf_format/")
        ckpt = find_most_recent_checkpoint(args.ckpt_output_dir)
        if ckpt:
            print(f"Most recent: {ckpt}")
        print("=" * 60)

    except Exception as exc:
        elapsed = time.time() - start
        print(f"\n{algo} training failed after {elapsed / 60:.1f} min: {exc}", file=sys.stderr)
        print("Try reducing --max-tokens-per-gpu for OOM errors", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
