"""Supervised Fine-Tuning (SFT) quickstart using Training Hub.

Full-parameter SFT updates every weight in the model, producing the
highest-quality results when you have enough data and GPU memory.
This script fine-tunes a Llama-3.1-8B-Instruct model on a chat-format
JSONL dataset using the ``training_hub.sft`` API.

Requirements:
    pip install training-hub
    # At least 2 GPUs with >= 40 GB VRAM each (A100 / H100 recommended)

Usage:
    torchrun --nproc_per_node 2 sft_quickstart.py
    torchrun --nproc_per_node 2 sft_quickstart.py --model meta-llama/Llama-3.1-8B-Instruct
    torchrun --nproc_per_node 2 sft_quickstart.py --data ./my_data.jsonl --epochs 5
"""

from __future__ import annotations

import argparse
import sys

from training_hub import sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model with full-parameter SFT.",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID or local path (default: Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--data",
        default="./training_data.jsonl",
        help="Path to chat-format JSONL training data",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/sft",
        help="Directory to write model checkpoints",
    )
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
    parser.add_argument(
        "--max-tokens-per-gpu",
        type=int,
        default=25000,
        help="Token budget per GPU per step (controls micro-batch packing)",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=2,
        help="Number of GPUs (must match torchrun --nproc_per_node)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("SFT Training Configuration")
    print("=" * 60)
    print(f"  Model:              {args.model}")
    print(f"  Data:               {args.data}")
    print(f"  Output dir:         {args.output_dir}")
    print(f"  Epochs:             {args.epochs}")
    print(f"  Effective batch:    {args.batch_size}")
    print(f"  Learning rate:      {args.lr}")
    print(f"  Max seq length:     {args.max_seq_len}")
    print(f"  Tokens / GPU:       {args.max_tokens_per_gpu}")
    print(f"  GPUs:               {args.nproc}")
    print("=" * 60)

    try:
        sft(
            model_path=args.model,
            data_path=args.data,
            ckpt_output_dir=args.output_dir,
            num_epochs=args.epochs,
            effective_batch_size=args.batch_size,
            learning_rate=args.lr,
            max_seq_len=args.max_seq_len,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            nproc_per_node=args.nproc,
            checkpoint_at_epoch=True,
        )
    except Exception as exc:
        print(f"\nTraining failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nTraining complete. Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
