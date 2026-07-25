"""LoRA / QLoRA fine-tuning quickstart using Training Hub.

Low-Rank Adaptation (LoRA) inserts small trainable matrices into each
transformer layer while keeping the base weights frozen.  This allows
fine-tuning large models on a single GPU with minimal memory overhead.

Pass --qlora to enable 4-bit quantization (QLoRA), which further reduces
VRAM usage at a small accuracy trade-off.

Requirements:
    pip install training-hub
    # Single GPU with >= 24 GB VRAM (A10G / L4 / A100)

Usage -- single GPU:
    python lora_quickstart.py
    python lora_quickstart.py --qlora
    python lora_quickstart.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 5

Usage -- multi-GPU with torchrun:
    torchrun --nproc_per_node 2 lora_quickstart.py --nproc 2
"""

from __future__ import annotations

import argparse
import sys

from training_hub import lora_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model with LoRA or QLoRA.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID or local path (default: Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--data",
        default="./training_data.jsonl",
        help="Path to chat-format JSONL training data",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/lora",
        help="Directory to write LoRA adapter checkpoints",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank -- higher values increase capacity and memory (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha scaling factor (default: 32)",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length in tokens",
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Enable 4-bit quantization (QLoRA) to reduce VRAM usage",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="Number of GPUs (use torchrun for multi-GPU, default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mode = "QLoRA (4-bit)" if args.qlora else "LoRA"

    print("=" * 60)
    print(f"{mode} Training Configuration")
    print("=" * 60)
    print(f"  Model:              {args.model}")
    print(f"  Data:               {args.data}")
    print(f"  Output dir:         {args.output_dir}")
    print(f"  LoRA rank (r):      {args.lora_r}")
    print(f"  LoRA alpha:         {args.lora_alpha}")
    print(f"  Epochs:             {args.epochs}")
    print(f"  Learning rate:      {args.lr}")
    print(f"  Max seq length:     {args.max_seq_len}")
    print(f"  4-bit quantized:    {args.qlora}")
    print(f"  GPUs:               {args.nproc}")
    print("=" * 60)

    kwargs: dict = dict(
        model_path=args.model,
        data_path=args.data,
        ckpt_output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        max_seq_len=args.max_seq_len,
    )
    if args.qlora:
        kwargs["load_in_4bit"] = True
    if args.nproc > 1:
        kwargs["nproc_per_node"] = args.nproc

    try:
        lora_sft(**kwargs)
    except Exception as exc:
        print(f"\nTraining failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nTraining complete. Adapter checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
