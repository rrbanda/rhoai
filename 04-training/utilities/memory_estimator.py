"""Estimate GPU VRAM requirements before training with Training Hub.

Uses ``training_hub.estimate()`` to calculate expected memory usage for
SFT, OSFT, LoRA, and QLoRA training methods.  Run this *before* launching
a training job to verify your hardware can handle the configuration.

Adapted from the Training Hub memory_estimator_example notebook.

Requirements:
    pip install training-hub

Usage:
    python memory_estimator.py --model ibm-granite/granite-3.3-2b-instruct
    python memory_estimator.py --model Qwen/Qwen2.5-7B-Instruct --method osft
    python memory_estimator.py --model meta-llama/Llama-3.1-8B-Instruct --method lora --lora-r 16
    python memory_estimator.py --method qlora --num-gpus 4 --gpu-memory 80
"""

from __future__ import annotations

import argparse
import sys

from training_hub import estimate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate GPU VRAM requirements for Training Hub fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="ibm-granite/granite-3.3-2b-instruct",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--method",
        choices=["sft", "osft", "lora", "qlora"],
        default="sft",
        help="Training method to estimate",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs to distribute training across",
    )
    parser.add_argument(
        "--gpu-memory",
        type=int,
        default=48,
        help="Per-GPU memory in GB",
    )
    parser.add_argument(
        "--max-tokens-per-gpu",
        type=int,
        default=8192,
        help="Max tokens per GPU per step (used by SFT/OSFT)",
    )

    osft_group = parser.add_argument_group("OSFT options")
    osft_group.add_argument(
        "--unfreeze-rank-ratio",
        type=float,
        default=0.25,
        help="Fraction of each weight matrix to unfreeze (OSFT only)",
    )

    lora_group = parser.add_argument_group("LoRA / QLoRA options")
    lora_group.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    lora_group.add_argument(
        "--batch-size", type=int, default=4, help="Per-device batch size (LoRA/QLoRA)"
    )
    lora_group.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length (LoRA/QLoRA)",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Verbosity level (0=silent, 1=summary, 2=detailed)",
    )
    return parser.parse_args()


def _fmt_gb(bytes_val: float) -> str:
    return f"{bytes_val / (2**30):.2f} GB"


def main() -> None:
    args = parse_args()
    gpu_memory_bytes = args.gpu_memory * (2**30)

    kwargs: dict = {
        "training_method": args.method,
        "num_gpus": args.num_gpus,
        "gpu_memory": gpu_memory_bytes,
        "model_path": args.model,
        "verbose": args.verbose,
    }

    if args.method in ("sft", "osft"):
        kwargs["max_tokens_per_gpu"] = args.max_tokens_per_gpu
    if args.method == "osft":
        kwargs["unfreeze_rank_ratio"] = args.unfreeze_rank_ratio
    if args.method in ("lora", "qlora"):
        kwargs["lora_r"] = args.lora_r
        kwargs["batch_size"] = args.batch_size
        kwargs["max_seq_len"] = args.max_seq_len

    print("=" * 60)
    print(f"Memory Estimation — {args.method.upper()}")
    print("=" * 60)
    print(f"  Model:           {args.model}")
    print(f"  Method:          {args.method}")
    print(f"  GPUs:            {args.num_gpus} x {args.gpu_memory} GB")
    if args.method in ("sft", "osft"):
        print(f"  Tokens / GPU:    {args.max_tokens_per_gpu}")
    if args.method == "osft":
        print(f"  Unfreeze ratio:  {args.unfreeze_rank_ratio}")
    if args.method in ("lora", "qlora"):
        print(f"  LoRA rank:       {args.lora_r}")
        print(f"  Batch size:      {args.batch_size}")
        print(f"  Max seq len:     {args.max_seq_len}")
    print("=" * 60)

    try:
        lower, expected, upper = estimate(**kwargs)
    except Exception as exc:
        print(f"\nEstimation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\nResults (per-GPU VRAM estimate):")
    print(f"  Lower bound:  {_fmt_gb(lower)}")
    print(f"  Expected:     {_fmt_gb(expected)}")
    print(f"  Upper bound:  {_fmt_gb(upper)}")
    print(f"  Available:    {_fmt_gb(gpu_memory_bytes)}")

    if upper > gpu_memory_bytes:
        print(
            "\nWARNING: Upper-bound estimate exceeds available GPU memory. "
            "Consider reducing max_tokens_per_gpu, batch_size, or using "
            "a smaller model / more GPUs."
        )
    elif expected > gpu_memory_bytes:
        print(
            "\nCAUTION: Expected estimate is close to or exceeds GPU memory. "
            "You may encounter OOM errors under certain conditions."
        )
    else:
        print("\nYour configuration should fit within GPU memory.")


if __name__ == "__main__":
    main()
