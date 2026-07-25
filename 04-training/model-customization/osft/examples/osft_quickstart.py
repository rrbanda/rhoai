"""Optimized Supervised Fine-Tuning (OSFT) quickstart using Training Hub.

OSFT selectively unfreezes a fraction of model parameters rather than
training every weight.  This dramatically reduces memory usage and
training time while retaining most of the quality of full-parameter SFT.

Key OSFT-specific parameters:
    unfreeze_rank_ratio  -- Fraction of weight-matrix singular-value
        directions to unfreeze (0.0-1.0).  Lower values save more memory
        but may limit learning capacity.  0.3 is a good default.
    unmask_messages      -- When True the loss is computed over all
        message roles (user + assistant), not just assistant turns.
        Helps the model learn conversational context.
    use_liger            -- Enable the Liger kernel for fused operations
        (RMSNorm, SwiGLU, CrossEntropy), reducing peak memory further.

Requirements:
    pip install training-hub
    # At least 2 GPUs with >= 40 GB VRAM each

Usage:
    torchrun --nproc_per_node 2 osft_quickstart.py
    torchrun --nproc_per_node 2 osft_quickstart.py --unfreeze-ratio 0.5
    torchrun --nproc_per_node 2 osft_quickstart.py --data ./my_data.jsonl --epochs 5
"""

from __future__ import annotations

import argparse
import sys

from training_hub import osft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model with Optimized SFT (OSFT).",
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
        default="./checkpoints/osft",
        help="Directory to write model checkpoints",
    )
    parser.add_argument(
        "--unfreeze-ratio",
        type=float,
        default=0.3,
        help="Fraction of singular-value directions to unfreeze (default: 0.3)",
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
        default=10000,
        help="Token budget per GPU per step",
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
    print("OSFT Training Configuration")
    print("=" * 60)
    print(f"  Model:              {args.model}")
    print(f"  Data:               {args.data}")
    print(f"  Output dir:         {args.output_dir}")
    print(f"  Unfreeze ratio:     {args.unfreeze_ratio}")
    print(f"  Epochs:             {args.epochs}")
    print(f"  Effective batch:    {args.batch_size}")
    print(f"  Learning rate:      {args.lr}")
    print(f"  Max seq length:     {args.max_seq_len}")
    print(f"  Tokens / GPU:       {args.max_tokens_per_gpu}")
    print(f"  GPUs:               {args.nproc}")
    print(f"  Unmask messages:    True")
    print(f"  Liger kernels:      True")
    print("=" * 60)

    try:
        osft(
            model_path=args.model,
            data_path=args.data,
            ckpt_output_dir=args.output_dir,
            unfreeze_rank_ratio=args.unfreeze_ratio,
            num_epochs=args.epochs,
            effective_batch_size=args.batch_size,
            learning_rate=args.lr,
            max_seq_len=args.max_seq_len,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            unmask_messages=True,
            use_liger=True,
            nproc_per_node=args.nproc,
            checkpoint_at_epoch=True,
        )
    except Exception as exc:
        print(f"\nTraining failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nTraining complete. Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
