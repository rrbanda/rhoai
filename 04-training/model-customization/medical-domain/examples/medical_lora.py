#!/usr/bin/env python3
"""LoRA fine-tuning of Ministral 3B on a medical flashcards dataset.

Low-Rank Adaptation (LoRA) adds small trainable matrices to the model's
attention and MLP layers, enabling parameter-efficient fine-tuning with
dramatically lower memory.  Supports QLoRA (4-bit quantisation) for
training on a single consumer GPU.

Adapted from the Training Hub ``lora_ministral_medical_example``.

Dataset: medalpaca/medical_meadow_medical_flashcards
    https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards

See the parent README.md for data-preparation instructions.

Hardware requirements:
    LoRA:  1x A100 40 GB (single GPU)
    QLoRA: 1x A10 24 GB or equivalent

Usage:
    # LoRA (single GPU)
    python medical_lora.py \\
        --data-path /path/to/medical_flashcards.jsonl \\
        --ckpt-output-dir /path/to/checkpoints

    # QLoRA with 4-bit quantisation
    python medical_lora.py \\
        --data-path /path/to/medical_flashcards.jsonl \\
        --ckpt-output-dir /path/to/checkpoints \\
        --qlora

    # Multi-GPU (requires torchrun)
    torchrun --nproc-per-node=8 medical_lora.py \\
        --data-path /path/to/medical_flashcards.jsonl \\
        --ckpt-output-dir /path/to/checkpoints \\
        --nproc-per-node 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from training_hub import lora_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning on medical flashcards (Ministral 3B).",
    )
    parser.add_argument(
        "--data-path", required=True, help="Path to training data (JSONL messages format)"
    )
    parser.add_argument(
        "--ckpt-output-dir", required=True, help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--model-path",
        default="mistralai/Ministral-3-3B-Instruct-2512",
        help="HuggingFace model ID or local path (default: Ministral-3-3B-Instruct-2512)",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=3, help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)",
    )
    parser.add_argument(
        "--lora-r", type=int, default=32, help="LoRA rank (default: 32)"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=64, help="LoRA alpha scaling (default: 64)"
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Enable QLoRA (4-bit quantisation) for reduced memory usage",
    )
    parser.add_argument(
        "--nproc-per-node", type=int, default=1, help="Number of GPUs (default: 1)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = "QLoRA (4-bit)" if args.qlora else "LoRA (full precision)"

    print("=" * 60)
    print("Medical LoRA: Ministral 3B on Medical Flashcards")
    print("=" * 60)
    print(f"  Model:          {args.model_path}")
    print(f"  Data:           {args.data_path}")
    print(f"  Output:         {args.ckpt_output_dir}")
    print(f"  Mode:           {mode}")
    print(f"  LoRA rank:      {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"  GPUs:           {args.nproc_per_node}")
    print(f"  Max seq len:    {args.max_seq_len:,}")
    print("=" * 60)

    if args.nproc_per_node > 1 and "RANK" not in os.environ:
        print(
            f"WARNING: --nproc-per-node={args.nproc_per_node} but this script "
            "is not running under torchrun.  Multi-GPU training requires:\n"
            f"  torchrun --nproc-per-node={args.nproc_per_node} "
            f"{os.path.basename(__file__)} ...\n"
        )

    start = time.time()

    try:
        lora_sft(
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            num_epochs=args.num_epochs,
            effective_batch_size=32,
            learning_rate=args.learning_rate,
            max_seq_len=args.max_seq_len,
            warmup_steps=10,
            load_in_4bit=args.qlora,
            bf16=True,
            sample_packing=True,
            logging_steps=10,
            save_steps=200,
            save_total_limit=3,
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
            node_rank=0,
            rdzv_id=302,
            rdzv_endpoint="127.0.0.1:29502",
        )

        elapsed = time.time() - start
        print("=" * 60)
        print(f"LoRA training completed in {elapsed / 3600:.2f} hours")
        print(f"Output: {args.ckpt_output_dir}")
        print("=" * 60)

    except Exception as exc:
        elapsed = time.time() - start
        print(f"\nTraining failed after {elapsed / 60:.1f} minutes: {exc}", file=sys.stderr)
        print("Troubleshooting:", file=sys.stderr)
        print("  - Enable QLoRA to reduce memory: --qlora", file=sys.stderr)
        print("  - Reduce sequence length: --max-seq-len 2048", file=sys.stderr)
        print("  - Verify data format (expects JSONL with 'messages' field)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
