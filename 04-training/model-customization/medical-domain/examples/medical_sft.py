#!/usr/bin/env python3
"""SFT fine-tuning of Ministral 3B on a medical flashcards dataset.

Full-parameter supervised fine-tuning on medical Q&A data.  All model weights
are updated, giving maximum adaptation to the medical domain at the cost of
potentially forgetting some general capabilities.

Adapted from the Training Hub ``sft_ministral_medical_example``.

Dataset: medalpaca/medical_meadow_medical_flashcards
    https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards

See the parent README.md for data-preparation instructions.

Hardware requirements:
    8x A100 40 GB recommended.  Reduce ``--nproc-per-node`` for fewer GPUs.

Usage:
    python medical_sft.py \\
        --data-path /path/to/medical_flashcards.jsonl \\
        --ckpt-output-dir /path/to/checkpoints
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

from training_hub import sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT fine-tuning on medical flashcards (Ministral 3B).",
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
        "--max-tokens-per-gpu",
        type=int,
        default=8192,
        help="Token budget per GPU per step (default: 8192)",
    )
    parser.add_argument(
        "--nproc-per-node", type=int, default=8, help="Number of GPUs (default: 8)"
    )
    parser.add_argument(
        "--data-output-dir",
        default="/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir(),
        help="Directory for processed data (default: /dev/shm or system temp)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Medical SFT: Ministral 3B on Medical Flashcards")
    print("=" * 60)
    print(f"  Model:          {args.model_path}")
    print(f"  Data:           {args.data_path}")
    print(f"  Output:         {args.ckpt_output_dir}")
    print(f"  GPUs:           {args.nproc_per_node}")
    print(f"  Tokens / GPU:   {args.max_tokens_per_gpu:,}")
    print("=" * 60)

    start = time.time()

    try:
        sft(
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,
            num_epochs=args.num_epochs,
            effective_batch_size=32,
            learning_rate=1e-5,
            max_seq_len=4096,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            data_output_dir=args.data_output_dir,
            warmup_steps=0,
            save_samples=0,
            checkpoint_at_epoch=True,
            accelerate_full_state_at_epoch=False,
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
            node_rank=0,
            rdzv_id=300,
            rdzv_endpoint="127.0.0.1:42069",
        )

        elapsed = time.time() - start
        print("=" * 60)
        print(f"Training completed in {elapsed / 3600:.2f} hours")
        print(f"Checkpoints: {args.ckpt_output_dir}/hf_format/")
        print("=" * 60)

    except Exception as exc:
        elapsed = time.time() - start
        print(f"\nTraining failed after {elapsed / 60:.1f} minutes: {exc}", file=sys.stderr)
        print("Try reducing --max-tokens-per-gpu for OOM errors", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
