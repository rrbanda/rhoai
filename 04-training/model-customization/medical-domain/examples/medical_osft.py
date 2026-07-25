#!/usr/bin/env python3
"""OSFT fine-tuning of Ministral 3B on a medical flashcards dataset.

Orthogonal Subspace Fine-Tuning constrains weight updates to low-rank
subspaces orthogonal to the model's critical knowledge subspace.  This
enables medical domain adaptation without catastrophic forgetting -- the
model keeps its general reasoning and language abilities intact.

Note: Liger kernels are not yet supported for the Ministral 3 architecture,
so ``use_liger`` is disabled here.

Adapted from the Training Hub ``osft_ministral_medical_example``.

Dataset: medalpaca/medical_meadow_medical_flashcards
    https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards

See the parent README.md for data-preparation instructions.

Hardware requirements:
    8x A100 40 GB recommended.  Reduce ``--nproc-per-node`` for fewer GPUs.

Usage:
    python medical_osft.py \\
        --data-path /path/to/medical_flashcards.jsonl \\
        --ckpt-output-dir /path/to/checkpoints

    python medical_osft.py \\
        --data-path data.jsonl --ckpt-output-dir ./ckpts \\
        --unfreeze-rank-ratio 0.3
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import time

from training_hub import osft


def _ratio_0_to_1(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0.0 and 1.0, got {parsed}")
    return parsed


def find_most_recent_checkpoint(output_dir: str) -> str | None:
    """Return the most recent checkpoint path, or *None* if none exist."""
    dirs = glob.glob(os.path.join(output_dir, "hf_format", "samples_*.0"))
    return max(dirs, key=os.path.getctime) if dirs else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OSFT fine-tuning on medical flashcards (Ministral 3B).",
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
        "--unfreeze-rank-ratio",
        type=_ratio_0_to_1,
        default=0.25,
        metavar="RATIO",
        help="Fraction of directions to unfreeze, 0.0-1.0 (default: 0.25)",
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
    print("Medical OSFT: Ministral 3B on Medical Flashcards")
    print("=" * 60)
    print(f"  Model:              {args.model_path}")
    print(f"  Data:               {args.data_path}")
    print(f"  Output:             {args.ckpt_output_dir}")
    print(f"  Unfreeze ratio:     {args.unfreeze_rank_ratio}")
    print(f"  GPUs:               {args.nproc_per_node}")
    print(f"  Tokens / GPU:       {args.max_tokens_per_gpu:,}")
    print("=" * 60)

    start = time.time()

    try:
        osft(
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,
            unfreeze_rank_ratio=args.unfreeze_rank_ratio,
            num_epochs=args.num_epochs,
            effective_batch_size=32,
            learning_rate=5e-6,
            max_seq_len=4096,
            max_tokens_per_gpu=args.max_tokens_per_gpu,
            data_output_dir=args.data_output_dir,
            warmup_steps=0,
            use_liger=False,  # Not supported for Ministral 3 architecture
            seed=42,
            lr_scheduler="cosine",
            checkpoint_at_epoch=True,
            save_final_checkpoint=True,
            nproc_per_node=args.nproc_per_node,
            nnodes=1,
            node_rank=0,
            rdzv_id=301,
            rdzv_endpoint="127.0.0.1:29501",
        )

        elapsed = time.time() - start
        checkpoint = find_most_recent_checkpoint(args.ckpt_output_dir)

        print("=" * 60)
        print(f"OSFT training completed in {elapsed / 3600:.2f} hours")
        print(f"Checkpoints: {args.ckpt_output_dir}/hf_format")
        if checkpoint:
            print(f"Most recent: {checkpoint}")
        print("=" * 60)

    except Exception as exc:
        elapsed = time.time() - start
        print(f"\nTraining failed after {elapsed / 60:.1f} minutes: {exc}", file=sys.stderr)
        print("Troubleshooting:", file=sys.stderr)
        print("  - Reduce --max-tokens-per-gpu for OOM errors", file=sys.stderr)
        print(
            "  - Try --unfreeze-rank-ratio between 0.2-0.4 for domain adaptation",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
