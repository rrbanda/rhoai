"""LAB Multi-Phase Training — end-to-end CLI script.

Runs the two-phase LAB (Large-scale Alignment for chatBots) training
pipeline using ``training_hub.sft``:

    Phase07 — Knowledge tuning on a focused knowledge dataset.
    Phase10 — Skills + replay training using a combined dataset that
              includes new skills, Phase07 knowledge replay, and base
              model instruction replay.

Adapted from the Training Hub lab_multiphase_training example.

Requirements:
    pip install training-hub

Usage:
    python lab_multiphase_training.py \\
        --base-model-path /path/to/model \\
        --phase07-data-path /path/to/knowledge.jsonl \\
        --phase10-data-path /path/to/skills_replay.jsonl \\
        --ckpt-output-base-dir /path/to/checkpoints

    # Resume from an existing Phase07 checkpoint:
    python lab_multiphase_training.py \\
        --base-model-path /path/to/model \\
        --phase07-data-path /path/to/knowledge.jsonl \\
        --phase10-data-path /path/to/skills_replay.jsonl \\
        --ckpt-output-base-dir /path/to/checkpoints \\
        --skip-phase07 --phase07-checkpoint /path/to/phase07/hf_format/samples_XXX
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

from training_hub import sft


def find_most_recent_checkpoint(checkpoint_dir: str) -> str | None:
    """Return the path to the most recently created sub-directory."""
    if not os.path.isdir(checkpoint_dir):
        print(f"  Checkpoint directory not found: {checkpoint_dir}")
        return None

    entries = [
        os.path.join(checkpoint_dir, d)
        for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if not entries:
        print(f"  No checkpoints found in {checkpoint_dir}")
        return None

    latest = max(entries, key=os.path.getctime)
    print(f"  Latest checkpoint: {latest}")
    return latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LAB Multi-Phase Training (Phase07 knowledge + Phase10 skills/replay).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data / model paths
    parser.add_argument(
        "--base-model-path", required=True,
        help="Path to the base model (HuggingFace ID or local)",
    )
    parser.add_argument(
        "--phase07-data-path", required=True,
        help="JSONL knowledge data for Phase07",
    )
    parser.add_argument(
        "--phase10-data-path", required=True,
        help="JSONL skills + replay data for Phase10",
    )
    parser.add_argument(
        "--ckpt-output-base-dir", required=True,
        help="Base directory for all checkpoint output",
    )
    parser.add_argument(
        "--experiment-prefix", default="lab_multiphase",
        help="Prefix for experiment names",
    )

    # Training hyperparameters
    parser.add_argument("--num-epochs", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-tokens-per-gpu", type=int, default=25000)
    parser.add_argument("--max-seq-len", type=int, default=20000)
    parser.add_argument("--phase07-batch-size", type=int, default=128)
    parser.add_argument("--phase10-batch-size", type=int, default=3840)

    # Distributed
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--rdzv-id", type=int, default=47)
    parser.add_argument("--rdzv-endpoint", default="0.0.0.0:12345")

    # Control
    parser.add_argument(
        "--skip-phase07", action="store_true",
        help="Skip Phase07 and use an existing checkpoint for Phase10",
    )
    parser.add_argument(
        "--phase07-checkpoint",
        help="Path to Phase07 checkpoint (required with --skip-phase07)",
    )

    args = parser.parse_args()
    if args.skip_phase07 and not args.phase07_checkpoint:
        parser.error("--phase07-checkpoint is required when --skip-phase07 is set")
    return args


def run_phase(
    label: str,
    model_path: str,
    data_path: str,
    ckpt_output_dir: str,
    args: argparse.Namespace,
    effective_batch_size: int,
    accelerate_full_state: bool = False,
) -> None:
    """Execute one phase of SFT training."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Model:       {model_path}")
    print(f"  Data:        {data_path}")
    print(f"  Output:      {ckpt_output_dir}")
    print(f"  Batch size:  {effective_batch_size}")
    print(f"  Epochs:      {args.num_epochs}")
    print(f"{'=' * 60}")

    t0 = time.time()
    sft(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=ckpt_output_dir,
        num_epochs=args.num_epochs,
        effective_batch_size=effective_batch_size,
        learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len,
        max_tokens_per_gpu=args.max_tokens_per_gpu,
        data_output_dir="/dev/shm",
        warmup_steps=0,
        save_samples=0,
        checkpoint_at_epoch=True,
        accelerate_full_state_at_epoch=accelerate_full_state,
        nproc_per_node=args.nproc_per_node,
        nnodes=args.nnodes,
        node_rank=args.node_rank,
        rdzv_id=args.rdzv_id,
        rdzv_endpoint=args.rdzv_endpoint,
    )
    elapsed = time.time() - t0
    print(f"\n  {label} completed in {elapsed / 3600:.2f} hours")


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # -- Phase07 ---------------------------------------------------------
    phase07_ckpt = args.phase07_checkpoint
    if not args.skip_phase07:
        phase07_dir = os.path.join(
            args.ckpt_output_base_dir,
            f"{args.experiment_prefix}_phase07_{timestamp}",
        )
        try:
            run_phase(
                label="Phase07 — Knowledge Tuning",
                model_path=args.base_model_path,
                data_path=args.phase07_data_path,
                ckpt_output_dir=phase07_dir,
                args=args,
                effective_batch_size=args.phase07_batch_size,
                accelerate_full_state=False,
            )
        except Exception as exc:
            print(f"\nPhase07 failed: {exc}", file=sys.stderr)
            sys.exit(1)

        phase07_ckpt = find_most_recent_checkpoint(
            os.path.join(phase07_dir, "hf_format")
        )
    else:
        print(f"\nSkipping Phase07 — using checkpoint: {phase07_ckpt}")

    if not phase07_ckpt:
        print("No Phase07 checkpoint available. Cannot continue.", file=sys.stderr)
        sys.exit(1)

    # -- Phase10 ---------------------------------------------------------
    phase10_dir = os.path.join(
        args.ckpt_output_base_dir,
        f"{args.experiment_prefix}_phase10_{timestamp}",
    )
    try:
        run_phase(
            label="Phase10 — Skills + Replay",
            model_path=phase07_ckpt,
            data_path=args.phase10_data_path,
            ckpt_output_dir=phase10_dir,
            args=args,
            effective_batch_size=args.phase10_batch_size,
            accelerate_full_state=True,
        )
    except Exception as exc:
        print(f"\nPhase10 failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # -- Summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("  LAB Multi-Phase Training Complete")
    print("=" * 60)
    if not args.skip_phase07:
        print(f"  Phase07 output: {phase07_dir}")
    print(f"  Phase10 output: {phase10_dir}")
    print(f"  Final model:    {phase10_dir}/hf_format/<latest>")
    print("=" * 60)


if __name__ == "__main__":
    main()
