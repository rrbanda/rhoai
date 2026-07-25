#!/usr/bin/env python3
"""Train Granite models (3.3 8B or 4.0 variants) with SFT or OSFT.

Supports both Granite 3.3 8B Instruct and Granite 4.0 model family.  After
training, optionally merges the trained model with the base model using
linear interpolation (controlled by ``--model-weight``).

Adapted from the Training Hub ``sft_granite_example``, ``osft_granite_example``,
and ``sft_granite4_example``.

Hardware requirements:
    Granite 3.3 8B:      2+ A100 40 GB
    Granite 4.0 H-Small: 8x A100 80 GB (uses FSDP CPU offloading)
    Granite 4.0 others:  2+ A100 40 GB

Usage:
    # Granite 3.3 SFT
    python train_granite.py --algorithm sft \\
        --data-path /path/to/data.jsonl \\
        --ckpt-output-dir /path/to/checkpoints

    # Granite 3.3 OSFT
    python train_granite.py --algorithm osft \\
        --data-path /path/to/data.jsonl \\
        --ckpt-output-dir /path/to/checkpoints

    # Granite 4.0 H-Small SFT
    python train_granite.py --algorithm sft --granite-version 4.0-h-small \\
        --data-path /path/to/data.jsonl \\
        --ckpt-output-dir /path/to/checkpoints

    # Skip post-training interpolation
    python train_granite.py --algorithm sft --model-weight 0.0 \\
        --data-path /path/to/data.jsonl \\
        --ckpt-output-dir /path/to/checkpoints
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from datetime import datetime

import torch

from training_hub import osft, sft

GRANITE_CONFIGS = {
    "3.3": {
        "model_path": "ibm-granite/granite-3.3-8b-instruct",
        "model_name": "Granite 3.3 8B Instruct",
        "min_gpus": 2,
        "sft": {"batch_size": 256, "lr": 2e-5, "max_seq_len": 20000, "max_tpg": 25000},
        "osft": {"batch_size": 128, "lr": 5e-6, "max_seq_len": 4096, "max_tpg": 10000},
        "kwargs": {},
    },
    "4.0-h-small": {
        "model_path": "ibm-granite/granite-4.0-h-small",
        "model_name": "Granite 4.0 H-Small",
        "min_gpus": 8,
        "sft": {"batch_size": 128, "lr": 2e-5, "max_seq_len": 20000, "max_tpg": 25000},
        "kwargs_import": True,
    },
    "4.0-h-tiny": {
        "model_path": "ibm-granite/granite-4.0-h-tiny",
        "model_name": "Granite 4.0 H-Tiny",
        "min_gpus": 2,
        "sft": {"batch_size": 256, "lr": 2e-5, "max_seq_len": 20000, "max_tpg": 25000},
        "kwargs": {},
    },
    "4.0-h-micro": {
        "model_path": "ibm-granite/granite-4.0-h-micro",
        "model_name": "Granite 4.0 H-Micro",
        "min_gpus": 2,
        "sft": {"batch_size": 256, "lr": 2e-5, "max_seq_len": 20000, "max_tpg": 25000},
        "kwargs": {},
    },
    "4.0-micro": {
        "model_path": "ibm-granite/granite-4.0-micro",
        "model_name": "Granite 4.0 Micro",
        "min_gpus": 2,
        "sft": {"batch_size": 256, "lr": 2e-5, "max_seq_len": 20000, "max_tpg": 25000},
        "kwargs": {},
    },
}


def find_most_recent_checkpoint(output_dir: str) -> str | None:
    dirs = glob.glob(os.path.join(output_dir, "hf_format", "samples_*"))
    return max(dirs, key=os.path.getctime) if dirs else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Granite models (3.3 or 4.0)")

    parser.add_argument(
        "--algorithm", choices=["sft", "osft"], default="sft", help="Algorithm (default: sft)"
    )
    parser.add_argument(
        "--granite-version",
        choices=list(GRANITE_CONFIGS.keys()),
        default="3.3",
        help="Granite version (default: 3.3)",
    )
    parser.add_argument("--data-path", required=True, help="JSONL training data")
    parser.add_argument("--ckpt-output-dir", required=True, help="Checkpoint directory")
    parser.add_argument("--model-path", default=None, help="Override model ID")
    parser.add_argument("--num-epochs", type=int, default=3, help="Epochs (default: 3)")
    parser.add_argument("--nproc-per-node", type=int, default=None, help="GPUs (auto-detected)")
    parser.add_argument("--max-tokens-per-gpu", type=int, default=None, help="Tokens/GPU (auto)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto)")
    parser.add_argument("--learning-rate", type=float, default=None, help="LR (auto)")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Max seq len (auto)")
    parser.add_argument(
        "--model-weight",
        type=float,
        default=0.5,
        help="Weight for post-training interpolation, 0.0 to skip (default: 0.5)",
    )

    # OSFT-specific
    parser.add_argument("--unfreeze-rank-ratio", type=float, default=0.3, help="OSFT ratio (default: 0.3)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    algo = args.algorithm
    config = GRANITE_CONFIGS[args.granite_version]
    algo_defaults = config.get(algo, config.get("sft", {}))

    model_path = args.model_path or config["model_path"]
    nproc = args.nproc_per_node or (torch.cuda.device_count() if torch.cuda.is_available() else 0)
    batch_size = args.batch_size or algo_defaults.get("batch_size", 128)
    lr = args.learning_rate or algo_defaults.get("lr", 2e-5)
    max_seq_len = args.max_seq_len or algo_defaults.get("max_seq_len", 4096)
    max_tpg = args.max_tokens_per_gpu or algo_defaults.get("max_tpg", 25000)

    if nproc < config["min_gpus"]:
        print(f"Warning: {config['model_name']} needs >= {config['min_gpus']} GPUs")

    # Build extra kwargs (e.g. FSDP for Granite 4.0 H-Small)
    extra_kwargs: dict = {}
    if config.get("kwargs_import"):
        from instructlab.training import FSDPOptions
        extra_kwargs["fsdp_options"] = FSDPOptions(cpu_offload_params=True)
    elif "kwargs" in config:
        extra_kwargs = config["kwargs"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_output_dir = f"data/{algo}_granite_{timestamp}"

    print("=" * 60)
    print(f"{algo.upper()} Training: {config['model_name']}")
    print("=" * 60)
    print(f"  Model:          {model_path}")
    print(f"  Data:           {args.data_path}")
    print(f"  Output:         {args.ckpt_output_dir}")
    print(f"  GPUs:           {nproc}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Learning rate:  {lr}")
    print(f"  Max seq len:    {max_seq_len:,}")
    print(f"  Tokens / GPU:   {max_tpg:,}")
    if algo == "osft":
        print(f"  Unfreeze ratio: {args.unfreeze_rank_ratio}")
    print(f"  Interpolation:  {args.model_weight}")
    print("=" * 60)

    start = time.time()

    try:
        if algo == "sft":
            sft(
                model_path=model_path,
                data_path=args.data_path,
                ckpt_output_dir=args.ckpt_output_dir,
                num_epochs=args.num_epochs,
                effective_batch_size=batch_size,
                learning_rate=lr,
                max_seq_len=max_seq_len,
                max_tokens_per_gpu=max_tpg,
                data_output_dir=data_output_dir,
                warmup_steps=100,
                save_samples=0,
                checkpoint_at_epoch=True,
                accelerate_full_state_at_epoch=True,
                nproc_per_node=nproc,
                nnodes=1,
                **extra_kwargs,
            )
        else:
            osft(
                model_path=model_path,
                data_path=args.data_path,
                ckpt_output_dir=args.ckpt_output_dir,
                unfreeze_rank_ratio=args.unfreeze_rank_ratio,
                num_epochs=args.num_epochs,
                effective_batch_size=batch_size,
                learning_rate=lr,
                max_seq_len=max_seq_len,
                max_tokens_per_gpu=max_tpg,
                data_output_dir=data_output_dir,
                warmup_steps=0,
                use_liger=True,
                seed=42,
                lr_scheduler="cosine",
                checkpoint_at_epoch=True,
                save_final_checkpoint=True,
                nproc_per_node=nproc,
                nnodes=1,
                **extra_kwargs,
            )

        elapsed = time.time() - start
        checkpoint = find_most_recent_checkpoint(args.ckpt_output_dir)

        print("=" * 60)
        print(f"{algo.upper()} training completed in {elapsed / 3600:.2f} hours")
        if checkpoint:
            print(f"Most recent checkpoint: {checkpoint}")

        # Post-training model interpolation
        if checkpoint and 0.0 < args.model_weight < 1.0:
            from interpolator import interpolate_models

            interp_path = interpolate_models(
                model_path, checkpoint, trained_model_weight=args.model_weight
            )
            print(f"Interpolated model: {interp_path}")

        print("=" * 60)

    except Exception as exc:
        elapsed = time.time() - start
        print(
            f"\n{algo.upper()} training failed after {elapsed / 60:.1f} min: {exc}",
            file=sys.stderr,
        )
        print("Try reducing --max-tokens-per-gpu for OOM errors", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
