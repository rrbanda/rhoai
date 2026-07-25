"""Group Relative Policy Optimization (GRPO) quickstart using Training Hub.

GRPO is a reinforcement-learning method that improves model reasoning by
sampling multiple candidate responses (a "group") for each prompt and
using verifiable reward signals to update the policy.

Key concepts:
    Iterations       -- Number of outer training loops.  Each iteration
        samples a batch of prompts, generates rollouts, scores them, and
        performs a policy-gradient update.
    Group size        -- Number of candidate completions generated per
        prompt.  Larger groups give more stable gradient estimates but
        cost more compute.
    Prompt batch size -- How many prompts are sampled per iteration.
    Verifiable rewards -- Rewards derived from programmatic checks (e.g.,
        tool-call success, format compliance) rather than a learned reward
        model, making training more stable and interpretable.
    Backend           -- "art" (Asynchronous Reinforcement Training) or
        "verl" (Volcano Engine RL).  ART is recommended for most setups.

Requirements:
    pip install training-hub
    # At least 2 GPUs with >= 40 GB VRAM each

Usage:
    torchrun --nproc_per_node 2 grpo_quickstart.py
    torchrun --nproc_per_node 2 grpo_quickstart.py --backend verl
    torchrun --nproc_per_node 4 grpo_quickstart.py --n-gpus 4 --group-size 16
"""

from __future__ import annotations

import argparse
import sys

from training_hub import lora_grpo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a language model with GRPO reinforcement learning.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B",
        help="HuggingFace model ID or local path (default: Qwen3-4B)",
    )
    parser.add_argument(
        "--data",
        default="./tool_call_traces.jsonl",
        help="Path to prompt JSONL data with verifiable reward signals",
    )
    parser.add_argument(
        "--output-dir",
        default="./checkpoints/grpo",
        help="Directory to write adapter checkpoints",
    )
    parser.add_argument(
        "--backend",
        choices=["art", "verl"],
        default="art",
        help="RL backend: 'art' (Async RL Training) or 'verl' (default: art)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank (default: 32)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha scaling factor (default: 64)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="Number of outer GRPO iterations (default: 15)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Rollout candidates per prompt (default: 8)",
    )
    parser.add_argument(
        "--prompt-batch",
        type=int,
        default=100,
        help="Prompts sampled per iteration (default: 100)",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=2,
        help="Number of GPUs (must match torchrun --nproc_per_node)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("GRPO Training Configuration")
    print("=" * 60)
    print(f"  Model:              {args.model}")
    print(f"  Data:               {args.data}")
    print(f"  Output dir:         {args.output_dir}")
    print(f"  Backend:            {args.backend}")
    print(f"  LoRA rank (r):      {args.lora_r}")
    print(f"  LoRA alpha:         {args.lora_alpha}")
    print(f"  Iterations:         {args.iterations}")
    print(f"  Group size:         {args.group_size}")
    print(f"  Prompt batch size:  {args.prompt_batch}")
    print(f"  Learning rate:      {args.lr}")
    print(f"  GPUs:               {args.n_gpus}")
    print("=" * 60)

    try:
        lora_grpo(
            model_path=args.model,
            data_path=args.data,
            ckpt_output_dir=args.output_dir,
            backend=args.backend,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            num_iterations=args.iterations,
            group_size=args.group_size,
            prompt_batch_size=args.prompt_batch,
            learning_rate=args.lr,
            nproc_per_node=args.n_gpus,
        )
    except Exception as exc:
        print(f"\nTraining failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nTraining complete. Checkpoints saved to {args.output_dir}")


if __name__ == "__main__":
    main()
