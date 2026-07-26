"""Train a small model with LoRA GRPO using Training Hub.

Uses the function-calling JSONL from step 02 to fine-tune a student model
(e.g., Qwen3-4B) with Group Relative Policy Optimization (GRPO). The student
learns to select the correct tools, format arguments properly, and chain
multi-step tool calls from the expert demonstrations.

Supports two backends:
  - art: Single-GPU training (default, good for prototyping)
  - verl: Multi-GPU distributed training (for production runs)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a student model with LoRA GRPO via Training Hub"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Student model to train (default: STUDENT_MODEL env var or Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training JSONL (default: OUTPUT_DIR/training_data.jsonl)",
    )
    parser.add_argument(
        "--ckpt-output-dir",
        type=str,
        default=None,
        help="Directory for training checkpoints (default: CHECKPOINT_DIR env var)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["art", "verl"],
        default="art",
        help="Training backend: 'art' for single-GPU, 'verl' for multi-GPU (default: art)",
    )
    parser.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="Number of GPUs for verl backend (default: 1)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=15,
        help="Number of GRPO training iterations (default: 15)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=8,
        help="LoRA alpha scaling factor (default: 8)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="GRPO group size for relative scoring (default: 8)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=100,
        help="Number of prompts per batch (default: 100)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    # Resolve paths and model from args or environment
    student_model = args.model_path or os.environ.get("STUDENT_MODEL", "Qwen/Qwen3-4B")
    output_dir = os.environ.get("OUTPUT_DIR", "./generated_data")
    data_path = args.data_path or os.path.join(output_dir, "training_data.jsonl")
    ckpt_output_dir = args.ckpt_output_dir or os.environ.get("CHECKPOINT_DIR", "./checkpoints")

    if not Path(data_path).exists():
        print(f"ERROR: Training data not found: {data_path}")
        print("Run 01_generate_tool_data.py and 02_format_training_data.py first.")
        sys.exit(1)

    # Count training examples
    with open(data_path) as f:
        n_examples = sum(1 for _ in f)

    # -- Print configuration --------------------------------------------------
    print("=" * 60)
    print("MCP Distillation -- GRPO Training")
    print("=" * 60)
    print(f"  Student model:    {student_model}")
    print(f"  Training data:    {data_path} ({n_examples} examples)")
    print(f"  Checkpoint dir:   {ckpt_output_dir}")
    print(f"  Backend:          {args.backend}")
    if args.backend == "verl":
        print(f"  GPUs:             {args.n_gpus}")
    print(f"  LoRA rank:        {args.lora_r}")
    print(f"  LoRA alpha:       {args.lora_alpha}")
    print(f"  Iterations:       {args.num_iterations}")
    print(f"  Group size:       {args.group_size}")
    print(f"  Prompt batch:     {args.prompt_batch_size}")
    print(f"  Learning rate:    {args.learning_rate}")
    print("=" * 60)

    # -- Run GRPO training ----------------------------------------------------
    from training_hub import lora_grpo

    train_kwargs = {
        "model_path": student_model,
        "data_path": data_path,
        "ckpt_output_dir": ckpt_output_dir,
        "backend": args.backend,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "num_iterations": args.num_iterations,
        "group_size": args.group_size,
        "prompt_batch_size": args.prompt_batch_size,
        "learning_rate": args.learning_rate,
    }

    if args.backend == "verl" and args.n_gpus > 1:
        train_kwargs["n_gpus"] = args.n_gpus

    print("\nStarting GRPO training...")
    print("-" * 60)

    result = lora_grpo(**train_kwargs)

    # -- Print results --------------------------------------------------------
    print("-" * 60)
    print("Training complete!")

    if isinstance(result, dict):
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print(f"  Result: {result}")

    print(f"\nCheckpoints saved to: {ckpt_output_dir}")
    print(
        "Next step: evaluate the trained model's tool-use accuracy "
        "against your MCP server."
    )


if __name__ == "__main__":
    main()
