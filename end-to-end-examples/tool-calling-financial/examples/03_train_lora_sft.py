"""Fine-tune a model with LoRA SFT using Training Hub.

Uses the chat-format JSONL from step 02 to fine-tune a student model
(e.g., Qwen3-4B) with LoRA (Low-Rank Adaptation) + SFT (Supervised Fine-Tuning).
The model learns to produce correct tool-calling responses by training on expert
demonstrations of financial tool usage.

Two data source options:
  1. Local JSONL from MCP distillation (steps 01 + 02): financial-domain-specific
  2. HuggingFace dataset (e.g., LipengCS/Table-GPT:All): general fallback

This script runs training locally via training_hub.lora_sft().
For production on RHOAI, use 03b_train_kfp_pipeline.py which wraps the same
algorithm in a Kubeflow Pipeline with dataset download, evaluation, and model
registry stages.

Defaults are tuned for a single NVIDIA L4 (24GB VRAM):
  - LoRA rank 16 (parameter-efficient)
  - QLoRA 4-bit quantization (fits on 24GB)
  - 2 epochs (standard for LoRA)
  - Learning rate 2e-4 (recommended for LoRA)

Usage:
    # With local SDG data (from steps 01+02):
    python 03_train_lora_sft.py

    # With a HuggingFace dataset:
    python 03_train_lora_sft.py --data-path hf://LipengCS/Table-GPT:All

    # Full capacity on A100:
    python 03_train_lora_sft.py --lora-r 32 --lora-alpha 64 --max-seq-len 8192

    # Multi-GPU data-parallel:
    torchrun --nproc-per-node=2 03_train_lora_sft.py --lora-r 32
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a model with LoRA SFT via Training Hub"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Base model to fine-tune (default: STUDENT_MODEL env or Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training JSONL or HuggingFace dataset URI "
        "(default: generated_data/training_data.jsonl)",
    )
    parser.add_argument(
        "--ckpt-output-dir",
        type=str,
        default=None,
        help="Directory for checkpoints (default: CHECKPOINT_DIR env or ./checkpoints)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank — higher captures more info, uses more memory (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA scaling factor, typically 2x lora_r (default: 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout rate (default: 0.0)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4, recommended for LoRA)",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=128,
        help="Effective batch size per optimizer step (default: 128)",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=2,
        help="Micro batch size per GPU (default: 2)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length in tokens (default: 4096 for L4; use 8192 for A100)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="Enable QLoRA 4-bit quantization (default: True)",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (use full precision)",
    )
    parser.add_argument(
        "--use-liger",
        action="store_true",
        default=True,
        help="Enable Liger kernel optimizations (default: True)",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        help="LR scheduler type: cosine, linear, constant (default: cosine)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--enable-model-splitting",
        action="store_true",
        help="Split model across GPUs (for very large models that don't fit on 1 GPU)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    student_model = args.model_path or os.environ.get("STUDENT_MODEL", "Qwen/Qwen3-4B")
    ckpt_output_dir = args.ckpt_output_dir or os.environ.get("CHECKPOINT_DIR", "./checkpoints")

    if args.data_path:
        data_path = args.data_path
    else:
        output_dir = os.environ.get("OUTPUT_DIR", "./generated_data")
        local_path = os.path.join(output_dir, "training_data.jsonl")
        if Path(local_path).exists():
            data_path = local_path
        else:
            print(f"ERROR: Training data not found at {local_path}")
            print("Run steps 01 + 02 first, or specify --data-path.")
            print("For the KFP pipeline approach (downloads data automatically), use 03b_train_kfp_pipeline.py.")
            sys.exit(1)

    n_examples = "?"
    if not data_path.startswith("hf://"):
        with open(data_path) as f:
            n_examples = sum(1 for _ in f)

    load_in_4bit = args.load_in_4bit and not args.no_4bit

    print("=" * 60)
    print("Tool-Calling Financial Model — LoRA SFT Training")
    print("=" * 60)
    print(f"  Base model:       {student_model}")
    print(f"  Training data:    {data_path} ({n_examples} examples)")
    print(f"  Checkpoint dir:   {ckpt_output_dir}")
    print(f"  LoRA rank:        {args.lora_r}")
    print(f"  LoRA alpha:       {args.lora_alpha}")
    print(f"  LoRA dropout:     {args.lora_dropout}")
    print(f"  Epochs:           {args.num_epochs}")
    print(f"  Learning rate:    {args.learning_rate}")
    print(f"  Batch size:       {args.effective_batch_size}")
    print(f"  Micro batch:      {args.micro_batch_size}")
    print(f"  Max seq length:   {args.max_seq_len}")
    print(f"  QLoRA 4-bit:      {load_in_4bit}")
    print(f"  LR scheduler:     {args.lr_scheduler}")
    print(f"  Seed:             {args.seed}")
    if args.enable_model_splitting:
        print(f"  Model splitting:  enabled")
    print("=" * 60)

    from training_hub import lora_sft

    train_kwargs = {
        "model_path": student_model,
        "data_path": data_path,
        "ckpt_output_dir": ckpt_output_dir,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "effective_batch_size": args.effective_batch_size,
        "micro_batch_size": args.micro_batch_size,
        "max_seq_len": args.max_seq_len,
        "load_in_4bit": load_in_4bit,
        "lr_scheduler": args.lr_scheduler,
        "seed": args.seed,
    }

    if args.enable_model_splitting:
        train_kwargs["enable_model_splitting"] = True

    print("\nStarting LoRA SFT training...")
    print("-" * 60)

    result = lora_sft(**train_kwargs)

    print("-" * 60)
    print("Training complete!")

    if isinstance(result, dict):
        for key, value in result.items():
            if key != "model" and key != "tokenizer":
                print(f"  {key}: {value}")
    else:
        print(f"  Result: {result}")

    print(f"\nCheckpoints saved to: {ckpt_output_dir}")
    print(
        "Next step: deploy the fine-tuned model with 04_deploy_model.py"
    )


if __name__ == "__main__":
    main()
