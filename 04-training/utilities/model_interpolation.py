"""Merge two model checkpoints via linear interpolation.

Given a base model and a fine-tuned checkpoint of the same architecture,
produces a merged checkpoint whose weights are a weighted average of the
two.  This is useful for:
- Blending a fine-tuned model with its base to retain more general
  capabilities while still benefiting from domain-specific training.
- Ensembling multiple training runs at the weight level.

Adapted from the Training Hub interpolator example script.

Requirements:
    pip install transformers torch

Usage:
    python model_interpolation.py \\
        --model-path /path/to/base/model \\
        --trained-model-path /path/to/trained/checkpoint

    python model_interpolation.py \\
        --model-path Qwen/Qwen2.5-7B-Instruct \\
        --trained-model-path ./checkpoints/sft/epoch_3 \\
        --trained-model-weight 0.7 \\
        --output-model-path ./merged_model
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def interpolate_models(
    model_path: str,
    trained_model_path: str,
    trained_model_weight: float = 0.5,
    output_model_path: str | None = None,
    torch_dtype: str | torch.dtype | None = "bfloat16",
) -> str:
    """Linearly interpolate between two model checkpoints.

    Args:
        model_path: Path to the base model (HuggingFace ID or local).
        trained_model_path: Path to the fine-tuned model checkpoint.
        trained_model_weight: Weight for the trained model in [0, 1].
            The base model gets weight ``1 - trained_model_weight``.
        output_model_path: Where to save the merged model.  Defaults to
            ``{trained_model_path}_interp``.
        torch_dtype: Data type for loading models.

    Returns:
        Path where the merged model was saved.
    """
    if output_model_path is None:
        output_model_path = f"{trained_model_path}_interp"

    if not 0.0 <= trained_model_weight <= 1.0:
        raise ValueError(
            f"trained_model_weight must be in [0, 1], got {trained_model_weight}"
        )

    model_kwargs: dict = {}
    if torch_dtype is not None:
        if isinstance(torch_dtype, str):
            low = torch_dtype.lower()
            if low == "auto":
                model_kwargs["torch_dtype"] = "auto"
            elif low in _DTYPE_MAP:
                model_kwargs["torch_dtype"] = _DTYPE_MAP[low]
            else:
                raise ValueError(f"Unsupported torch-dtype: {torch_dtype}")
        else:
            model_kwargs["torch_dtype"] = torch_dtype

    print(f"Loading base model from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    state_dict = model.state_dict()
    base_weight = 1.0 - trained_model_weight
    for key in state_dict:
        state_dict[key] = state_dict[key] * base_weight

    print(f"Loading trained model from {trained_model_path} ...")
    trained_model = AutoModelForCausalLM.from_pretrained(
        trained_model_path, **model_kwargs
    )
    trained_state_dict = trained_model.state_dict()
    for key in state_dict:
        state_dict[key] += trained_state_dict[key] * trained_model_weight

    print(f"Saving merged model to {output_model_path} ...")
    model.save_pretrained(output_model_path, state_dict=state_dict)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_model_path)

    print(f"Merged model saved at {output_model_path}")
    return output_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two model checkpoints via linear interpolation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the base model (HuggingFace ID or local path)",
    )
    parser.add_argument(
        "--trained-model-path",
        required=True,
        help="Path to the fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--trained-model-weight",
        type=float,
        default=0.5,
        help="Weight for the trained model in [0, 1]",
    )
    parser.add_argument(
        "--output-model-path",
        default=None,
        help="Path for the merged output model (default: <trained>_interp)",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32", "auto"],
        help="Torch dtype for loading models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Model Interpolation")
    print("=" * 60)
    print(f"  Base model:        {args.model_path}")
    print(f"  Trained model:     {args.trained_model_path}")
    print(f"  Trained weight:    {args.trained_model_weight}")
    print(f"  Base weight:       {1.0 - args.trained_model_weight}")
    print(f"  Output:            {args.output_model_path or '<trained>_interp'}")
    print(f"  Dtype:             {args.torch_dtype}")
    print("=" * 60)

    try:
        output = interpolate_models(
            model_path=args.model_path,
            trained_model_path=args.trained_model_path,
            trained_model_weight=args.trained_model_weight,
            output_model_path=args.output_model_path,
            torch_dtype=args.torch_dtype,
        )
    except Exception as exc:
        print(f"\nInterpolation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. Merged model at: {output}")


if __name__ == "__main__":
    main()
