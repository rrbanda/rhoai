# Model Interpolation

Model interpolation (checkpoint blending) creates a new model by combining the weights of two models. This can improve performance by averaging the strengths of a base model and a fine-tuned variant.

## When to Use

- You want to **blend** a fine-tuned model with its base model to balance domain expertise and general capabilities
- You want to **smooth out** training noise by controlling the interpolation weight
- You're doing **checkpoint averaging** for more robust predictions

## Usage

Model interpolation uses the `interpolate_models` helper, which works directly with HuggingFace `transformers`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def interpolate_models(
    model_path: str,
    trained_model_path: str,
    trained_model_weight: float = 0.5,
    output_model_path: str = "./interpolated",
    torch_dtype: str = "bfloat16",
) -> str:
    """Blend base and trained model weights."""
    dtype = getattr(torch, torch_dtype)

    base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    trained = AutoModelForCausalLM.from_pretrained(trained_model_path, torch_dtype=dtype)

    for name, param in base.named_parameters():
        trained_param = dict(trained.named_parameters())[name]
        param.data = (
            (1 - trained_model_weight) * param.data
            + trained_model_weight * trained_param.data
        )

    base.save_pretrained(output_model_path)
    AutoTokenizer.from_pretrained(model_path).save_pretrained(output_model_path)
    return output_model_path
```

!!! note
    This function is provided as a utility script in the repository, not as part of the `training_hub` package. See [`04-training/utilities/model_interpolation.py`](https://github.com/rrbanda/rhoai/blob/main/04-training/utilities/model_interpolation.py).

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | Path to the base (pre-trained) model |
| `trained_model_path` | str | required | Path to the fine-tuned model |
| `trained_model_weight` | float | `0.5` | Blend weight for the trained model (0.0-1.0) |
| `output_model_path` | str | `"./interpolated"` | Where to save the blended model |
| `torch_dtype` | str | `"bfloat16"` | Data type for loading models |

## The `trained_model_weight` Parameter

| Value | Result |
|-------|--------|
| `0.0` | 100% base model (no fine-tuning effect) |
| `0.3` | Light blend — mostly base with some fine-tuning |
| `0.5` | Equal blend |
| `0.7` | Strong blend — mostly fine-tuned |
| `1.0` | 100% fine-tuned model |

!!! tip "Finding the Best Blend"
    Try multiple weight values (0.3, 0.5, 0.7) and evaluate each on your test set. The optimal blend is task-dependent.

## Example: Balancing Domain and General Knowledge

Blend a fine-tuned medical model with the original base to retain more general capabilities:

```python
output = interpolate_models(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    trained_model_path="./medical-sft/hf_format/samples_0",
    trained_model_weight=0.6,
    output_model_path="./medical-blended",
)
```

## Limitations

- Only works with models of the **same architecture** (can't blend Llama with Qwen)
- Works best when the trained model was fine-tuned from the **same base model**
- Not applicable to LoRA adapters (merge the adapter first with `model.merge_and_unload()`)

## Related

- [Plot Loss](plot-loss.md) — Compare training runs before interpolating
- [OSFT](../training/osft.md) — Alternative approach to preserving base capabilities
- [Model-Specific Configs](../domains/model-specific.md) — Per-model training parameters
