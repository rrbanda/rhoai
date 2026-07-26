# Model Interpolation

Model interpolation (checkpoint blending) creates a new model by combining the weights of two models. This can improve performance by averaging the strengths of a base model and a fine-tuned variant.

## When to Use

- You want to **blend** a fine-tuned model with its base model to balance domain expertise and general capabilities
- You want to **smooth out** training noise by controlling the interpolation weight
- You're doing **checkpoint averaging** for more robust predictions

## Usage

Model interpolation uses the `interpolate_models` helper from [`04-training/utilities/model_interpolation.py`](https://github.com/rrbanda/rhoai/blob/main/04-training/utilities/model_interpolation.py):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def interpolate_models(
    model_path: str,
    trained_model_path: str,
    trained_model_weight: float = 0.5,
    output_model_path: str | None = None,
    torch_dtype: str | torch.dtype | None = "bfloat16",
) -> str:
    """Linearly interpolate between two model checkpoints."""
    if output_model_path is None:
        output_model_path = f"{trained_model_path}_interp"

    if not 0.0 <= trained_model_weight <= 1.0:
        raise ValueError(f"trained_model_weight must be in [0, 1], got {trained_model_weight}")

    # ... dtype handling omitted for brevity ...

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    state_dict = model.state_dict()
    base_weight = 1.0 - trained_model_weight
    for key in state_dict:
        state_dict[key] = state_dict[key] * base_weight

    trained_model = AutoModelForCausalLM.from_pretrained(trained_model_path, torch_dtype=dtype)
    trained_state_dict = trained_model.state_dict()
    for key in state_dict:
        state_dict[key] += trained_state_dict[key] * trained_model_weight

    model.save_pretrained(output_model_path, state_dict=state_dict)
    AutoTokenizer.from_pretrained(model_path).save_pretrained(output_model_path)
    return output_model_path
```

!!! note
    This function is provided as a utility script in the repository, not as part of the `training_hub` package. The upstream version lives at [`training_hub/examples/scripts/interpolator.py`](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/scripts/interpolator.py).

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | Path to the base (pre-trained) model |
| `trained_model_path` | str | required | Path to the fine-tuned model |
| `trained_model_weight` | float | `0.5` | Blend weight for the trained model (0.0-1.0) |
| `output_model_path` | str \| None | `None` → `{trained_model_path}_interp` | Where to save the blended model |
| `torch_dtype` | str \| torch.dtype \| None | `"bfloat16"` | Data type for loading models |

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
