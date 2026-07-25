# Model Interpolation

Model interpolation (checkpoint blending) creates a new model by combining the weights of two or more trained models. This can improve performance by averaging the strengths of different training runs.

## When to Use

- You have **multiple training runs** with different hyperparameters and want to combine their strengths
- You want to **smooth out** the noise from a single training run
- You're doing **checkpoint averaging** for more robust predictions

## Usage

```python
from training_hub import interpolate

interpolate(
    model_a="./sft-run1",
    model_b="./sft-run2",
    output_dir="./interpolated-model",
    alpha=0.5,
)
```

## The `alpha` Parameter

| Value | Result |
|-------|--------|
| `0.0` | 100% model A |
| `0.5` | Equal blend of A and B |
| `1.0` | 100% model B |
| `0.7` | 70% model B, 30% model A |

!!! tip "Finding the Best Blend"
    Try multiple `alpha` values (0.3, 0.5, 0.7) and evaluate each on your test set. The optimal blend is task-dependent.

## Use Cases

### Combining Domain Experts

Blend a medical expert and a legal expert into a multi-domain model:

```python
from training_hub import interpolate

interpolate(
    model_a="./medical-sft",
    model_b="./legal-sft",
    output_dir="./multi-domain",
    alpha=0.5,
)
```

### Checkpoint Averaging

Average the last N checkpoints from a single run for smoother predictions:

```python
from training_hub import interpolate

interpolate(
    model_a="./output/checkpoint-1000",
    model_b="./output/checkpoint-1500",
    output_dir="./averaged",
    alpha=0.5,
)
```

## Limitations

- Only works with models of the **same architecture** (can't blend Llama with Qwen)
- Works best when models were trained from the **same base model**
- Not applicable to LoRA adapters (merge adapters first)

## Related

- [Plot Loss](plot-loss.md) — Compare training runs before interpolating
- [OSFT](../training/osft.md) — Alternative approach to multi-domain models via continual learning
- [Model-Specific Configs](../domains/model-specific.md) — Per-model training parameters
