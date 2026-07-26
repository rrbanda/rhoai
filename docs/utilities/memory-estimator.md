# Memory Estimator

Estimate GPU VRAM requirements before launching a training job. This avoids costly OOM (out-of-memory) errors by calculating the memory footprint of your model, optimizer, and data.

## Quick Estimate

```python
from training_hub import estimate

lower, expected, upper = estimate(
    training_method="sft",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    num_gpus=2,
    gpu_memory=85_899_345_920,  # 80 GB in bytes
)

print(f"Estimated VRAM per GPU: {expected / 1e9:.1f} GB")
print(f"Range: {lower / 1e9:.1f} - {upper / 1e9:.1f} GB")
```

The function returns a **3-tuple of integers** `(lower_bound, expected, upper_bound)` in **bytes** (per-GPU VRAM estimate).

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `training_method` | str | `"sft"` | One of `"sft"`, `"osft"`, `"lora"`, `"qlora"` |
| `model_path` | str | `"ibm-granite/granite-3.3-8b-instruct"` | HuggingFace model ID |
| `num_gpus` | int | `8` | Number of GPUs |
| `gpu_memory` | int | `85899345920` | Per-GPU VRAM in bytes (default: 80 GB) |
| `max_tokens_per_gpu` | int | None | Token budget per GPU (SFT/OSFT) |
| `unfreeze_rank_ratio` | float | `0.25` | OSFT unfreeze ratio |
| `lora_r` | int | `32` | LoRA rank (LoRA/QLoRA) |
| `batch_size` | int | None | Batch size (LoRA/QLoRA) |
| `max_seq_len` | int | None | Max sequence length (LoRA/QLoRA) |
| `verbose` | int | `1` | Print detailed breakdown |

## Examples by Method

```python
from training_hub import estimate

# SFT estimate for 8B model on 4 GPUs
lower, expected, upper = estimate(
    training_method="sft",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    num_gpus=4,
    max_tokens_per_gpu=18000,
    verbose=1,
)

# OSFT estimate
lower, expected, upper = estimate(
    training_method="osft",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    num_gpus=4,
    max_tokens_per_gpu=16384,
    unfreeze_rank_ratio=0.25,
)

# LoRA estimate (single GPU)
lower, expected, upper = estimate(
    training_method="lora",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    num_gpus=1,
    lora_r=16,
    batch_size=4,
    max_seq_len=4096,
)

# QLoRA estimate (single GPU, 4-bit)
lower, expected, upper = estimate(
    training_method="qlora",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    num_gpus=1,
    lora_r=16,
    batch_size=4,
    max_seq_len=4096,
)
```

## How Memory is Calculated

Training memory consists of four components:

| Component | Formula | Example (8B model) |
|-----------|---------|-------------------|
| Model weights | params x dtype_size | 8B x 2 bytes = 16 GB |
| Optimizer states | params x 8 bytes (AdamW) | 8B x 8 = 64 GB |
| Gradients | params x dtype_size | 8B x 2 = 16 GB |
| Activations | batch x seq_len x hidden x layers | ~20 GB |

**Total for 8B SFT:** ~116 GB (split across GPUs with FSDP)

### LoRA Reduces Memory

With LoRA, only adapter parameters need optimizer states and gradients:

| Component | LoRA (r=16) |
|-----------|------------|
| Base model weights | 16 GB (frozen, loaded in fp16) |
| Adapter parameters | ~0.1 GB |
| Optimizer states | ~0.8 GB (adapters only) |
| Activations | ~10 GB |

**Total for 8B LoRA:** ~27 GB (1x A100 80GB)

## Quick Reference Table

| Model | SFT | OSFT | LoRA | QLoRA |
|-------|-----|------|------|-------|
| Ministral 3B | 1x A100 | 1x A100 | 1x L40 | 1x T4 |
| Phi 4 Mini 3.8B | 1x A100 | 1x A100 | 1x L40 | 1x T4 |
| Qwen 2.5 7B | 2x A100 | 2x A100 | 1x A100 | 1x L40 |
| Llama 3.1 8B | 2x A100 | 2x A100 | 1x A100 | 1x L40 |
| Granite 3.3 8B | 2x A100 | 2x A100 | 1x A100 | 1x L40 |
| GPT-OSS 20B | 4x A100 | 4x A100 | 2x A100 | 1x A100 |

## Related

- [GPU Requirements](../reference/gpu-requirements.md) — Full GPU comparison table
- [LoRA](../training/lora.md) — Memory-efficient training
- [Model-Specific Configs](../domains/model-specific.md) — Per-model hyperparameters
