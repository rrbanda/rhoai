# Memory Estimator

Estimate GPU VRAM requirements before launching a training job. This avoids costly OOM (out-of-memory) errors by calculating the memory footprint of your model, optimizer, and data.

## Quick Estimate

```python
from training_hub import estimate

result = estimate(
    model="meta-llama/Llama-3.1-8B-Instruct",
    batch_size=32,
    max_seq_len=4096,
    method="sft",
)

print(f"Estimated VRAM: {result['total_gb']:.1f} GB")
print(f"Recommended GPUs: {result['recommended_gpus']}")
```

## How Memory is Calculated

Training memory consists of four components:

| Component | Formula | Example (8B model) |
|-----------|---------|-------------------|
| Model weights | params × dtype_size | 8B × 2 bytes = 16 GB |
| Optimizer states | params × 8 bytes (AdamW) | 8B × 8 = 64 GB |
| Gradients | params × dtype_size | 8B × 2 = 16 GB |
| Activations | batch × seq_len × hidden × layers | ~20 GB |

**Total for 8B SFT:** ~116 GB (≈ 2x A100 80GB)

### LoRA Reduces Memory

With LoRA, only adapter parameters need optimizer states and gradients:

| Component | LoRA (r=16) |
|-----------|------------|
| Base model weights | 16 GB (frozen, loaded in fp16) |
| Adapter parameters | ~0.1 GB |
| Optimizer states | ~0.8 GB (adapters only) |
| Activations | ~10 GB |

**Total for 8B LoRA:** ~27 GB (≈ 1x A100 80GB)

### QLoRA Further Reduces Memory

QLoRA quantizes the base model to 4-bit:

| Component | QLoRA (r=16) |
|-----------|-------------|
| Base model weights | 4 GB (4-bit quantized) |
| Adapter parameters | ~0.1 GB |
| Optimizer states | ~0.8 GB |
| Activations | ~10 GB |

**Total for 8B QLoRA:** ~15 GB (≈ 1x L40 48GB or T4 16GB)

## Parameter Effects

| Parameter change | Memory impact |
|-----------------|---------------|
| Batch size ×2 | Activations ×2 |
| Sequence length ×2 | Activations ×2 |
| LoRA rank ×2 | Adapter memory ×2 (negligible) |
| Model size ×2 | Everything ×2 |

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
