# GPU Requirements

Comprehensive GPU requirements for all model sizes and training algorithms. Use the [Memory Estimator](../utilities/memory-estimator.md) for exact calculations with your specific configuration.

## GPU Comparison

| GPU | VRAM | Interconnect | Typical Use |
|-----|------|-------------|-------------|
| NVIDIA T4 | 16 GB | PCIe | QLoRA on small models |
| NVIDIA L40 | 48 GB | PCIe | LoRA on 7-8B models |
| NVIDIA A100 | 80 GB | NVLink | SFT/OSFT on 7-8B models |
| NVIDIA H100 | 80 GB | NVLink | All methods, faster than A100 |

## Requirements by Algorithm and Model

### SFT / OSFT (Full Parameter)

| Model | Min GPUs | Recommended | Est. VRAM |
|-------|----------|-------------|-----------|
| Ministral 3B | 1x A100 | 2x A100 | ~45 GB |
| Phi 4 Mini 3.8B | 1x A100 | 2x A100 | ~55 GB |
| Qwen 2.5 7B | 2x A100 | 4x A100 | ~110 GB |
| Llama 3.1 8B | 2x A100 | 4x A100 | ~120 GB |
| Granite 3.3 8B | 2x A100 | 4x A100 | ~120 GB |
| GPT-OSS 20B | 4x A100 | 8x A100 | ~300 GB |

### LoRA

| Model | Min GPUs | Recommended | Est. VRAM |
|-------|----------|-------------|-----------|
| Ministral 3B | 1x L40 | 1x A100 | ~20 GB |
| Phi 4 Mini 3.8B | 1x L40 | 1x A100 | ~25 GB |
| Qwen 2.5 7B | 1x A100 | 1x A100 | ~40 GB |
| Llama 3.1 8B | 1x A100 | 1x A100 | ~45 GB |
| Granite 3.3 8B | 1x A100 | 1x A100 | ~45 GB |
| GPT-OSS 20B | 2x A100 | 2x A100 | ~100 GB |

### QLoRA (4-bit)

| Model | Min GPUs | Recommended | Est. VRAM |
|-------|----------|-------------|-----------|
| Ministral 3B | 1x T4 | 1x L40 | ~8 GB |
| Phi 4 Mini 3.8B | 1x T4 | 1x L40 | ~10 GB |
| Qwen 2.5 7B | 1x L40 | 1x A100 | ~15 GB |
| Llama 3.1 8B | 1x L40 | 1x A100 | ~18 GB |
| Granite 3.3 8B | 1x L40 | 1x A100 | ~18 GB |
| GPT-OSS 20B | 1x A100 | 1x A100 | ~35 GB |

### GRPO (LoRA-based RL)

| Model | Min GPUs | Recommended | Est. VRAM |
|-------|----------|-------------|-----------|
| Ministral 3B | 1x A100 | 2x A100 | ~40 GB |
| Phi 4 Mini 3.8B | 1x A100 | 2x A100 | ~50 GB |
| Qwen 2.5 7B | 2x A100 | 4x A100 | ~90 GB |
| Llama 3.1 8B | 2x A100 | 4x A100 | ~100 GB |

!!! note "GRPO Memory"
    GRPO requires more memory than LoRA SFT because it generates multiple candidate responses per example and holds them in memory for advantage computation.

## Batch Size Impact

Memory scales linearly with batch size. If you're close to the VRAM limit:

| Action | Memory saved |
|--------|-------------|
| Halve batch size | ~20-30% |
| Halve sequence length | ~15-25% |
| Enable gradient checkpointing | ~30-40% |
| Switch to QLoRA | ~60-70% |

## Cost Estimation

Approximate training cost on cloud GPU instances (per training run):

| Model + Method | GPU Hours | Est. Cost (A100) |
|----------------|-----------|-------------------|
| 3B SFT, 4 epochs | 2-4 hrs | $6-12 |
| 8B SFT, 4 epochs | 8-16 hrs | $24-48 |
| 8B LoRA, 4 epochs | 2-4 hrs | $6-12 |
| 8B OSFT, 4 epochs | 8-16 hrs | $24-48 |
| 20B SFT, 4 epochs | 24-48 hrs | $72-144 |

!!! tip "Start Small"
    Begin with LoRA on a single GPU for fast iteration. Move to SFT/OSFT once you've validated your data quality and hyperparameters.

## Related

- [Memory Estimator](../utilities/memory-estimator.md) — Calculate exact VRAM for your config
- [Choosing an Algorithm](../getting-started/choosing-an-algorithm.md) — Pick based on GPU constraints
- [Model-Specific Configs](../domains/model-specific.md) — Per-model hyperparameters
