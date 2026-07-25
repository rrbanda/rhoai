# Model-Specific Configurations

Tuned hyperparameters for each supported model architecture. These configurations have been tested and optimized for knowledge tuning tasks.

## Llama 3.1 8B

Meta's flagship open model. Strong general capabilities and well-supported across tooling.

```python
from training_hub import sft

sft(
    model="meta-llama/Llama-3.1-8B-Instruct",
    data="training_data.jsonl",
    output_dir="./llama-output",
    num_epochs=4,
    batch_size=32,
    max_seq_len=4096,
    lr=2e-5,
    warmup_ratio=0.1,
)
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max seq len | 4096 | Supports up to 128K, but 4K is efficient for training |
| Batch size | 32 | Adjust based on GPU count |
| Learning rate | 2e-5 | Standard for 8B models |
| GPU requirement | 2x A100 80GB (SFT) | 1x A100 for LoRA |

## Qwen 2.5 7B

Strong multilingual and reasoning capabilities. Good for non-English domains.

```python
from training_hub import sft

sft(
    model="Qwen/Qwen2.5-7B-Instruct",
    data="training_data.jsonl",
    output_dir="./qwen-output",
    num_epochs=4,
    batch_size=32,
    max_seq_len=4096,
    lr=2e-5,
    warmup_ratio=0.1,
)
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max seq len | 4096 | Supports up to 32K |
| Batch size | 32 | |
| Learning rate | 2e-5 | |
| GPU requirement | 2x A100 80GB (SFT) | Similar to Llama 8B |

## Phi 4 Mini (~3.8B)

Microsoft's dense, efficient model. Best cost-performance ratio for many tasks.

```python
from training_hub import sft

sft(
    model="microsoft/Phi-4-mini-instruct",
    data="training_data.jsonl",
    output_dir="./phi-output",
    num_epochs=5,
    batch_size=64,
    max_seq_len=4096,
    lr=3e-5,
    warmup_ratio=0.1,
)
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max seq len | 4096 | |
| Batch size | 64 | Smaller model allows larger batch |
| Learning rate | 3e-5 | Slightly higher than 7-8B models |
| GPU requirement | 1x A100 80GB (SFT) | 1x L40 for LoRA |

!!! tip "Best for Constrained Deployments"
    Phi 4 Mini delivers strong performance at 3.8B parameters. Fine-tuned, it can match larger models on domain tasks while being deployable on cheaper hardware.

## Granite 3.3 / 4.0

Red Hat / IBM's enterprise-focused models. Optimized for enterprise workloads.

```python
from training_hub import sft

sft(
    model="ibm-granite/granite-3.3-8b-instruct",
    data="training_data.jsonl",
    output_dir="./granite-output",
    num_epochs=4,
    batch_size=32,
    max_seq_len=4096,
    lr=2e-5,
    warmup_ratio=0.1,
)
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max seq len | 4096 | Supports 8K+ |
| Batch size | 32 | |
| Learning rate | 2e-5 | |
| GPU requirement | 2x A100 80GB (SFT) | |

## GPT-OSS 20B

Large-scale open model. Use when maximum capacity is needed and GPU resources are available.

```python
from training_hub import sft

sft(
    model="gpt-oss/gpt-oss-20b",
    data="training_data.jsonl",
    output_dir="./gpt-oss-output",
    num_epochs=3,
    batch_size=16,
    max_seq_len=4096,
    lr=1e-5,
    warmup_ratio=0.1,
)
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max seq len | 4096 | |
| Batch size | 16 | Smaller batch due to model size |
| Learning rate | 1e-5 | Lower for larger models |
| GPU requirement | 4x A100 80GB (SFT) | 2x A100 for LoRA |

## Ministral 3B

Compact Mistral model for highly constrained environments.

```python
from training_hub import sft

sft(
    model="mistralai/Ministral-3B-Instruct",
    data="training_data.jsonl",
    output_dir="./ministral-output",
    num_epochs=5,
    batch_size=64,
    max_seq_len=4096,
    lr=3e-5,
)
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max seq len | 4096 | |
| Batch size | 64 | Small model, large batch |
| Learning rate | 3e-5 | |
| GPU requirement | 1x A100 80GB (SFT) | 1x L40 for LoRA |

## GPU Summary

| Model | SFT | OSFT | LoRA | QLoRA |
|-------|-----|------|------|-------|
| Ministral 3B | 1x A100 | 1x A100 | 1x L40 | 1x T4 |
| Phi 4 Mini 3.8B | 1x A100 | 1x A100 | 1x L40 | 1x T4 |
| Qwen 2.5 7B | 2x A100 | 2x A100 | 1x A100 | 1x L40 |
| Llama 3.1 8B | 2x A100 | 2x A100 | 1x A100 | 1x L40 |
| Granite 3.3 8B | 2x A100 | 2x A100 | 1x A100 | 1x L40 |
| GPT-OSS 20B | 4x A100 | 4x A100 | 2x A100 | 1x A100 |

## Related

- [Choosing an Algorithm](../getting-started/choosing-an-algorithm.md) — Pick the right algorithm for your use case
- [Memory Estimator](../utilities/memory-estimator.md) — Calculate exact VRAM requirements
- [GPU Requirements](../reference/gpu-requirements.md) — Full GPU comparison
