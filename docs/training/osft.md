# Orthogonal Subspace Fine-Tuning (OSFT)

OSFT constrains weight updates to an orthogonal subspace of the pretrained weights, allowing the model to learn new knowledge **without forgetting** existing capabilities. It's the recommended algorithm when you need to add domain expertise while preserving the model's general-purpose abilities.

## When to Use OSFT

- You need to **add domain knowledge** (medical, legal, financial) to a model
- The model must **retain its general capabilities** after training
- You plan to do **continual learning** (training on new data over time)
- You have **2+ A100 80GB GPUs** available

## Key Concept: `unfreeze_rank_ratio`

The `unfreeze_rank_ratio` parameter controls the trade-off between learning new knowledge and preserving existing capabilities:

| Value | Effect | Use Case |
|-------|--------|----------|
| `0.005` | Very conservative — preserves almost all base knowledge | Light knowledge injection |
| `0.01` | Balanced — good default for most tasks | Domain adaptation |
| `0.05` | Aggressive — learns more but may drift from base | Heavy domain specialization |
| `0.1` | Very aggressive — close to full SFT behavior | Maximum learning capacity |

!!! tip
    Start with `unfreeze_rank_ratio=0.01` and increase if the model doesn't learn enough, or decrease if general capabilities degrade.

## Quick Start

```python
from training_hub import osft

osft(
    model="meta-llama/Llama-3.1-8B-Instruct",
    data="training_data.jsonl",
    output_dir="./osft-output",
    num_epochs=4,
    batch_size=32,
    max_seq_len=4096,
    unfreeze_rank_ratio=0.01,
)
```

## Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | required | HuggingFace model ID or local path |
| `data` | str | required | Path to JSONL training data |
| `output_dir` | str | required | Where to save the trained model |
| `num_epochs` | int | `4` | Number of training epochs |
| `batch_size` | int | `32` | Effective batch size |
| `max_seq_len` | int | `4096` | Maximum sequence length |
| `lr` | float | `2e-5` | Learning rate |
| `unfreeze_rank_ratio` | float | `0.01` | Controls learning vs preservation |
| `warmup_ratio` | float | `0.1` | Warmup proportion |
| `chat_template` | str | auto | Override chat template format |

## Continual Learning

OSFT is uniquely suited for continual learning — training on new data batches over time without forgetting previous training:

```python
from training_hub import osft

# Phase 1: Train on medical data
osft(
    model="meta-llama/Llama-3.1-8B-Instruct",
    data="medical_data.jsonl",
    output_dir="./phase1",
    unfreeze_rank_ratio=0.01,
)

# Phase 2: Add legal knowledge to the same model
osft(
    model="./phase1",
    data="legal_data.jsonl",
    output_dir="./phase2",
    unfreeze_rank_ratio=0.01,
)
```

The orthogonal constraint ensures Phase 2 training doesn't overwrite what was learned in Phase 1. See the [Continual Learning](continual-learning.md) guide for a detailed walkthrough.

## GPU Requirements

| Model Size | Min GPUs | Recommended |
|-----------|----------|-------------|
| 3B (Phi, Ministral) | 1x A100 80GB | 2x A100 80GB |
| 7-8B (Llama, Qwen) | 2x A100 80GB | 4x A100 80GB |
| 20B (GPT-OSS) | 4x A100 80GB | 8x A100 80GB |

## Related

- [SFT](sft.md) — If you don't need knowledge preservation
- [Continual Learning](continual-learning.md) — Multi-phase OSFT training
- [Medical Domain](../domains/medical.md) — OSFT for medical knowledge
- [Knowledge Tuning Pipeline](../end-to-end/knowledge-tuning.md) — Full pipeline with OSFT
