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
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="training_data.jsonl",
    ckpt_output_dir="./osft-output",
    unfreeze_rank_ratio=0.01,
    effective_batch_size=32,
    max_tokens_per_gpu=16384,
    max_seq_len=4096,
    learning_rate=2e-5,
    num_epochs=4,
)
```

## Parameter Reference

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_path` | str | Yes | — | HuggingFace model ID or local path |
| `data_path` | str | Yes | — | Path to JSONL training data |
| `ckpt_output_dir` | str | Yes | — | Where to save the trained model |
| `unfreeze_rank_ratio` | float | Yes | — | Controls learning vs preservation trade-off |
| `effective_batch_size` | int | Yes | — | Effective batch size |
| `max_tokens_per_gpu` | int | Yes | — | Token budget per GPU per step |
| `max_seq_len` | int | Yes | — | Maximum sequence length |
| `learning_rate` | float | Yes | — | Learning rate |
| `num_epochs` | int | No | None | Number of training epochs |
| `warmup_steps` | int | No | None | Number of warmup steps |
| `use_liger` | bool | No | None | Use Liger kernel for efficiency |
| `unmask_messages` | bool | No | None | Train on all message roles |
| `is_pretraining` | bool | No | None | Enable continued pretraining mode |
| `checkpoint_at_epoch` | bool | No | None | Save checkpoint at each epoch |
| `nproc_per_node` | int/str | No | None | Number of GPUs |

!!! warning "Required Parameters"
    Unlike SFT, OSFT requires you to explicitly set `unfreeze_rank_ratio`, `effective_batch_size`, `max_tokens_per_gpu`, `max_seq_len`, and `learning_rate`. These have no defaults.

## Continual Learning

OSFT is uniquely suited for continual learning — training on new data batches over time without forgetting previous training:

```python
from training_hub import osft

osft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="medical_data.jsonl",
    ckpt_output_dir="./phase1",
    unfreeze_rank_ratio=0.01,
    effective_batch_size=32,
    max_tokens_per_gpu=16384,
    max_seq_len=4096,
    learning_rate=2e-5,
)

osft(
    model_path="./phase1/hf_format/samples_0",
    data_path="legal_data.jsonl",
    ckpt_output_dir="./phase2",
    unfreeze_rank_ratio=0.01,
    effective_batch_size=32,
    max_tokens_per_gpu=16384,
    max_seq_len=4096,
    learning_rate=2e-5,
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
