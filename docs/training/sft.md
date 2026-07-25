# Supervised Fine-Tuning (SFT)

SFT updates all model parameters to learn from your training data. It provides the highest learning capacity but requires significant GPU resources and can cause catastrophic forgetting of base model capabilities.

## When to Use SFT

- You have **abundant, high-quality training data** (thousands of examples)
- You want **maximum performance** on your specific domain or task
- You can afford **2+ A100 80GB GPUs**
- You **don't need** the model to retain strong general capabilities

!!! warning "Catastrophic Forgetting"
    SFT updates all model weights, which means the model may lose general capabilities it had before training. If you need to preserve base knowledge, use [OSFT](osft.md) instead.

## Quick Start

```python
from training_hub import sft

sft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="training_data.jsonl",
    ckpt_output_dir="./sft-output",
    num_epochs=4,
    effective_batch_size=32,
    max_seq_len=4096,
)
```

## Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | HuggingFace model ID or local path |
| `data_path` | str | required | Path to JSONL training data |
| `ckpt_output_dir` | str | required | Where to save the trained model |
| `num_epochs` | int | None | Number of training epochs |
| `effective_batch_size` | int | None | Effective batch size (across all GPUs) |
| `max_seq_len` | int | None | Maximum sequence length |
| `learning_rate` | float | None | Learning rate |
| `warmup_steps` | int | None | Number of warmup steps |
| `max_tokens_per_gpu` | int | None | Token budget per GPU per step |
| `is_pretraining` | bool | None | Enable continued pretraining mode (trains on all tokens) |
| `block_size` | int | None | Document packing block size (for pretraining) |
| `checkpoint_at_epoch` | bool | None | Save checkpoint at each epoch boundary |
| `nproc_per_node` | int/str | None | Number of GPUs to use |
| `mlflow_tracking_uri` | str | None | MLflow server URI for tracking |
| `mlflow_experiment_name` | str | None | MLflow experiment name |

## Data Format

SFT expects JSONL files with the messages format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is RHOAI?"},
    {"role": "assistant", "content": "Red Hat OpenShift AI is..."}
  ]
}
```

By default, loss is computed only on assistant turns. Set `is_pretraining=True` to train on all tokens (useful for continued pretraining on raw text).

## GPU Requirements

| Model Size | Min GPUs | Recommended |
|-----------|----------|-------------|
| 3B (Phi, Ministral) | 1x A100 80GB | 2x A100 80GB |
| 7-8B (Llama, Qwen) | 2x A100 80GB | 4x A100 80GB |
| 20B (GPT-OSS) | 4x A100 80GB | 8x A100 80GB |

!!! tip "Estimate Before You Train"
    Use the [Memory Estimator](../utilities/memory-estimator.md) to calculate exact VRAM requirements for your model and batch size before launching a training job.

## Multi-GPU Training

SFT automatically uses all available GPUs via FSDP (Fully Sharded Data Parallelism). Control the number of GPUs with `nproc_per_node`:

```python
from training_hub import sft

sft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="training_data.jsonl",
    ckpt_output_dir="./sft-output",
    nproc_per_node=4,
)
```

## MLflow Integration

Track training metrics with MLflow:

```python
from training_hub import sft

sft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="training_data.jsonl",
    ckpt_output_dir="./sft-output",
    mlflow_tracking_uri="http://mlflow.example.com:5000",
    mlflow_experiment_name="sft-knowledge-tuning",
)
```

## Related

- [OSFT](osft.md) — If you need to preserve base knowledge
- [LoRA](lora.md) — If you're memory constrained
- [Knowledge Tuning Pipeline](../end-to-end/knowledge-tuning.md) — Full end-to-end example with SFT
- [Medical Domain](../domains/medical.md) — Medical fine-tuning example comparing SFT, OSFT, and LoRA
