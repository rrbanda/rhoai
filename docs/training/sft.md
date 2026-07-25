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
    model="meta-llama/Llama-3.1-8B-Instruct",
    data="training_data.jsonl",
    output_dir="./sft-output",
    num_epochs=4,
    batch_size=32,
    max_seq_len=4096,
)
```

## Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | required | HuggingFace model ID or local path |
| `data` | str | required | Path to JSONL training data |
| `output_dir` | str | required | Where to save the trained model |
| `num_epochs` | int | `4` | Number of training epochs |
| `batch_size` | int | `32` | Effective batch size (across all GPUs) |
| `max_seq_len` | int | `4096` | Maximum sequence length |
| `lr` | float | `2e-5` | Learning rate |
| `warmup_ratio` | float | `0.1` | Warmup proportion of total steps |
| `gradient_accumulation_steps` | int | auto | Steps to accumulate before update |
| `chat_template` | str | auto | Override chat template format |
| `unmask_input` | bool | `False` | Train on input tokens too (not just output) |

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

By default, loss is computed only on assistant turns. Set `unmask_input=True` to train on user turns as well (useful for continued pretraining).

## GPU Requirements

| Model Size | Min GPUs | Recommended |
|-----------|----------|-------------|
| 3B (Phi, Ministral) | 1x A100 80GB | 2x A100 80GB |
| 7-8B (Llama, Qwen) | 2x A100 80GB | 4x A100 80GB |
| 20B (GPT-OSS) | 4x A100 80GB | 8x A100 80GB |

!!! tip "Estimate Before You Train"
    Use the [Memory Estimator](../utilities/memory-estimator.md) to calculate exact VRAM requirements for your model and batch size before launching a training job.

## Multi-GPU Training

SFT automatically uses all available GPUs via FSDP (Fully Sharded Data Parallelism):

```bash
# Training on 4 GPUs happens automatically
torchrun --nproc_per_node=4 -m training_hub.sft \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data training_data.jsonl \
    --output_dir ./sft-output
```

## MLflow Integration

Track training metrics with MLflow:

```python
from training_hub import sft

sft(
    model="meta-llama/Llama-3.1-8B-Instruct",
    data="training_data.jsonl",
    output_dir="./sft-output",
    mlflow_tracking_uri="http://mlflow.example.com:5000",
    mlflow_experiment_name="sft-knowledge-tuning",
)
```

## Related

- [OSFT](osft.md) — If you need to preserve base knowledge
- [LoRA](lora.md) — If you're memory constrained
- [Knowledge Tuning Pipeline](../end-to-end/knowledge-tuning.md) — Full end-to-end example with SFT
- [Medical Domain](../domains/medical.md) — Medical fine-tuning example comparing SFT, OSFT, and LoRA
