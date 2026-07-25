# LoRA / QLoRA

Low-Rank Adaptation (LoRA) adds small trainable adapter matrices to the model while keeping the original weights frozen. This reduces memory usage dramatically — you can fine-tune a 7B model on a **single A100 or L40 GPU**.

## When to Use LoRA

- You have **limited GPU memory** (single GPU)
- You want **fast training iterations** for experimentation
- You need **multiple task-specific adapters** for the same base model
- You want to try **QLoRA** (4-bit quantized base model + LoRA adapters) for even lower memory

## Quick Start

```python
from training_hub import lora_sft

lora_sft(
    model="meta-llama/Llama-3.1-8B-Instruct",
    data="training_data.jsonl",
    output_dir="./lora-output",
    num_epochs=4,
    lora_r=16,
    lora_alpha=32,
)
```

## Key Parameters: `lora_r` and `lora_alpha`

| Parameter | What it does | Guidance |
|-----------|-------------|----------|
| `lora_r` | Rank of the adapter matrices. Higher = more capacity | Start with 16, increase to 64 for complex tasks |
| `lora_alpha` | Scaling factor. Controls how much the adapter affects output | Typically set to `2 * lora_r` |
| `lora_dropout` | Dropout on adapter weights | `0.05` for regularization |

!!! tip "Rule of Thumb"
    Set `lora_alpha = 2 * lora_r`. For most tasks, `lora_r=16, lora_alpha=32` works well. Increase `lora_r` to 64 for complex domain adaptation.

## QLoRA — 4-bit Quantized Training

QLoRA quantizes the base model to 4-bit precision, reducing memory usage by ~4x while maintaining training quality:

```python
from training_hub import lora_sft

lora_sft(
    model="meta-llama/Llama-3.1-8B-Instruct",
    data="training_data.jsonl",
    output_dir="./qlora-output",
    num_epochs=4,
    lora_r=16,
    lora_alpha=32,
    quantize=True,  # Enable QLoRA
)
```

## Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | required | HuggingFace model ID or local path |
| `data` | str | required | Path to JSONL training data |
| `output_dir` | str | required | Where to save the adapter |
| `num_epochs` | int | `4` | Number of training epochs |
| `lora_r` | int | `16` | Rank of LoRA adapter matrices |
| `lora_alpha` | int | `32` | LoRA scaling factor |
| `lora_dropout` | float | `0.05` | Dropout for LoRA layers |
| `batch_size` | int | `32` | Effective batch size |
| `max_seq_len` | int | `4096` | Maximum sequence length |
| `lr` | float | `2e-4` | Learning rate (higher than SFT) |
| `quantize` | bool | `False` | Enable QLoRA (4-bit base model) |
| `target_modules` | list | auto | Which layers to apply LoRA to |

## GPU Requirements

| Model Size | LoRA | QLoRA |
|-----------|------|-------|
| 3B | 1x L40 48GB | 1x T4 16GB |
| 7-8B | 1x A100 80GB | 1x L40 48GB |
| 20B | 2x A100 80GB | 1x A100 80GB |

## Adapter Merging

After training, you can merge the LoRA adapter back into the base model for simplified deployment:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base, "./lora-output")
merged = model.merge_and_unload()
merged.save_pretrained("./merged-model")
```

Or serve the adapter directly with vLLM, which supports loading LoRA adapters at runtime.

## Related

- [SFT](sft.md) — If you have sufficient GPU resources and want maximum capacity
- [GRPO](grpo.md) — LoRA + reinforcement learning for tool-use
- [Medical Domain](../domains/medical.md) — LoRA fine-tuning for medical data
- [GPU Requirements](../reference/gpu-requirements.md) — Full GPU comparison table
