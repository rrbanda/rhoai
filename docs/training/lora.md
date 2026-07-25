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
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="training_data.jsonl",
    ckpt_output_dir="./lora-output",
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
| `lora_dropout` | Dropout on adapter weights | Optional, for regularization |

!!! tip "Rule of Thumb"
    Set `lora_alpha = 2 * lora_r`. For most tasks, `lora_r=16, lora_alpha=32` works well. Increase `lora_r` to 64 for complex domain adaptation.

## QLoRA — 4-bit Quantized Training

QLoRA quantizes the base model to 4-bit precision, reducing memory usage by ~4x while maintaining training quality:

```python
from training_hub import lora_sft

lora_sft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="training_data.jsonl",
    ckpt_output_dir="./qlora-output",
    num_epochs=4,
    lora_r=16,
    lora_alpha=32,
    load_in_4bit=True,
)
```

## Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | HuggingFace model ID or local path |
| `data_path` | str | required | Path to JSONL training data |
| `ckpt_output_dir` | str | required | Where to save the adapter |
| `num_epochs` | int | None | Number of training epochs |
| `lora_r` | int | None | Rank of LoRA adapter matrices |
| `lora_alpha` | int | None | LoRA scaling factor |
| `lora_dropout` | float | None | Dropout for LoRA layers |
| `effective_batch_size` | int | None | Effective batch size |
| `micro_batch_size` | int | None | Per-device batch size |
| `max_seq_len` | int | None | Maximum sequence length |
| `learning_rate` | float | None | Learning rate |
| `load_in_4bit` | bool | None | Enable QLoRA (4-bit base model) |
| `load_in_8bit` | bool | None | Enable 8-bit quantization |
| `target_modules` | list | None | Which layers to apply LoRA to |
| `nproc_per_node` | int/str | None | Number of GPUs |

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
