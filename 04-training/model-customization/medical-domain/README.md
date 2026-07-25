# Medical Domain Fine-Tuning

Fine-tune language models on medical Q&A data using three different Training Hub
algorithms, each with distinct trade-offs.

## Overview

These examples train **Ministral 3B** on the
[medalpaca/medical_meadow_medical_flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)
dataset.  The same approach generalises to any model and medical dataset in JSONL
messages format.

| Algorithm | Script | Key benefit | GPU requirement |
|-----------|--------|-------------|-----------------|
| **SFT** | `examples/medical_sft.py` | Simplest, full-parameter training | 8x A100 40 GB |
| **OSFT** | `examples/medical_osft.py` | Preserves base capabilities (no catastrophic forgetting) | 8x A100 40 GB |
| **LoRA** | `examples/medical_lora.py` | Parameter-efficient, works on a single GPU | 1x A100 40 GB (QLoRA: 1x A10 24 GB) |

## Data Preparation

Convert the HuggingFace dataset to JSONL messages format:

```python
from datasets import load_dataset
import json

ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
with open("medical_flashcards.jsonl", "w") as f:
    for row in ds:
        messages = [
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]
        f.write(json.dumps({"messages": messages}) + "\n")
```

## Choosing an Algorithm

- **SFT** -- Use when you want maximum adaptation and are willing to retrain
  all parameters.  Risk of catastrophic forgetting if the dataset is small or
  narrow.

- **OSFT** -- Use when you need domain adaptation *and* want to preserve the
  model's general-purpose abilities (reasoning, language, instruction following).
  Constrains updates to an orthogonal subspace.

- **LoRA** -- Use when GPU memory is limited or you need to serve multiple
  adapters from the same base model.  Supports QLoRA for 4-bit quantised
  training on consumer GPUs.

## Quick Start

```bash
# SFT
python examples/medical_sft.py \
    --data-path ./medical_flashcards.jsonl \
    --ckpt-output-dir ./checkpoints/medical-sft

# OSFT (preserves base capabilities)
python examples/medical_osft.py \
    --data-path ./medical_flashcards.jsonl \
    --ckpt-output-dir ./checkpoints/medical-osft

# LoRA (single GPU)
python examples/medical_lora.py \
    --data-path ./medical_flashcards.jsonl \
    --ckpt-output-dir ./checkpoints/medical-lora

# QLoRA (even less memory)
python examples/medical_lora.py \
    --data-path ./medical_flashcards.jsonl \
    --ckpt-output-dir ./checkpoints/medical-qlora \
    --qlora
```
