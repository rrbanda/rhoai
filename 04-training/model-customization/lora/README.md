# Low-Rank Adaptation (LoRA)

**Status:** GA

LoRA is a parameter-efficient fine-tuning method that injects trainable low-rank matrices into frozen model layers. This dramatically reduces VRAM requirements and training time while maintaining competitive performance. Training Hub uses the Unsloth backend for optimized LoRA training.

## What's Covered

- Parameter-efficient fine-tuning with low-rank adapters
- Key parameters: `lora_r` (rank), `lora_alpha` (scaling factor)
- Unsloth backend for memory-efficient training
- Adapter merging and export for deployment

## Quick Start

```python
from training_hub import lora_sft
```

## When to Use LoRA

- You are compute-constrained or have limited VRAM
- You want fast iteration with smaller adapter checkpoints
- You need to maintain multiple task-specific adapters for one base model

## Official Documentation

- [Customize models for Gen AI and Agentic AI](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/customize_models_for_gen_ai_and_agentic_ai_applications)
- [Training Hub repository](https://github.com/Red-Hat-AI-Innovation-Team/training_hub)

## What's in examples/

Examples show LoRA fine-tuning on a single GPU, tuning `lora_r` and `lora_alpha`, merging adapters back into the base model, and serving the result.
