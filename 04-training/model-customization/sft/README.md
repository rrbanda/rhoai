# Supervised Fine-Tuning (SFT)

**Status:** GA

Supervised Fine-Tuning performs full-parameter training on labeled datasets. It updates all model weights, providing maximum learning capacity at the cost of potentially overwriting prior knowledge. SFT supports both single-node and multi-node distributed training.

## What's Covered

- Full-parameter weight updates on supervised instruction/response pairs
- Single-node and multi-node distributed training configurations
- Integration with JSONL datasets produced by SDG Hub
- Checkpoint saving and resumption

## Quick Start

```python
from training_hub import sft
```

## When to Use SFT

- You need maximum model capacity for a new domain
- Catastrophic forgetting of prior knowledge is acceptable
- You have sufficient compute for full-parameter updates

## Official Documentation

- [Customize models for Gen AI and Agentic AI](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/customize_models_for_gen_ai_and_agentic_ai_applications)
- [Training Hub repository](https://github.com/Red-Hat-AI-Innovation-Team/training_hub)

## What's in examples/

Examples show how to run SFT on instruction-tuning datasets, configure distributed training across multiple GPUs, and evaluate the fine-tuned model.
