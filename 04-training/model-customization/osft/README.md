# Orthogonal Subspace Fine-Tuning (OSFT)

**Status:** GA

OSFT trains model parameters in orthogonal subspaces, enabling the addition of new knowledge while preserving existing capabilities. This approach prevents catastrophic forgetting by constraining weight updates to directions that do not interfere with previously learned representations.

## What's Covered

- Training in orthogonal subspaces to protect prior knowledge
- Key parameter: `unfreeze_rank_ratio` (default 0.5) -- controls the fraction of subspace allocated to new learning
- Integration with JSONL datasets produced by SDG Hub
- Comparison guidance vs. SFT and LoRA

## Quick Start

```python
from training_hub import osft
```

## When to Use OSFT

- You need to add new knowledge without degrading existing model capabilities
- Catastrophic forgetting is not acceptable
- You want full-parameter-level capacity with knowledge preservation guarantees

## Official Documentation

- [Customize models for Gen AI and Agentic AI](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/customize_models_for_gen_ai_and_agentic_ai_applications)
- [Training Hub repository](https://github.com/Red-Hat-AI-Innovation-Team/training_hub)

## What's in examples/

Examples demonstrate OSFT training with various `unfreeze_rank_ratio` settings, comparing forgetting metrics against standard SFT on the same dataset.
