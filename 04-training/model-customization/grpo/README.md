# Group Relative Policy Optimization (GRPO)

**Status:** GA

GRPO applies reinforcement learning from verifiable rewards to train tool-calling agents. It optimizes model behavior based on objective success criteria (e.g., correct tool invocations) rather than human preference labels. GRPO supports both single-turn and multi-turn tool-call training data.

## What's Covered

- Reinforcement learning with verifiable reward signals for agentic models
- Single-turn and multi-turn tool-call data formats
- Backend options: OpenPipe ART (single GPU), verl (multi-GPU distributed)
- Reward function design for tool-use correctness

## Quick Start

```python
from training_hub import lora_grpo
```

## When to Use GRPO

- You are training a model to use tools or APIs reliably
- You have verifiable success criteria (not just preference pairs)
- You need to improve agentic behavior beyond what SFT alone achieves

## Official Documentation

- [Customize models for Gen AI and Agentic AI](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/customize_models_for_gen_ai_and_agentic_ai_applications)
- [Training Hub repository](https://github.com/Red-Hat-AI-Innovation-Team/training_hub)

## What's in examples/

Examples demonstrate GRPO training on tool-call traces, configuring reward functions, choosing between ART and verl backends, and evaluating agent task completion rates.
