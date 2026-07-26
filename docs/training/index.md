# Training Algorithms

Training Hub provides multiple fine-tuning algorithms, each optimized for different constraints. Pick the one that matches your use case, GPU budget, and data type.

Not sure which to pick? Use the [decision flowchart](../getting-started/choosing-an-algorithm.md).

## Algorithm Overview

| Algorithm | Parameters Trained | Min GPU | Best For | Output |
|-----------|-------------------|---------|----------|--------|
| [SFT](sft.md) | All (100%) | 2x A100 80GB | Maximum learning from abundant data | Full model |
| [OSFT](osft.md) | All (constrained) | 2x A100 80GB | Adding knowledge without forgetting | Full model |
| [LoRA](lora.md) | ~1% (adapters) | 1x L4 24GB (QLoRA) | Single-GPU, tool-calling agents, multi-adapter serving | Adapter |
| [GRPO](grpo.md) | ~1% (LoRA) | 1-4x A100 | Reward-based tool-use learning | Adapter |

## Which Algorithm for Which Track?

=== "Knowledge Track"

    Teaching a model domain knowledge (financial regulations, medical literature, product docs):

    - **OSFT** (recommended) — Adds knowledge while preserving general capabilities
    - **SFT** — Maximum learning capacity when you have abundant data and don't need base knowledge retention
    - **LoRA** — Memory-efficient option when GPU resources are limited

=== "Agent Track"

    Building a tool-calling agent (MCP servers, APIs):

    - **LoRA SFT** (recommended, [validated on RHOAI](../end-to-end/financial-agent.md)) — Train on expert demonstrations from MCP distillation
    - **GRPO** — Learn from rewards when expert traces are unavailable

## Advanced Algorithms

| Algorithm | Use Case | Guide |
|-----------|----------|-------|
| [LAB Multi-Phase](lab-multiphase.md) | InstructLab's phased training pipeline | [Guide](lab-multiphase.md) |
| [Continual Learning](continual-learning.md) | Incrementally add knowledge without retraining | [Guide](continual-learning.md) |
| [Continued Pretraining](continued-pretraining.md) | Extend base model with large-scale domain text | [Guide](continued-pretraining.md) |
