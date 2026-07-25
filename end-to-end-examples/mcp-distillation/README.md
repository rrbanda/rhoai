# MCP Distillation End-to-End

**Status:** GA

Teach a small language model to use MCP server tools through synthetic data generation and reinforcement learning. Combines SDG Hub's MCP Distillation flow for tool-use data generation with Training Hub's LoRA GRPO method for efficient fine-tuning.

## What's Covered

- Generating tool-use training data with SDG Hub MCP Distillation flow
- Configuring MCP server connections for trace collection
- Training with LoRA GRPO via Training Hub
- Evaluating tool-use accuracy of the distilled model
- Deploying the tool-capable model

## Official Documentation

- [SDG Hub MCP Distillation Examples](https://github.com/red-hat-data-services/sdg_hub/tree/main/examples/agentic/mcp_distillation_training)

## What's in examples/

Examples will include MCP distillation flow configurations, sample MCP server setups, GRPO training parameters, evaluation scripts for tool-use accuracy, and deployment manifests.
