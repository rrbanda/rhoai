# Training Pipeline

**Status:** GA

A Kubeflow Pipeline for end-to-end model fine-tuning using Training Hub. Orchestrates the full workflow: dataset download, fine-tuning (SFT/OSFT/GRPO), evaluation with LM-Eval, and model registration to a model registry.

## What's Covered

- End-to-end training pipeline architecture
- Dataset download and preparation stages
- Fine-tuning with Training Hub (SFT, OSFT, GRPO)
- Automated evaluation using LM-Eval
- Model registration upon successful evaluation
- Pipeline parameterization and configuration

## Official Documentation

- [Training Pipeline Reference](https://github.com/red-hat-data-services/red-hat-ai-examples/tree/main/examples/fine-tuning/pipelines/training-hub)

## What's in examples/

Examples will include complete pipeline definitions, stage configurations for each training method, evaluation criteria setup, and model registry integration patterns.
