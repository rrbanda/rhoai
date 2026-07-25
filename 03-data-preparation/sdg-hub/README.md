# SDG Hub

**Status:** GA (General Availability)

SDG Hub is a Python framework for building synthetic data generation pipelines using composable blocks and YAML-defined flows. Blocks are processing units that transform datasets; flows chain blocks into pipelines. It supports 100+ LLM providers via LiteLLM, enabling flexible model selection for generation tasks.

## What's Covered

- Installing and configuring SDG Hub
- Understanding blocks (processing units) and flows (YAML pipelines)
- Generating synthetic data for knowledge tuning and skills tuning
- Running SDG pipelines as Kubeflow Pipeline (KFP) components on OpenShift AI
- Using `dry_run()` to validate pipelines before making LLM calls
- Configuring LLM providers and API keys via LiteLLM

## Official Documentation

- [Generate Synthetic Data](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/customize_models_for_gen_ai_and_agentic_ai_applications/generate-synthetic-data_custom-models)
- [SDG Hub Repository](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub)

## Subdirectories

| Directory | Description |
|-----------|-------------|
| [knowledge-tuning/](knowledge-tuning/) | Synthetic Q&A and document augmentation for domain knowledge |
| [skills-tuning/](skills-tuning/) | Synthetic data for instruction following, reasoning, and tool use |
| [kfp-pipeline/](kfp-pipeline/) | Running SDG flows as Kubeflow Pipeline components |

## What's in examples/

Examples demonstrate building and running SDG pipelines -- from defining custom blocks and flows to generating JSONL training data for downstream model fine-tuning.
