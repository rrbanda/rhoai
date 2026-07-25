# Skills Tuning with SDG Hub

**Status:** GA (General Availability)

Skills tuning generates synthetic data for teaching models specific capabilities such as instruction following, reasoning, and tool use. Each skill type has its own dedicated flow YAML that defines the generation pipeline. The output is JSONL training data formatted for Training Hub.

## What's Covered

- Generating synthetic data for instruction-following skills
- Generating synthetic data for reasoning and chain-of-thought skills
- Generating synthetic data for tool-use and function-calling skills
- Using skill-specific flow YAMLs to define generation pipelines
- Customizing skill flows with runtime parameters

## Official Documentation

- [Generate Synthetic Data](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/customize_models_for_gen_ai_and_agentic_ai_applications/generate-synthetic-data_custom-models)

## What's in examples/

Examples demonstrate running skill-specific SDG flows, configuring teacher models for different skill types, and producing training datasets for instruction following, reasoning, and tool use.
