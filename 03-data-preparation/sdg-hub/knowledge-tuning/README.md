# Knowledge Tuning with SDG Hub

**Status:** GA (General Availability)

Knowledge tuning generates synthetic Q&A pairs and document augmentations to inject domain knowledge into models. Augmentations include detailed summaries, extractive summaries, and atomic facts. The enhanced multi-summary Q&A flows produce JSONL training data ready for Training Hub.

## What's Covered

- Generating synthetic question-answer pairs from domain documents
- Producing document augmentations (detailed summaries, extractive summaries, atomic facts)
- Using the enhanced multi-summary Q&A flow YAMLs
- Formatting output as JSONL for downstream fine-tuning with Training Hub
- Configuring teacher models for high-quality knowledge generation

## Official Documentation

- [Generate Synthetic Data](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/customize_models_for_gen_ai_and_agentic_ai_applications/generate-synthetic-data_custom-models)

## What's in examples/

Examples show how to run knowledge-tuning flows against domain documents, configure teacher models, and produce JSONL training datasets with Q&A pairs and document augmentations.
