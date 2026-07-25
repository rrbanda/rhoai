# Data Preparation

**Status:** GA (General Availability) -- RHOAI 3.4

Data preparation is a critical step before model training. This section covers the tools and workflows available in Red Hat OpenShift AI 3.4 for transforming raw data into high-quality training datasets.

## What's Covered

- **[SDG Hub](sdg-hub/)** -- Synthetic data generation using composable blocks and YAML-defined flows
- **[Data Processing](data-processing/)** -- Document ingestion and preprocessing with Docling (URLs, PDFs, Markdown)
- **[Feature Store](feature-store/)** -- Defining, storing, and serving reusable ML features (Technology Preview)

## Official Documentation

- [RHOAI 3.4 Documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4)
- [Customize Models for Gen AI and Agentic AI](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/customize_models_for_gen_ai_and_agentic_ai_applications)

## What's in examples/

Examples in subdirectories demonstrate end-to-end data preparation workflows -- from raw document ingestion through synthetic data generation to producing JSONL training data ready for Training Hub.
