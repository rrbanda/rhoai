# Knowledge Tuning End-to-End

**Status:** GA

Full pipeline for injecting domain-specific knowledge into a language model. Covers the complete workflow from document processing through deployment: Data Processing (Docling) -> Synthetic Data Generation (SDG Hub) -> Training (Training Hub SFT/OSFT) -> Evaluation (LM-Eval) -> Deployment.

## What's Covered

- Document ingestion and processing with Docling
- Generating training data with SDG Hub knowledge flows
- Fine-tuning with Training Hub (SFT or OSFT)
- Evaluating the tuned model with LM-Eval benchmarks
- Deploying the final model via KServe/vLLM

## Official Documentation

- [Knowledge Tuning Example](https://github.com/red-hat-data-services/red-hat-ai-examples/tree/main/examples/knowledge-tuning)

## What's in examples/

Examples will include the complete pipeline configuration, sample source documents, SDG flow definitions, training configurations, evaluation benchmarks, and deployment manifests for each stage.
