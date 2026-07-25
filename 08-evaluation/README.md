# Model Evaluation

**Status:** GA

Model evaluation in Red Hat OpenShift AI 3.4 provides tools to systematically assess AI model quality, accuracy, and performance against standardized benchmarks. This section covers evaluation frameworks available on the platform.

## What's Covered

- LM-Eval framework for language model evaluation
- Configuring evaluation jobs via custom resources
- Selecting and running benchmark tasks
- Retrieving and comparing evaluation metrics across models
- Generating RAG evaluation datasets with ground truth context
- Creating execution-verified coding benchmarks and evaluating coding models
- Evaluating tool-use agents with synthetic MCP benchmarks and LLM-as-judge scoring

## Evaluation Methods

| Method | Directory | Description |
|--------|-----------|-------------|
| **LM-Eval** | [`lm-eval/`](lm-eval/) | Standard language model evaluation using LMEvalJob CRDs with benchmarks like MMLU, HellaSwag, and ARC |
| **RAG Evaluation** | [`rag-evaluation/`](rag-evaluation/) | Generate Q&A datasets with ground truth context for evaluating RAG systems, compatible with RAGAS |
| **Code Evaluation** | [`code-evaluation/`](code-evaluation/) | Generate execution-verified coding benchmarks and evaluate coding models with pass@1 scoring |
| **Agent Evaluation** | [`agent-evaluation/`](agent-evaluation/) | Generate synthetic MCP benchmarks and evaluate tool-use agents across LLMs with LLM-as-judge scoring |

## Official Documentation

- [Evaluating AI Systems](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/evaluating_ai_systems)

## What's in examples/

Examples include sample LMEvalJob configurations, benchmark task selections for common use cases, SDG Hub-based evaluation dataset generation scripts, coding model evaluation harnesses, and agent tool-use evaluation pipelines.
