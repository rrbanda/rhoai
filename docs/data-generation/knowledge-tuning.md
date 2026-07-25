# Knowledge Tuning Data Generation

Knowledge tuning flows generate Q&A training pairs from your domain documents. SDG Hub provides four complementary flow variants that produce diverse training examples from the same source documents.

## Flow Variants

| Variant | Description | Output style |
|---------|-------------|-------------|
| **Extractive Summary** | Generates Q&A pairs grounded in extractive summaries | Factual, concise answers |
| **Detailed Summary** | Produces Q&A using detailed document summaries | Comprehensive, explanatory answers |
| **Key Facts** | Extracts key facts and generates verification Q&A | Knowledge verification pairs |
| **Document Direct QA** | Generates Q&A directly from document content | Diverse question types |

!!! tip "Use Multiple Variants"
    Running multiple variants on the same documents produces diverse training data. The [knowledge mixing](#mixing-variants) step combines and deduplicates the results.

## Generate Knowledge Data

```python
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()

# Prepare your documents
dataset = Dataset.from_dict({
    "document": [
        "RHOAI 3.4 brings Models-as-a-Service (MaaS) to GA...",
        "The Kubeflow Training Operator manages distributed...",
    ],
    "domain": ["rhoai", "rhoai"],
})

# Pick a flow variant
FLOW_VARIANTS = {
    "extractive_summary": "Extractive Summary Knowledge Tuning Dataset Generation Flow",
    "detailed_summary": "Detailed Summary Knowledge Tuning Dataset Generation Flow",
    "key_facts": "Key Facts Knowledge Tuning Dataset Generation Flow",
    "doc_direct_qa": "Document Based Knowledge Tuning Dataset Generation Flow",
}

# Generate with each variant
for name, flow_name in FLOW_VARIANTS.items():
    flow = Flow.from_yaml(FlowRegistry.get_flow_path(flow_name))
    flow.set_model_config(model="gpt-4o-mini")
    result = flow.generate(dataset)
    result.to_json(f"{name}_data.jsonl", orient="records", lines=True)
    print(f"{name}: {len(result)} examples")
```

## Mixing Variants

Combine outputs from multiple variants into a balanced training set:

```python
import pandas as pd

variants = [
    "extractive_summary_data.jsonl",
    "detailed_summary_data.jsonl",
    "key_facts_data.jsonl",
    "doc_direct_qa_data.jsonl",
]

dfs = [pd.read_json(f, lines=True) for f in variants]
combined = pd.concat(dfs, ignore_index=True)
combined = combined.drop_duplicates(subset=["messages"])

combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

combined.to_json("knowledge_mixed.jsonl", orient="records", lines=True)
print(f"Combined: {len(combined)} unique examples")
```

## Multilingual Knowledge Generation

Generate knowledge data in multiple languages by specifying the target language:

```python
flow = Flow.from_yaml(FlowRegistry.get_flow_path(
    "Document Based Knowledge Tuning Dataset Generation Flow"
))
flow.set_model_config(model="gpt-4o-mini")

result = flow.generate(
    dataset,
    runtime_params={
        "gen_questions": {"language": "Spanish"},
        "gen_answers": {"language": "Spanish"},
    }
)
```

## Document Preparation

### From URLs or PDFs (via Docling)

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("https://example.com/docs/rhoai-3.4.html")
markdown_text = result.document.export_to_markdown()
```

### Chunking Strategy

For long documents, split into chunks of 500-2000 tokens. Each chunk becomes one row in the seed dataset:

```python
chunks = [markdown_text[i:i+2000] for i in range(0, len(markdown_text), 1500)]

dataset = Dataset.from_dict({
    "document": chunks,
    "domain": ["your-domain"] * len(chunks),
})
```

## Output Format

Knowledge tuning flows produce JSONL with the messages format expected by Training Hub:

```json
{
  "messages": [
    {"role": "system", "content": "You are a domain expert."},
    {"role": "user", "content": "What is Models-as-a-Service in RHOAI?"},
    {"role": "assistant", "content": "Models-as-a-Service (MaaS) in RHOAI 3.4 allows users to access hosted LLMs directly..."}
  ]
}
```

## Related

- [SDG Hub Overview](index.md) — Core concepts and architecture
- [Skills Tuning](skills-tuning.md) — Generate instruction-following data
- [Knowledge Tuning Pipeline](../end-to-end/knowledge-tuning.md) — Full E2E pipeline including training
- [Data Formats](../reference/data-formats.md) — Detailed format specification
