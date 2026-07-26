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
    "document_outline": [
        "Overview of RHOAI 3.4 MaaS feature",
        "Kubeflow Training Operator architecture",
    ],
    "domain": ["rhoai", "rhoai"],
    "icl_document": ["", ""],
    "icl_query_1": ["", ""],
    "icl_query_2": ["", ""],
    "icl_query_3": ["", ""],
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
    flow_path = FlowRegistry.get_flow_path(flow_name)
    if flow_path is None:
        print(f"WARNING: {flow_name} not found, skipping")
        continue
    flow = Flow.from_yaml(flow_path)
    flow.set_model_config(model="gpt-4o-mini")
    result = flow.generate(dataset)
    result_df = result.to_pandas() if hasattr(result, "to_pandas") else result
    result_df.to_json(f"{name}_data.jsonl", orient="records", lines=True)
    print(f"{name}: {len(result_df)} examples")
```

## Convert and Mix Variants

!!! info "SDG Hub output format"
    Knowledge flows output rows with `question` and `response` columns — **not** `messages`. You must convert to Training Hub's `messages` format before training. For knowledge tuning, set `"unmask": true` so the loss covers all message roles.

Convert and combine outputs from multiple variants:

```python
import pandas as pd

variants = [
    "extractive_summary_data.jsonl",
    "detailed_summary_data.jsonl",
    "key_facts_data.jsonl",
    "doc_direct_qa_data.jsonl",
]

def convert_to_messages(df):
    records = []
    for _, row in df.iterrows():
        if "question" not in row or "response" not in row:
            continue
        records.append({
            "messages": [
                {"role": "user", "content": str(row["question"])},
                {"role": "assistant", "content": str(row["response"])},
            ],
            "unmask": True,
        })
    return pd.DataFrame(records)

dfs = []
for f in variants:
    raw = pd.read_json(f, lines=True)
    dfs.append(convert_to_messages(raw))

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

!!! warning "SDG Hub output ≠ Training Hub input"
    Knowledge tuning flows produce rows with **`question` and `response` columns** — they do **not** output `messages` format directly. You must convert before training. See the [Convert and Mix](#convert-and-mix-variants) section above.

**Raw SDG Hub output** (one JSONL row):

```json
{"question": "What is Models-as-a-Service in RHOAI?", "response": "Models-as-a-Service (MaaS) in RHOAI 3.4 allows users to access hosted LLMs directly...", "domain": "rhoai"}
```

**After conversion** (what Training Hub expects):

```json
{"messages": [{"role": "user", "content": "What is Models-as-a-Service in RHOAI?"}, {"role": "assistant", "content": "Models-as-a-Service (MaaS) in RHOAI 3.4 allows users to access hosted LLMs directly..."}], "unmask": true}
```

## Related

- [SDG Hub Overview](index.md) — Core concepts and architecture
- [Knowledge Tuning Pipeline](../end-to-end/knowledge-tuning.md) — Full E2E pipeline including training
- [Data Formats](../reference/data-formats.md) — Detailed format specification
