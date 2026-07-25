# Text Analysis & Structured Insights

SDG Hub's text analysis flows extract structured insights from documents — sentiment, key themes, entities, and classifications. The output can be used as training data for smaller classification or extraction models, or as structured labels for downstream analytics.

## When to Use Text Analysis

- You need **sentiment analysis** training data from product reviews or feedback
- You want **named entity recognition** training from domain documents
- You're building **topic classification** models for document categorization
- You need **key fact extraction** for knowledge bases or structured databases
- You want to create **labeled datasets** without manual annotation

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **SDG Hub** | `pip install sdg_hub` |
| **LLM API key** | OpenAI, Anthropic, or any LiteLLM-supported provider |
| **Source documents** | Raw text documents to analyze |

## Generate Text Analysis Data

```python
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()

dataset = Dataset.from_dict({
    "document": [
        "The new GPU instances on RHOAI reduced our training time by 60%. "
        "The setup was straightforward, though documentation could be clearer.",
        "Customer support was unresponsive for 3 days. The billing system "
        "charged us twice for the same service.",
        "Excellent onboarding experience. The tutorials walked us through "
        "every step and the model was serving in under an hour.",
    ],
    "domain": ["product-review", "product-review", "product-review"],
})

text_flows = FlowRegistry.search_flows(tag="text-analysis")
flow_info = text_flows[0]

flow = Flow.from_yaml(FlowRegistry.get_flow_path(flow_info["name"]))
flow.set_model_config(model="gpt-4o-mini")
result = flow.generate(dataset)

result.to_json("text_analysis.jsonl", orient="records", lines=True)
print(f"Generated {len(result)} analyzed documents")
```

## Output Structure

Text analysis flows produce structured outputs with multiple extracted fields:

```json
{
  "document": "The new GPU instances on RHOAI reduced our training time by 60%...",
  "sentiment": "mixed_positive",
  "key_themes": ["performance", "documentation"],
  "entities": [
    {"text": "RHOAI", "type": "product"},
    {"text": "GPU", "type": "technology"}
  ],
  "summary": "Positive review of GPU performance with documentation improvement suggestion"
}
```

## Use Cases

### Training a Sentiment Classifier

Convert extracted sentiments into SFT training data for a smaller model:

```python
import pandas as pd

df = pd.read_json("text_analysis.jsonl", lines=True)

training_data = []
for _, row in df.iterrows():
    training_data.append({
        "messages": [
            {"role": "user", "content": f"Classify the sentiment: {row['document']}"},
            {"role": "assistant", "content": row["sentiment"]},
        ]
    })

pd.DataFrame(training_data).to_json(
    "sentiment_training.jsonl", orient="records", lines=True
)
print(f"Created {len(training_data)} sentiment training examples")
```

Then fine-tune a smaller model for fast, cost-effective inference:

```python
from training_hub import lora_sft

lora_sft(
    model_path="microsoft/Phi-4-mini-instruct",
    data_path="sentiment_training.jsonl",
    ckpt_output_dir="./sentiment-model",
    num_epochs=5,
    lora_r=16,
    lora_alpha=32,
)
```

### Training an Entity Extractor

Convert entity annotations into instruction-following data:

```python
training_data = []
for _, row in df.iterrows():
    entities_str = ", ".join(
        f"{e['text']} ({e['type']})" for e in row["entities"]
    )
    training_data.append({
        "messages": [
            {"role": "user", "content": f"Extract entities: {row['document']}"},
            {"role": "assistant", "content": entities_str},
        ]
    })

pd.DataFrame(training_data).to_json(
    "entity_training.jsonl", orient="records", lines=True
)
```

### Building a Topic Classifier

Use extracted themes as topic labels:

```python
training_data = []
for _, row in df.iterrows():
    topics = ", ".join(row["key_themes"])
    training_data.append({
        "messages": [
            {"role": "user", "content": f"Classify the topics: {row['document']}"},
            {"role": "assistant", "content": topics},
        ]
    })

pd.DataFrame(training_data).to_json(
    "topic_training.jsonl", orient="records", lines=True
)
```

## Processing at Scale

For large document collections, process in batches and track progress:

```python
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry
import pandas as pd

FlowRegistry.discover_flows()
text_flows = FlowRegistry.search_flows(tag="text-analysis")
flow = Flow.from_yaml(FlowRegistry.get_flow_path(text_flows[0]["name"]))
flow.set_model_config(model="gpt-4o-mini")

documents = pd.read_json("raw_documents.jsonl", lines=True)
batch_size = 50
all_results = []

for i in range(0, len(documents), batch_size):
    batch = documents.iloc[i:i+batch_size]
    dataset = Dataset.from_pandas(batch)
    result = flow.generate(dataset)
    all_results.append(result)
    print(f"Processed {min(i+batch_size, len(documents))}/{len(documents)}")

combined = pd.concat(all_results, ignore_index=True)
combined.to_json("text_analysis_full.jsonl", orient="records", lines=True)
```

## Tips and Troubleshooting

!!! tip "Validate Extracted Labels"
    Spot-check a sample of 20-50 extracted labels for accuracy. LLM-generated labels are generally high quality but can misclassify edge cases, especially for domain-specific sentiment.

!!! tip "Combine with Manual Labels"
    Mix LLM-generated labels with a small set of manually verified examples. This gives you the scale of synthetic data with the precision of human annotation for ambiguous cases.

!!! warning "Sentiment Granularity"
    Coarse sentiment labels (positive/negative/neutral) are more reliable than fine-grained ones (slightly_positive, mixed_positive). Start coarse and refine based on your model's accuracy.

## Related

- [SDG Hub Overview](index.md) — Core concepts and flow architecture
- [Knowledge Tuning](knowledge-tuning.md) — Generate Q&A training data instead
- [Data Formats](../reference/data-formats.md) — JSONL format specification
- [LoRA](../training/lora.md) — Memory-efficient fine-tuning for classification models
