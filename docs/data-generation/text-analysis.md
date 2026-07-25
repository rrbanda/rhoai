# Text Analysis & Structured Insights

SDG Hub's text analysis flows extract structured insights from documents — sentiment, key themes, entities, and classifications. The output can be used as training data for smaller classification or extraction models.

## Use Cases

- **Sentiment analysis** training data from product reviews
- **Named entity recognition** training from domain documents
- **Topic classification** labels for document categorization
- **Key fact extraction** for building knowledge bases

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
    ],
    "domain": ["product-review", "product-review"],
})

# Find text analysis flows by tag
text_flows = FlowRegistry.search_flows(tag="text-analysis")
flow_info = text_flows[0]

flow = Flow.from_yaml(FlowRegistry.get_flow_path(flow_info["name"]))
flow.set_model_config(model="gpt-4o-mini")
result = flow.generate(dataset)

result.to_json("text_analysis.jsonl", orient="records", lines=True)
```

## Output Structure

Text analysis flows produce structured outputs with multiple extracted fields:

```json
{
  "document": "The new GPU instances...",
  "sentiment": "mixed_positive",
  "key_themes": ["performance", "documentation"],
  "entities": [
    {"text": "RHOAI", "type": "product"},
    {"text": "GPU", "type": "technology"}
  ],
  "summary": "Positive review of GPU performance with documentation improvement suggestion"
}
```

## Training a Classification Model

Use the extracted labels to fine-tune a smaller model for classification:

```python
import pandas as pd

# Load text analysis results
df = pd.read_json("text_analysis.jsonl", lines=True)

# Convert to messages format for SFT
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
```

## Related

- [SDG Hub Overview](index.md) — Core concepts
- [Knowledge Tuning](knowledge-tuning.md) — Generate Q&A training data instead
- [Data Formats](../reference/data-formats.md) — JSONL format specification
