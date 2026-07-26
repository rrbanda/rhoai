# Data Formats

All data flowing between SDG Hub and Training Hub uses standardized formats. Understanding these is essential for data preparation, debugging, and custom pipeline integration.

## Messages Format (Chat)

The standard format for instruction-tuning data. Each line is a JSON object with a `messages` array:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is RHOAI?"},
    {"role": "assistant", "content": "Red Hat OpenShift AI is an MLOps platform..."}
  ]
}
```

### Roles

| Role | Purpose | Required |
|------|---------|----------|
| `system` | Sets the assistant's behavior/persona | Optional |
| `user` | User's input/question | Required |
| `assistant` | Model's response (what the model learns to generate) | Required |
| `tool` | Result of a tool call (for agent training) | Optional |

### Multi-Turn Conversations

Conversations can include multiple user-assistant turns:

```json
{
  "messages": [
    {"role": "user", "content": "What is OSFT?"},
    {"role": "assistant", "content": "Orthogonal Subspace Fine-Tuning..."},
    {"role": "user", "content": "How does it differ from SFT?"},
    {"role": "assistant", "content": "The key difference is..."}
  ]
}
```

## Tool-Use Format

For training tool-use models (GRPO), messages include tool calls and results:

```json
{
  "messages": [
    {"role": "user", "content": "Find products under $20"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "type": "function",
        "function": {
          "name": "search_products",
          "arguments": "{\"max_price\": 20}"
        }
      }]
    },
    {
      "role": "tool",
      "content": "[{\"name\": \"Widget\", \"price\": 15.99}]"
    },
    {
      "role": "assistant",
      "content": "I found Widget at $15.99."
    }
  ]
}
```

## Pretraining Format

For [continued pretraining](../training/continued-pretraining.md), data uses a `document` column containing raw text:

```json
{"document": "The VLOOKUP function searches for a value in the first column of a table range and returns a value in the same row from another column. Syntax: VLOOKUP(lookup_value, table_array, col_index_num, [range_lookup])..."}
```

Training Hub's `sft()` function with `is_pretraining=True` and `document_column_name="document"` handles tokenization and packing of these raw text entries.

## The `is_pretraining` Flag

By default, Training Hub computes loss only on **assistant** tokens (the model learns to generate responses, not to repeat questions). The `is_pretraining=True` flag changes this:

| Setting | Trains on | Data format | Use case |
|---------|----------|-------------|----------|
| `is_pretraining=False` (default) | Assistant tokens only | Messages JSONL | SFT, OSFT, LoRA |
| `is_pretraining=True` | All tokens | Document JSONL | Continued pretraining |

## SDG Hub Output

SDG Hub flows produce pandas DataFrames (or HuggingFace Datasets, matching the input type). The output columns depend on the flow type:

| Flow Type | Output Columns | Needs Conversion? |
|-----------|---------------|-------------------|
| Knowledge Tuning | `question`, `response`, `domain`, ... | **Yes** — convert to `messages` format |
| MCP Distillation | `tool_trace`, `question`, `mcp_server_name`, ... | **Yes** — convert to `messages` with `tool_calls` |
| Text Analysis | `text`, `sentiment`, `key_themes`, ... | No — used for analysis, not training |

### Converting Knowledge Tuning Output to Training Format

Knowledge flows output `question`/`response` columns. Training Hub expects `messages` format. For knowledge tuning, set `"unmask": true` so the loss covers all message roles:

```python
import pandas as pd

raw = pd.read_json("sdg_output.jsonl", lines=True)

records = []
for _, row in raw.iterrows():
    records.append({
        "messages": [
            {"role": "user", "content": str(row["question"])},
            {"role": "assistant", "content": str(row["response"])},
        ],
        "unmask": True,
    })

training_df = pd.DataFrame(records)
training_df.to_json("training_data.jsonl", orient="records", lines=True)
```

### Saving Raw SDG Output

```python
result = flow.generate(dataset)
result_df = result.to_pandas() if hasattr(result, "to_pandas") else result
result_df.to_json("output.jsonl", orient="records", lines=True)
```

## Validating Data

Quick validation of your JSONL file:

```python
import json

with open("training_data.jsonl") as f:
    for i, line in enumerate(f):
        record = json.loads(line)
        assert "messages" in record, f"Line {i}: missing 'messages'"
        assert len(record["messages"]) >= 2, f"Line {i}: need at least user + assistant"

        roles = [m["role"] for m in record["messages"]]
        assert "assistant" in roles, f"Line {i}: no assistant message"

print(f"Validated {i+1} records")
```

## Related

- [Knowledge Tuning](../data-generation/knowledge-tuning.md) — Generate messages-format data
- [SFT](../training/sft.md) — Training with messages data
- [Continued Pretraining](../training/continued-pretraining.md) — Using `is_pretraining`
- [GPU Requirements](gpu-requirements.md) — Plan compute resources
