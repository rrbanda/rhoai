# Data Formats

All data flowing between SDG Hub and Training Hub uses the **JSONL messages format**. Understanding this format is essential for data preparation, debugging, and custom pipeline integration.

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

For [continued pretraining](../training/continued-pretraining.md), data uses the same messages format but with `unmask_input=True` during training:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "The VLOOKUP function searches for a value in the first column of a table range and returns a value in the same row from another column. Syntax: VLOOKUP(lookup_value, table_array, col_index_num, [range_lookup])..."
    }
  ]
}
```

## The `unmask` Flag

By default, Training Hub computes loss only on **assistant** tokens (the model learns to generate responses, not to repeat questions). The `unmask_input=True` flag changes this:

| Setting | Trains on | Use case |
|---------|----------|----------|
| `unmask_input=False` (default) | Assistant tokens only | SFT, OSFT, LoRA |
| `unmask_input=True` | All tokens | Continued pretraining |

## SDG Hub Output

SDG Hub flows produce pandas DataFrames that can be saved directly as JSONL:

```python
result = flow.generate(dataset)

# Save as JSONL (one JSON object per line)
result.to_json("output.jsonl", orient="records", lines=True)
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
- [Continued Pretraining](../training/continued-pretraining.md) — Using `unmask_input`
- [GPU Requirements](gpu-requirements.md) — Plan compute resources
