# Skills Tuning Data Generation

Skills tuning flows generate instruction-following training data. Unlike knowledge tuning (which teaches the model *what to know*), skills tuning teaches the model *how to behave* — formatting, reasoning, summarization, and other capabilities.

## When to Use Skills Tuning

- You want to improve the model's **instruction-following ability**
- You're doing [LAB multi-phase training](../training/lab-multiphase.md) and need **Phase 2 data**
- You want to teach the model specific **output formats or styles**

## Generate Skills Data

```python
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()

# Seed data with task descriptions
dataset = Dataset.from_dict({
    "task_description": [
        "Summarize technical documents in bullet point format",
        "Convert natural language queries to SQL",
        "Classify customer support tickets by priority",
    ],
    "domain": ["general", "data", "support"],
})

# Search for skills-related flows by tag
skills_flows = FlowRegistry.search_flows(tag="skills")

for flow_info in skills_flows:
    print(f"{flow_info['id']}: {flow_info['name']}")

# Use a skills flow
flow = Flow.from_yaml(FlowRegistry.get_flow_path(skills_flows[0]["name"]))
flow.set_model_config(model="gpt-4o-mini")
result = flow.generate(dataset)

result.to_json("skills_data.jsonl", orient="records", lines=True)
```

## Combining with Knowledge Data

For [LAB multi-phase training](../training/lab-multiphase.md), keep knowledge and skills data separate:

| Phase | Data | Purpose |
|-------|------|---------|
| Phase 1 | Knowledge data (from [knowledge tuning](knowledge-tuning.md)) | Inject domain knowledge |
| Phase 2 | Skills data (from this page) | Teach instruction following |

```python
from training_hub import sft

sft(model_path="base-model", data_path="knowledge_data.jsonl",
    ckpt_output_dir="./phase1", num_epochs=7, learning_rate=2e-5)

sft(model_path="./phase1/hf_format/samples_0", data_path="skills_data.jsonl",
    ckpt_output_dir="./phase2", num_epochs=3, learning_rate=5e-6)
```

## Output Format

Skills data uses the same messages format as knowledge data:

```json
{
  "messages": [
    {"role": "user", "content": "Summarize this in 3 bullet points: ..."},
    {"role": "assistant", "content": "• Point 1\n• Point 2\n• Point 3"}
  ]
}
```

## Related

- [Knowledge Tuning](knowledge-tuning.md) — Generate domain knowledge data
- [LAB Multi-Phase](../training/lab-multiphase.md) — Two-phase training with knowledge + skills
- [SDG Hub Overview](index.md) — Core concepts
