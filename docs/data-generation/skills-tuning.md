# Skills Tuning Data Generation

Skills tuning flows generate instruction-following training data. Unlike knowledge tuning (which teaches the model *what to know*), skills tuning teaches the model *how to behave* — formatting, reasoning, summarization, and other capabilities.

## When to Use Skills Tuning

- You want to improve the model's **instruction-following ability**
- You're doing [LAB multi-phase training](../training/lab-multiphase.md) and need **Phase 2 data**
- You want to teach the model specific **output formats or styles**
- You need to **align** a model after [continued pretraining](../training/continued-pretraining.md)

!!! info "Knowledge vs Skills"
    **Knowledge tuning** teaches the model *facts* — "What is MaaS in RHOAI 3.4?" **Skills tuning** teaches the model *behaviors* — "Summarize this in 3 bullet points" or "Convert this query to SQL." Most fine-tuning projects benefit from both, combined via [LAB multi-phase training](../training/lab-multiphase.md).

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **SDG Hub** | `pip install sdg-hub` |
| **LLM API key** | OpenAI, Anthropic, or any LiteLLM-supported provider |
| **Task descriptions** | Clear descriptions of the skills you want to teach |

## Generate Skills Data

```python
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()

dataset = Dataset.from_dict({
    "task_description": [
        "Summarize technical documents in bullet point format",
        "Convert natural language queries to SQL",
        "Classify customer support tickets by priority",
        "Extract key entities from legal contracts",
    ],
    "domain": ["general", "data", "support", "legal"],
})

skills_flows = FlowRegistry.search_flows(tag="skills")

for flow_info in skills_flows:
    print(f"{flow_info['id']}: {flow_info['name']}")

flow = Flow.from_yaml(FlowRegistry.get_flow_path(skills_flows[0]["name"]))
flow.set_model_config(model="gpt-4o-mini")
result = flow.generate(dataset)

result.to_json("skills_data.jsonl", orient="records", lines=True)
print(f"Generated {len(result)} skills training examples")
```

## Seed Data Design

The quality of your skills data depends heavily on the seed task descriptions. Write clear, specific descriptions with examples of the desired behavior.

### Good Seed Data

| Task Description | Why it works |
|-----------------|-------------|
| "Summarize technical documents in 3-5 bullet points, focusing on key findings and action items" | Specific format, clear purpose |
| "Convert natural language questions about sales data into PostgreSQL queries using the `orders`, `products`, and `customers` tables" | Specifies SQL dialect and schema |
| "Classify support tickets as P1 (outage), P2 (degraded), P3 (minor), or P4 (question)" | Enumerated categories |

### Weak Seed Data

| Task Description | Why it's weak | Improvement |
|-----------------|--------------|-------------|
| "Summarize text" | Too vague — no format guidance | Add format, length, and focus criteria |
| "Write SQL" | No schema context | Specify tables, dialect, and query types |
| "Help with support tickets" | No clear skill target | Define specific classification or routing task |

## Output Format

Skills data uses the same messages format as knowledge data:

```json
{
  "messages": [
    {"role": "user", "content": "Summarize this in 3 bullet points: The new GPU instances on RHOAI reduced training time by 60%..."},
    {"role": "assistant", "content": "• RHOAI GPU instances cut training time by 60%\n• Setup process was straightforward\n• Documentation could benefit from more examples"}
  ]
}
```

## Combining with Knowledge Data

For [LAB multi-phase training](../training/lab-multiphase.md), keep knowledge and skills data separate. Each phase uses different hyperparameters optimized for its data type.

| Phase | Data | Source | Purpose |
|-------|------|--------|---------|
| Phase 1 | Knowledge data | [Knowledge tuning](knowledge-tuning.md) flows | Inject domain knowledge |
| Phase 2 | Skills data | Skills tuning flows (this page) | Teach instruction following |

=== "SFT Multi-Phase"

    ```python
    from training_hub import sft

    sft(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        data_path="knowledge_data.jsonl",
        ckpt_output_dir="./phase1",
        num_epochs=7,
        effective_batch_size=32,
        max_seq_len=4096,
        learning_rate=2e-5,
    )

    sft(
        model_path="./phase1/hf_format/samples_0",
        data_path="skills_data.jsonl",
        ckpt_output_dir="./phase2",
        num_epochs=3,
        effective_batch_size=32,
        max_seq_len=4096,
        learning_rate=5e-6,
    )
    ```

=== "OSFT Multi-Phase"

    ```python
    from training_hub import osft

    osft(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        data_path="knowledge_data.jsonl",
        ckpt_output_dir="./phase1",
        unfreeze_rank_ratio=0.01,
        effective_batch_size=32,
        max_tokens_per_gpu=16384,
        max_seq_len=4096,
        learning_rate=2e-5,
        num_epochs=7,
    )

    osft(
        model_path="./phase1/hf_format/samples_0",
        data_path="skills_data.jsonl",
        ckpt_output_dir="./phase2",
        unfreeze_rank_ratio=0.005,
        effective_batch_size=32,
        max_tokens_per_gpu=16384,
        max_seq_len=4096,
        learning_rate=5e-6,
        num_epochs=3,
    )
    ```

## Quality Filtering

Review generated skills data before training. Look for:

- **Consistency**: Does the assistant response match the requested format?
- **Accuracy**: Are SQL queries syntactically valid? Are classifications correct?
- **Diversity**: Do examples cover edge cases, not just the happy path?

```python
import pandas as pd

df = pd.read_json("skills_data.jsonl", lines=True)

print(f"Total examples: {len(df)}")
print(f"Avg message length: {df['messages'].apply(lambda m: len(str(m))).mean():.0f}")

for _, row in df.sample(5).iterrows():
    user_msg = next(m["content"] for m in row["messages"] if m["role"] == "user")
    asst_msg = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    print(f"\nQ: {user_msg[:100]}...")
    print(f"A: {asst_msg[:100]}...")
```

## Tips and Troubleshooting

!!! tip "Start with 3-5 Distinct Skills"
    Generate data for a focused set of skills rather than trying to cover everything at once. You can always add more skills in subsequent training phases via [continual learning](../training/continual-learning.md).

!!! tip "Balance Your Dataset"
    Ensure roughly equal representation across skill types. If 80% of your data is summarization and 20% is SQL, the model will be biased toward summarization.

!!! warning "Don't Mix Skills and Knowledge in Phase 1"
    LAB multi-phase training works because each phase has tailored hyperparameters. Mixing data types in a single phase produces worse results than the two-phase approach.

## Related

- [Knowledge Tuning](knowledge-tuning.md) — Generate domain knowledge data (Phase 1)
- [LAB Multi-Phase](../training/lab-multiphase.md) — Two-phase training with knowledge + skills
- [SDG Hub Overview](index.md) — Core concepts and flow architecture
- [Data Formats](../reference/data-formats.md) — JSONL format specification
