# Quickstart

Get from raw documents to a fine-tuned model in under 5 minutes (excluding training time).

## Prerequisites

```bash
pip install sdg_hub training_hub
```

Set your LLM API key for synthetic data generation:

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Step 1: Generate Training Data

Create a seed document and generate Q&A training pairs:

```python
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()

seed_data = Dataset.from_dict({
    "document": [
        "RHOAI 3.4 introduces Models-as-a-Service (MaaS), allowing users "
        "to access hosted LLMs directly from the dashboard without deploying "
        "model servers. Supported providers include OpenAI, Anthropic, and "
        "IBM watsonx. MaaS endpoints are rate-limited and metered per-token."
    ],
    "domain": ["rhoai"],
})

flow = Flow.from_yaml(
    FlowRegistry.get_flow_path(
        "Document Based Knowledge Tuning Dataset Generation Flow"
    )
)
flow.set_model_config(model="gpt-4o-mini")
result = flow.generate(seed_data)

result.to_json("training_data.jsonl", orient="records", lines=True)
print(f"Generated {len(result)} training examples")
```

## Step 2: Train a Model

Fine-tune a small model on the generated data:

=== "SFT (Full Fine-Tune)"

    ```python
    from training_hub import sft

    sft(
        model="meta-llama/Llama-3.1-8B-Instruct",
        data="training_data.jsonl",
        output_dir="./my-model",
        num_epochs=4,
        batch_size=32,
        max_seq_len=4096,
    )
    ```

=== "OSFT (Knowledge-Preserving)"

    ```python
    from training_hub import osft

    osft(
        model="meta-llama/Llama-3.1-8B-Instruct",
        data="training_data.jsonl",
        output_dir="./my-model",
        num_epochs=4,
        batch_size=32,
        max_seq_len=4096,
    )
    ```

=== "LoRA (Memory-Efficient)"

    ```python
    from training_hub import lora_sft

    lora_sft(
        model="meta-llama/Llama-3.1-8B-Instruct",
        data="training_data.jsonl",
        output_dir="./my-model",
        num_epochs=4,
        lora_r=16,
        lora_alpha=32,
    )
    ```

## Step 3: Evaluate

Check training loss convergence:

```python
from training_hub import plot_loss

plot_loss("./my-model")
```

## Next Steps

- [Choosing an Algorithm](choosing-an-algorithm.md) — When to use SFT vs OSFT vs LoRA vs GRPO
- [Knowledge Tuning Pipeline](../end-to-end/knowledge-tuning.md) — Full end-to-end walkthrough with evaluation
- [Memory Estimator](../utilities/memory-estimator.md) — Check GPU requirements before training
