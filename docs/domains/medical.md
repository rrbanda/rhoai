# Medical Domain Fine-Tuning

This guide shows how to fine-tune a model on medical knowledge using three different algorithms. Medical fine-tuning is a representative example of domain adaptation — the same approach works for legal, financial, or any other specialized domain.

## Algorithm Comparison for Medical Data

| Approach | Preserves general knowledge | GPU requirement | Best when |
|----------|---------------------------|-----------------|-----------|
| [SFT](#sft-approach) | No | 2-4x A100 | Maximum medical accuracy is the only goal |
| [OSFT](#osft-approach) | **Yes** | 2-4x A100 | Model must also handle general queries |
| [LoRA](#lora-approach) | Partially | 1x A100 | Limited GPU budget |

## Preparing Medical Data

Use SDG Hub to generate medical Q&A pairs from clinical guidelines, textbooks, or research papers:

```python
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()

medical_docs = Dataset.from_dict({
    "document": [
        "Type 2 diabetes mellitus (T2DM) management includes lifestyle "
        "modifications and pharmacotherapy. First-line treatment is metformin, "
        "initiated at 500mg daily with gradual titration to 2000mg...",
    ],
    "document_outline": ["T2DM treatment guidelines and pharmacotherapy"],
    "domain": ["medical"],
    "icl_document": [""],
    "icl_query_1": [""],
    "icl_query_2": [""],
    "icl_query_3": [""],
})

flow = Flow.from_yaml(FlowRegistry.get_flow_path(
    "Document Based Knowledge Tuning Dataset Generation Flow"
))
flow.set_model_config(model="gpt-4o-mini")
result = flow.generate(medical_docs)
result_df = result.to_pandas() if hasattr(result, "to_pandas") else result
result_df.to_json("medical_training_data.jsonl", orient="records", lines=True)
```

## SFT Approach

Maximum learning capacity. Best when the model will be used exclusively for medical tasks.

```python
from training_hub import sft

sft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="medical_training_data.jsonl",
    ckpt_output_dir="./medical-sft",
    num_epochs=5,
    effective_batch_size=32,
    max_seq_len=4096,
    learning_rate=2e-5,
)
```

## OSFT Approach

Adds medical knowledge while preserving general capabilities. **Recommended** when the model serves both medical and general queries.

```python
from training_hub import osft

osft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="medical_training_data.jsonl",
    ckpt_output_dir="./medical-osft",
    unfreeze_rank_ratio=0.01,
    effective_batch_size=32,
    max_tokens_per_gpu=16384,
    max_seq_len=4096,
    learning_rate=2e-5,
    num_epochs=5,
)
```

## LoRA Approach

Memory-efficient fine-tuning. Best when you have limited GPU resources or want to maintain multiple specialty adapters.

```python
from training_hub import lora_sft

lora_sft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="medical_training_data.jsonl",
    ckpt_output_dir="./medical-lora",
    num_epochs=5,
    lora_r=32,
    lora_alpha=64,
    max_seq_len=4096,
)
```

!!! tip "Multiple Specialties"
    With LoRA, you can maintain separate adapters for cardiology, oncology, endocrinology, etc. — all sharing the same base model. Swap adapters at inference time based on the query domain.

## Related

- [Choosing an Algorithm](../getting-started/choosing-an-algorithm.md) — General algorithm selection guide
- [OSFT](../training/osft.md) — Recommended for medical (preserves general knowledge)
- [Continual Learning](../training/continual-learning.md) — Add multiple medical specialties sequentially
