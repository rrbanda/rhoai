# End-to-End Knowledge Tuning Pipeline

This walkthrough takes you through the complete model customization lifecycle: preparing documents, generating synthetic training data, training a model, evaluating it, and preparing for deployment.

## Pipeline Overview

```mermaid
graph TD
    A[1. Prepare Documents] --> B[2. Generate Training Data<br/>SDG Hub]
    B --> C[3. Mix & Validate Data]
    C --> D[4. Train Model<br/>Training Hub]
    D --> E[5. Evaluate]
    E -->|"Iterate"| B
    E -->|"Ready"| F[6. Serve on RHOAI]
```

## Step 1: Prepare Documents

Convert your source documents to structured text. Use [Docling](https://github.com/DS4SD/docling) for PDFs and web pages:

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

sources = [
    "https://docs.example.com/product-guide.html",
    "/path/to/technical-manual.pdf",
]

documents = []
for source in sources:
    result = converter.convert(source)
    documents.append(result.document.export_to_markdown())

print(f"Prepared {len(documents)} documents")
```

Create a seed dataset with document chunks:

```python
from datasets import Dataset

chunks = []
for doc in documents:
    for i in range(0, len(doc), 1500):
        chunks.append(doc[i:i+2000])

dataset = Dataset.from_dict({
    "document": chunks,
    "domain": ["your-domain"] * len(chunks),
})

print(f"Created {len(dataset)} document chunks")
```

## Step 2: Generate Training Data

Run multiple SDG Hub knowledge tuning flow variants for diverse training examples:

```python
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()

FLOW_VARIANTS = {
    "extractive_summary": "Extractive Summary Knowledge Tuning Dataset Generation Flow",
    "detailed_summary": "Detailed Summary Knowledge Tuning Dataset Generation Flow",
    "key_facts": "Key Facts Knowledge Tuning Dataset Generation Flow",
    "doc_direct_qa": "Document Based Knowledge Tuning Dataset Generation Flow",
}

generated_data = {}
for variant_name, flow_display_name in FLOW_VARIANTS.items():
    flow = Flow.from_yaml(FlowRegistry.get_flow_path(flow_display_name))
    flow.set_model_config(model="gpt-4o-mini")
    result = flow.generate(dataset)
    result.to_json(f"{variant_name}.jsonl", orient="records", lines=True)
    generated_data[variant_name] = result
    print(f"{variant_name}: {len(result)} examples")
```

## Step 3: Mix and Validate

Combine all variants into a single, deduplicated training set:

```python
import pandas as pd

dfs = [pd.read_json(f"{name}.jsonl", lines=True) for name in FLOW_VARIANTS]
combined = pd.concat(dfs, ignore_index=True)
combined = combined.drop_duplicates(subset=["messages"])
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

combined.to_json("training_data.jsonl", orient="records", lines=True)
print(f"Final training set: {len(combined)} examples")
```

## Step 4: Train

Choose your algorithm based on the [decision guide](../getting-started/choosing-an-algorithm.md):

=== "OSFT (Recommended)"

    Preserves base knowledge while adding domain expertise:

    ```python
    from training_hub import osft

    osft(
        model="meta-llama/Llama-3.1-8B-Instruct",
        data="training_data.jsonl",
        output_dir="./knowledge-model",
        num_epochs=4,
        batch_size=32,
        max_seq_len=4096,
        unfreeze_rank_ratio=0.01,
    )
    ```

=== "SFT (Maximum Capacity)"

    Full parameter update for maximum learning:

    ```python
    from training_hub import sft

    sft(
        model="meta-llama/Llama-3.1-8B-Instruct",
        data="training_data.jsonl",
        output_dir="./knowledge-model",
        num_epochs=4,
        batch_size=32,
        max_seq_len=4096,
    )
    ```

=== "LoRA (Memory Efficient)"

    Single-GPU fine-tuning:

    ```python
    from training_hub import lora_sft

    lora_sft(
        model="meta-llama/Llama-3.1-8B-Instruct",
        data="training_data.jsonl",
        output_dir="./knowledge-model",
        num_epochs=4,
        lora_r=16,
        lora_alpha=32,
    )
    ```

## Step 5: Evaluate

### Training Loss

```python
from training_hub import plot_loss

plot_loss("./knowledge-model")
```

### Domain Accuracy

Generate a held-out evaluation set and test the model:

```python
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()
flow = Flow.from_yaml(FlowRegistry.get_flow_path(
    "Document Based Knowledge Tuning Dataset Generation Flow"
))
flow.set_model_config(model="gpt-4o-mini")
eval_data = flow.generate(held_out_dataset)
```

Compare the fine-tuned model's answers against the teacher's answers using LLM-as-judge or exact match metrics.

## Step 6: Serve on RHOAI

Deploy the trained model via KServe with vLLM runtime:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: knowledge-model
spec:
  predictor:
    model:
      modelFormat:
        name: vLLM
      runtime: vllm-runtime
      storageUri: s3://models/knowledge-model
      resources:
        limits:
          nvidia.com/gpu: "1"
```

## Full Example

The complete pipeline is available as a runnable Jupyter notebook:

- **Notebook:** [`model_customization_e2e.ipynb`](https://github.com/rrbanda/rhoai/blob/main/end-to-end-examples/knowledge-tuning/model_customization_e2e.ipynb)
- **Scripts:** [`end-to-end-examples/knowledge-tuning/examples/`](https://github.com/rrbanda/rhoai/tree/main/end-to-end-examples/knowledge-tuning/examples)

## Related

- [Choosing an Algorithm](../getting-started/choosing-an-algorithm.md) — Algorithm selection guide
- [Knowledge Tuning Data](../data-generation/knowledge-tuning.md) — Detailed data generation guide
- [MCP Distillation Pipeline](mcp-distillation.md) — Alternative pipeline for tool-use training
