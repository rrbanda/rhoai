# What is Model Customization?

Model customization is the process of adapting a pre-trained language model to perform well on your specific domain or task. Instead of training from scratch (which requires enormous compute), you take an existing model and teach it new knowledge, skills, or behaviors using your data.

## Why Smaller Models?

Large frontier models (70B+ parameters) are powerful but expensive to serve. A well-customized smaller model (3B–8B parameters) can **match or exceed frontier model performance** on your specific task while being:

- **Cheaper to serve** — 4-10x lower inference cost
- **Faster** — Lower latency for real-time applications
- **Deployable on-premise** — Runs on a single GPU node
- **Private** — Your data never leaves your infrastructure

## The RHOAI Model Customization Stack

The RHOAI model customization workflow uses two open-source Python libraries, pre-installed in RHOAI training and workbench images. These are community libraries maintained by the Red Hat AI Innovation Team — not official RHOAI product components with GA support SLAs. The underlying platform capabilities (Kubeflow Trainer, KServe, AI Pipelines) carry full Red Hat support.

### SDG Hub — Synthetic Data Generation

SDG Hub generates high-quality training data from your raw documents. A teacher LLM (like GPT-4 or Claude) reads your documents and generates question-answer pairs, summaries, and other training examples. Knowledge tuning flows output `question`/`response` columns which must be converted to `messages` format before training (see [Data Formats](../reference/data-formats.md)).

```python
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()
flow = Flow.from_yaml(FlowRegistry.get_flow_path("Document Based Knowledge Tuning Dataset Generation Flow"))
flow.set_model_config(model="gpt-4o-mini", api_key="...")
result = flow.generate(dataset)
```

### Training Hub — Model Fine-Tuning

Training Hub provides multiple fine-tuning algorithms optimized for different use cases:

```python
from training_hub import sft, osft, lora_sft

sft(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="training_data.jsonl",
    ckpt_output_dir="./output",
)
```

## The Pipeline

Every model customization project follows the same high-level flow:

```mermaid
graph LR
    A[1. Prepare<br/>Documents] --> B[2. Generate<br/>Data]
    B --> C[3. Choose<br/>Algorithm]
    C --> D[4. Train]
    D --> E[5. Evaluate]
    E -->|Iterate| B
    E -->|Ready| F[6. Deploy]
```

1. **Prepare** — Gather your domain documents (PDFs, Markdown, web pages)
2. **Generate** — Use SDG Hub to create synthetic training data from your documents
3. **Choose** — Pick the right [training algorithm](choosing-an-algorithm.md) for your use case
4. **Train** — Fine-tune the model with Training Hub
5. **Evaluate** — Measure performance against your benchmarks
6. **Deploy** — Serve the model via KServe / vLLM on RHOAI

## Next Steps

Ready to try it? Continue to [Setup & First Pipeline](quickstart.md) for environment setup and your first training run.

- [Choosing an Algorithm](choosing-an-algorithm.md) — Decide between SFT, OSFT, LoRA, and GRPO
- [Knowledge Tuning Pipeline](../end-to-end/knowledge-tuning.md) — Full end-to-end walkthrough
- [Tool-Calling Model Pipeline](../end-to-end/tool-calling-financial.md) — Fine-tune a model for tool-calling with MCP distillation + LoRA SFT (validated on RHOAI 3.4.2, financial services example)
