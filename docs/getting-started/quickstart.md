# Setup & First Pipeline

This page walks through environment setup, package installation, and a minimal end-to-end pipeline: generate training data from a document, train a model, and verify convergence.

!!! info "This quickstart follows the Knowledge Track"
    The pipeline below demonstrates **knowledge tuning** — generating Q&A training data from documents and fine-tuning with SFT, OSFT, or LoRA. If your goal is to fine-tune a model for **tool calling** (MCP servers, APIs), start with the [Tool-Calling Model Pipeline](../end-to-end/tool-calling-financial.md) instead.

For conceptual background on model customization, see the [Overview](overview.md).

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10+ |
| **GPU** | NVIDIA A100 (80GB) or L40 for training. Not required for data generation. |
| **LLM API key** | OpenAI, Anthropic, or any [LiteLLM-supported provider](https://docs.redhat.com/en/documentation/red_hat_openshift_ai/3.4) for synthetic data generation |
| **Model access** | Hugging Face token for gated models (Llama, Mistral) |

## Environment Options

=== "RHOAI Workbench"

    SDG Hub and Training Hub are pre-installed in Red Hat-curated workbench images. Launch a workbench from the RHOAI dashboard and select an image with GPU support.

    ```bash
    # Verify installation
    python -c "import sdg_hub; print(sdg_hub.__version__)"
    python -c "import training_hub; print(training_hub.__version__)"
    ```

=== "Local / Custom Environment"

    ```bash
    pip install sdg-hub[examples] training-hub datasets
    ```

    For QLoRA (4-bit quantized training):

    ```bash
    pip install bitsandbytes
    ```

## Configure API Keys

SDG Hub uses LiteLLM under the hood, which reads standard environment variables:

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

For gated Hugging Face models:

```bash
export HF_TOKEN="hf_..."
```

## Check GPU Resources

Before training, verify you have enough VRAM:

```python
from training_hub import estimate

lower, expected, upper = estimate(
    training_method="lora",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    num_gpus=1,
    max_seq_len=4096,
)
print(f"Estimated VRAM: {expected / 1e9:.1f} GB (range: {lower / 1e9:.1f}–{upper / 1e9:.1f} GB)")
```

See [GPU Requirements](../reference/gpu-requirements.md) for per-model and per-algorithm breakdowns.

## Step 1: Prepare Seed Data

Training data generation starts with a dataset containing your domain documents. Each row needs a `document` column with the text content, plus additional columns required by the knowledge flow (`document_outline`, and optionally `icl_document`, `icl_query_1/2/3` for in-context learning examples):

```python
from datasets import Dataset

document_text = (
    "RHOAI 3.4 brings Models-as-a-Service (MaaS) to GA, allowing users "
    "to access hosted LLMs directly from the dashboard without deploying "
    "model servers. Supported providers include OpenAI, Anthropic, and "
    "IBM watsonx. MaaS endpoints are rate-limited and metered per-token."
)

seed_data = Dataset.from_dict({
    "document": [document_text],
    "document_outline": ["Overview of RHOAI 3.4 Models-as-a-Service feature"],
    "domain": ["rhoai"],
    "icl_document": [""],
    "icl_query_1": [""],
    "icl_query_2": [""],
    "icl_query_3": [""],
})
```

For real workloads, use [Docling](https://docs.redhat.com/en/documentation/red_hat_openshift_ai/3.4) to extract text from PDFs, HTML, or DOCX files. See [Knowledge Tuning Data Generation](../data-generation/knowledge-tuning.md) for details.

## Step 2: Generate Training Data

Use SDG Hub to generate Q&A pairs from your documents:

```python
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()

flow_path = FlowRegistry.get_flow_path(
    "Document Based Knowledge Tuning Dataset Generation Flow"
)
if flow_path is None:
    raise RuntimeError("Flow not found. Install: pip install sdg-hub[examples]")

flow = Flow.from_yaml(flow_path)
flow.set_model_config(model="gpt-4o-mini")

result = flow.generate(seed_data)
result_df = result.to_pandas() if hasattr(result, "to_pandas") else result
result_df.to_json("training_data.jsonl", orient="records", lines=True)
print(f"Generated {len(result_df)} training examples")
```

!!! tip "Dry run first"
    Use `flow.dry_run(seed_data)` to execute the pipeline on a small subset and catch configuration errors before processing the full dataset.

!!! warning "SDG Hub output needs conversion"
    Knowledge tuning flows output `question` and `response` columns — **not** `messages` format. The code above saves the raw output. Before training, you must convert to `messages` format:

    ```python
    import pandas as pd, json
    raw = pd.read_json("training_data.jsonl", lines=True)
    converted = [{"messages": [
        {"role": "user", "content": str(r["question"])},
        {"role": "assistant", "content": str(r["response"])}
    ], "unmask": True} for _, r in raw.iterrows() if "question" in r and "response" in r]
    with open("training_data.jsonl", "w") as f:
        for rec in converted:
            f.write(json.dumps(rec) + "\n")
    print(f"Converted {len(converted)} examples to messages format")
    ```

See [Data Formats](../reference/data-formats.md) for the full specification.

## Step 3: Train

Choose an algorithm based on your constraints. Each tab shows minimal working code — see the individual [training guides](../training/sft.md) for all parameters.

=== "SFT — Maximum learning"

    Updates all parameters. Best when you have ample data and GPU.

    ```python
    from training_hub import sft

    sft(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        data_path="training_data.jsonl",
        ckpt_output_dir="./my-model",
        num_epochs=4,
        effective_batch_size=32,
        max_seq_len=4096,
    )
    ```

=== "OSFT — Preserve base knowledge"

    Constrains updates to an orthogonal subspace. Best for adding domain knowledge without forgetting.

    ```python
    from training_hub import osft

    osft(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        data_path="training_data.jsonl",
        ckpt_output_dir="./my-model",
        num_epochs=4,
        effective_batch_size=32,
        max_seq_len=4096,
        learning_rate=2e-5,
        max_tokens_per_gpu=16384,
        unfreeze_rank_ratio=0.25,
    )
    ```

=== "LoRA — Single GPU"

    Trains ~1% of parameters via low-rank adapters. Runs on a single A100 or L40.

    ```python
    from training_hub import lora_sft

    lora_sft(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        data_path="training_data.jsonl",
        ckpt_output_dir="./my-model",
        num_epochs=4,
        lora_r=16,
        lora_alpha=32,
    )
    ```

Not sure which to pick? See [Choosing an Algorithm](choosing-an-algorithm.md).

## Step 4: Verify Training

Check that training loss converged:

```python
from training_hub import plot_loss

plot_loss("./my-model")
```

A healthy training run shows loss decreasing and flattening. If loss plateaus early or spikes, adjust learning rate or add more data. See [Plot Loss](../utilities/plot-loss.md) for interpretation guidance.

## What to Read Next

[Choosing an Algorithm](choosing-an-algorithm.md) — Decision flowchart and side-by-side comparison of SFT, OSFT, LoRA, and GRPO

Then pick a track:

=== "Knowledge Track"

    Teach a model your domain knowledge (financial regulations, medical literature, product docs):

    1. [Knowledge Tuning Pipeline](../end-to-end/knowledge-tuning.md) — Full end-to-end walkthrough
    2. [Deploy & Serve](../serving/index.md) — KServe + vLLM deployment

=== "Tool-Calling Track"

    Fine-tune a model to call tools from MCP servers and APIs:

    1. [Tool-Calling Model Pipeline](../end-to-end/tool-calling-financial.md) — Validated end-to-end on RHOAI 3.4.2 (MCP distillation + LoRA SFT + vLLM serving + guardrails), uses financial services as the example domain
    2. [MCP Distillation](../end-to-end/mcp-distillation.md) — Generic MCP distillation pipeline

[GPU Requirements](../reference/gpu-requirements.md) — Per-model VRAM estimates and hardware guidance
