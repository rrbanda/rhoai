# Financial Domain Fine-Tuning

This guide covers fine-tuning models for financial services — from knowledge-based Q&A (investment research, regulatory compliance) to tool-calling agents (portfolio management, trade execution). Financial fine-tuning adds two challenges beyond general domain adaptation: **regulatory compliance** and **tool-use accuracy**.

## Two Paths for Financial Models

| Path | Algorithm | Use Case | Example |
|------|-----------|----------|---------|
| **Knowledge injection** | SFT / OSFT | Answer questions about financial products, regulations, markets | "Explain the wash-sale rule" |
| **Tool-calling agent** | LoRA SFT | Call financial APIs, execute trades, manage portfolios | "Buy 100 shares of AAPL in my portfolio" |

For a full end-to-end worked example of the tool-calling path, see the [Tool-Calling Model Pipeline](../end-to-end/financial-agent.md).

## Path 1: Financial Knowledge Tuning

### Prepare Financial Documents

Use Docling to extract text from SEC filings, compliance manuals, product documentation, and research reports:

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
sources = [
    "/path/to/compliance-manual.pdf",
    "/path/to/product-terms.pdf",
    "/path/to/market-research.pdf",
]

documents = []
for source in sources:
    result = converter.convert(source)
    documents.append(result.document.export_to_markdown())
```

### Generate Financial Training Data

```python
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()

financial_docs = Dataset.from_dict({
    "document": documents,
    "domain": ["financial"] * len(documents),
})

flow = Flow.from_yaml(FlowRegistry.get_flow_path(
    "Document Based Knowledge Tuning Dataset Generation Flow"
))
flow.set_model_config(model="gpt-4o-mini")
result = flow.generate(financial_docs)
result.to_json("financial_training_data.jsonl", orient="records", lines=True)
```

### Train with OSFT (Recommended)

OSFT is recommended for financial knowledge because it preserves the model's general reasoning ability while adding domain expertise:

```python
from training_hub import osft

osft(
    model_path="Qwen/Qwen3-4B",
    data_path="financial_training_data.jsonl",
    ckpt_output_dir="./financial-knowledge-model",
    unfreeze_rank_ratio=0.01,
    effective_batch_size=32,
    max_tokens_per_gpu=16384,
    max_seq_len=4096,
    learning_rate=2e-5,
    num_epochs=4,
)
```

## Path 2: Financial Tool-Calling Agent

Train a model to call financial APIs using MCP distillation + LoRA SFT. This path produces an agent that can manage portfolios, execute trades, and perform risk analysis.

### Set Up a Financial MCP Server

Create or use a financial MCP server with tools organized into functional domains:

| Domain | Tools | Purpose |
|--------|-------|---------|
| Market Data | `get_stock_quote`, `get_market_summary`, `get_historical_prices`, `screen_stocks` | Real-time and historical market information |
| Portfolio Management | `get_portfolio_positions`, `get_portfolio_performance`, `get_account_summary`, `get_transaction_history` | Client account and holdings management |
| Risk & Analytics | `calculate_portfolio_risk`, `get_sector_exposure`, `run_stress_test`, `analyze_stock` | Risk assessment and analysis |
| Trading & Compliance | `submit_trade_order`, `check_compliance`, `get_regulatory_status` | Trade execution with pre-trade validation |

A complete demo server is included in the [financial-agent example](https://github.com/rrbanda/rhoai/tree/main/end-to-end-examples/financial-agent/demo_server).

### Generate Tool-Use Training Data

Use SDG Hub's MCP distillation flow to generate expert tool-use traces:

```python
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()
flow = Flow.from_yaml(FlowRegistry.get_flow_path("MCP Server Distillation"))
flow.set_model_config(model="gemini/gemini-3.6-flash", api_key="...")

flow.set_agent_config(
    agent_framework="langflow",
    agent_url="http://localhost:7860/api/v1/run/your-flow-id",
)

result = flow.generate(tool_dataset)
result.to_parquet("distillation_output.parquet")
```

### Train with LoRA SFT

LoRA SFT trains the model on expert tool-calling demonstrations. The model learns which tool to call for a given query, how to format arguments, and how to chain tools for multi-step workflows:

```python
from training_hub import lora_sft

lora_sft(
    model_path="Qwen/Qwen3-4B",
    data_path="training_data.jsonl",
    ckpt_output_dir="./financial-agent",
    lora_r=16,
    lora_alpha=32,
    num_epochs=2,
    learning_rate=2e-4,
    load_in_4bit=True,
)
```

For production on RHOAI, use the [LoRA KFP Pipeline](https://github.com/red-hat-data-services/pipelines-components/tree/main/pipelines/training/finetuning/lora) which wraps the same algorithm in a four-stage pipeline (data download, training, evaluation, model registry).

### Deploy and Serve

After training, deploy the model with tool-calling support via KServe + vLLM. Two options:

- **LoRA adapter serving** (recommended) — Serve the adapter directly from the training PVC without merging
- **Merged model** — Merge adapter into base model and deploy from S3

Full YAML manifests and step-by-step deployment instructions are in the [Tool-Calling Model Pipeline Step 4](../end-to-end/financial-agent.md#step-4-deploy-the-fine-tuned-model-on-rhoai) and the [`serving/` directory](https://github.com/rrbanda/rhoai/tree/main/end-to-end-examples/financial-agent/serving).

### Add Financial Guardrails

Financial agents require compliance rails. See [Guardrails](../guardrails/index.md) for full details. Key rails for financial services:

- **PII detection** — Mask account numbers, SSNs, routing numbers
- **Pre-trade compliance** — Block trades that violate concentration limits or restricted lists
- **Disclaimer injection** — Append regulatory disclaimers to investment advice
- **Trade size validation** — Enforce risk tolerance thresholds

## Financial-Specific Considerations

!!! warning "Regulatory Compliance"
    Financial models deployed in production must comply with applicable regulations (SEC, FINRA, MiFID II). Guardrails are a technical control, not a substitute for regulatory review. Work with your compliance team before deploying.

!!! tip "Ambiguous Tool Selection"
    Financial APIs often have overlapping functionality (e.g., `get_portfolio_positions` vs `get_account_summary`). Ensure your training data includes diverse examples covering these overlapping tools so the model learns the correct tool for each context. For reinforcement-learning-based training (GRPO), see the [LoRA GRPO docs](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/docs/algorithms/lora_grpo.md) — GRPO KFP pipeline support is planned.

!!! tip "Deterministic Test Data"
    Use fixed random seeds for financial test data so evaluations are reproducible. The [demo server](https://github.com/rrbanda/rhoai/tree/main/end-to-end-examples/financial-agent/demo_server) uses `SEED=42` for all generated data.

## Validated Environment

!!! success "Tested on RHOAI 3.4.2"
    The tool-calling model path (MCP distillation → LoRA SFT → vLLM deployment) has been validated end-to-end on RHOAI 3.4.2 / OpenShift 4.18.21 with an NVIDIA L4 GPU. The full walkthrough is in the [Tool-Calling Model Pipeline](../end-to-end/financial-agent.md).

## Related

- [Tool-Calling Model Pipeline](../end-to-end/financial-agent.md) — Full end-to-end walkthrough
- [Training Hub LoRA docs](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/docs/algorithms/lora.md) — LoRA SFT algorithm details
- [Guardrails](../guardrails/index.md) — NeMo Guardrails configuration
- [Medical Domain](medical.md) — Similar approach for medical knowledge
- [MCP Distillation](../end-to-end/mcp-distillation.md) — General MCP distillation pipeline
