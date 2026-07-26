# Sample Data

Pre-generated MCP distillation output for validating the pipeline without Langflow or API keys.

## Contents

| File | Description |
|------|-------------|
| `distillation_output.jsonl` | 10 tool-calling examples covering 13/15 ShopInsights tools across all 4 ambiguity clusters |
| `training_data.jsonl` | Formatted training data in modern `tool_calls` format, ready for GRPO training |

## Usage

```bash
# Skip Steps 1-3 — use pre-formatted training data directly
python 03_train_grpo.py --data-path sample_data/training_data.jsonl

# Or format the raw output yourself (Step 2-3)
python 02_format_training_data.py \
    --input-file sample_data/distillation_output.jsonl \
    --output-dir sample_data
```

## When to generate your own data

Use this sample data to validate that the GRPO training and deployment pipeline works end-to-end on your RHOAI cluster. Once validated, generate your own data with `01_generate_tool_data.py` (requires Langflow + teacher model API key) for production-quality training.
