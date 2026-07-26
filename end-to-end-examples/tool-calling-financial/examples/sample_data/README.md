# Sample Data

Pre-generated distillation output for validating the pipeline without Langflow or API keys.

## Contents

| File | Description |
|------|-------------|
| `distillation_output.parquet` | 10 tool-calling examples covering 13/15 financial tools across all 4 domains |

## Usage

```bash
# Format the sample data (Step 2)
python 02_format_training_data.py \
    --input-file sample_data/distillation_output.parquet \
    --output-dir sample_data

# Then continue with Steps 3-6 using sample_data/training_data.jsonl
```

## When to generate your own data

Use this sample data to validate that the training, deployment, and serving pipeline works end-to-end on your RHOAI cluster. Once validated, generate your own data with `01_generate_tool_data.py` (requires Langflow + teacher model API key) for production-quality training.
