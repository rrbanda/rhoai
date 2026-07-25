# MLflow

## Status

**GA** (General Availability) — New in RHOAI 3.4

## Overview

MLflow provides experiment tracking, run comparison, and model versioning within Red Hat OpenShift AI. It is deployed and managed through the Red Hat-built MLflow Operator. MLflow is integrated into the RHOAI dashboard and included in all workbench images, with support for W&B, MLflow tracking, and TensorBoard loggers out of the box.

## What's Covered

- Deploying MLflow via the MLflow Operator
- Tracking experiments and logging metrics, parameters, and artifacts
- Comparing runs across experiments
- Model versioning through the MLflow model registry
- Dashboard integration for viewing experiment results
- Configuring experiment tracking loggers (W&B, MLflow, TensorBoard)
- Using MLflow from workbench notebooks
- **Training Hub integration** — SFT and OSFT with automatic MLflow logging

## Official Documentation

- [Working with MLflow](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/working_with_mlflow)

## MLflow + Training Hub Integration

Training Hub provides native MLflow integration. When you pass `mlflow_tracking_uri`, `mlflow_experiment_name`, and `mlflow_run_name` to the `sft()` or `osft()` functions, Training Hub automatically:

1. **Logs hyperparameters** — learning rate, batch size, epochs, sequence length, rank ratio (OSFT)
2. **Logs metrics** — training loss, validation loss, learning rate schedule per step
3. **Saves checkpoints** — model artifacts are logged to the MLflow artifact store
4. **Creates runs** — each training invocation creates a tracked run under the specified experiment

This eliminates boilerplate MLflow instrumentation code and ensures consistent tracking across all training jobs.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow server endpoint | `http://localhost:5000` |
| `MLFLOW_EXPERIMENT_NAME` | Experiment name for grouping runs | `sft-training` / `osft-training` |
| `MODEL_PATH` | HuggingFace model ID or local path | `Qwen/Qwen2.5-7B-Instruct` |
| `CKPT_OUTPUT_DIR` | Directory for model checkpoints | `./sft-mlflow-checkpoints` |

## What's in examples/

| File | Description |
|------|-------------|
| [`sft_with_mlflow.py`](examples/sft_with_mlflow.py) | SFT training with MLflow experiment tracking — logs params, metrics, and model artifacts |
| [`osft_with_mlflow.py`](examples/osft_with_mlflow.py) | OSFT (Orthogonal Subspace Fine-Tuning) with MLflow — preserves general capabilities while adding domain knowledge |
| [`requirements.txt`](examples/requirements.txt) | Python dependencies for running the examples |

### Quick Start

```bash
# Install dependencies
pip install -r examples/requirements.txt

# Start MLflow server (or use the RHOAI-managed instance)
mlflow server --host 0.0.0.0 --port 5000

# Run SFT with MLflow tracking
python examples/sft_with_mlflow.py \
    --data-path /path/to/training_data.jsonl \
    --mlflow-uri http://localhost:5000

# Run OSFT with MLflow tracking
python examples/osft_with_mlflow.py \
    --data-path /path/to/training_data.jsonl \
    --unfreeze-rank-ratio 0.1

# View results
# Open http://localhost:5000 in your browser
```

### SFT vs OSFT

| Method | Use Case | Key Difference |
|--------|----------|----------------|
| **SFT** | General instruction tuning, chat fine-tuning | Full parameter updates |
| **OSFT** | Domain adaptation without forgetting | Constrains updates to orthogonal subspace via `unfreeze_rank_ratio` |

Both methods support the same MLflow integration pattern — the only difference is the training algorithm and OSFT-specific parameters.
