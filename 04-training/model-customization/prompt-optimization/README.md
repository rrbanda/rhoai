# GEPA Prompt Optimization

Optimize textual prompts using evolutionary search -- no model weight
modifications required.

## Overview

GEPA (Genetic-Pareto) uses evolutionary search with Pareto-based selection
and LLM-driven reflection to evolve prompts that maximise task performance.
Unlike weight-training algorithms (SFT, OSFT, LoRA, GRPO), GEPA optimises
the **prompt itself**:

- No GPU required for training (only for serving the model)
- No model weights are modified
- Useful for improving system prompts, few-shot templates, and agent instructions
- Works with any LLM accessible via an API endpoint

## Prerequisites

```bash
pip install 'training-hub[gepa]'
```

A running vLLM server (or any OpenAI-compatible endpoint):

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.5
```

## Quick Start

```bash
# Run with default GEPA backend
python examples/gepa_prompt_optimization.py \
    --data-path ./qa_data.jsonl \
    --output-dir ./gepa_output

# Use MLflow backend for experiment tracking
python examples/gepa_prompt_optimization.py \
    --data-path ./qa_data.jsonl \
    --output-dir ./gepa_output \
    --backend mlflow
```

## Data Format

GEPA expects JSONL with `input` and `answer` fields:

```json
{"input": "What is 2 + 3?", "answer": "5"}
{"input": "A store sells apples for $3 each. If you buy 4, how much?", "answer": "12"}
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed-prompt` | `"Answer the question."` | Starting prompt to optimise |
| `--max-metric-calls` | `300` | Budget for evaluation calls (>= 15x dataset size) |
| `--reflection-minibatch-size` | `5` | Examples examined per reflection step |
| `--model` | `openai/Qwen/Qwen2.5-3B-Instruct` | Task and reflection model |
| `--api-base` | `http://localhost:8000/v1` | vLLM or OpenAI-compatible endpoint |

## Tips

- **Reflection model size**: Must be >= 3B parameters for coherent reflections.
  Models under 3B produce incoherent output instead of improved prompts.
- **Budget rule**: Use `max_metric_calls >= 15 * dataset_size` for meaningful
  optimisation.
- **Baseline accuracy**: Target 20-40% baseline so GEPA has enough failing
  examples to learn from.  If baseline is already 80%+, increase
  `--reflection-minibatch-size`.
