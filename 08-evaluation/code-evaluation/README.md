# Code Evaluation

Generate execution-verified coding benchmarks and evaluate model performance using SDG Hub's code interpreter.

## Overview

This section provides two complementary workflows:

1. **Benchmark Generation** — Create domain-specific coding benchmarks where every problem is backed by a verified reference solution and test suite
2. **Model Evaluation** — Evaluate coding models on the generated benchmark and compute pass@1 scores

### Benchmark Generation Pipeline

Inspired by [AutoCodeBench](https://arxiv.org/abs/2508.09101), the pipeline generates problems in reverse: code is written and verified *first*, then the problem description is generated from the working code.

```
Domain + Function Specs (seed data)
  → LLM generates function implementation       (Phase 1)
  → LLM generates assert-based test suite        (Phase 2)
  → PythonInterpreterBlock verifies execution     (Phase 3)
  → LLM reverse-generates problem description     (Phase 4)
  → Verified benchmark: (problem, solution, tests)
```

Every benchmark problem is backed by a solution and test suite that provably executes — no hallucinated test cases, no broken reference solutions.

### Model Evaluation Pipeline

```
Benchmark (problem_description, test_code)
  → Model generates function from problem description
  → PythonInterpreterBlock runs tests against model output
  → pass@1 = fraction of problems where all tests pass
```

## Prerequisites

- Python 3.10+
- SDG Hub with code interpreter: `pip install sdg-hub[code]`
- An LLM endpoint (OpenAI, vLLM, or any LiteLLM-compatible provider)

## Environment Variables

```bash
export INFERENCE_MODEL="openai/gpt-4o"       # Model for benchmark generation
export URL="https://your-endpoint/v1"          # API base URL (optional for native OpenAI)
export API_KEY="your-api-key"                  # API key
```

## Usage

### Step 1: Generate a Benchmark

```bash
python examples/generate_code_benchmark.py \
  --output domain_code_benchmark.jsonl
```

### Step 2: Evaluate Models

```bash
python examples/evaluate_code_models.py \
  --benchmark domain_code_benchmark.jsonl \
  --models gpt-4o gpt-4o-mini \
  --output evaluation_results.jsonl
```

## Benchmark Format

Each record in the generated JSONL contains:

| Field | Description |
|-------|-------------|
| `domain` | Problem domain (e.g., "financial calculations", "algorithms") |
| `difficulty` | Difficulty level: beginner, intermediate, advanced |
| `function_spec` | Original specification used to generate the problem |
| `problem_description` | What the evaluated model sees (generated from verified code) |
| `function_code` | Reference solution (hidden from the model) |
| `test_code` | Assert-based test suite for grading |
| `time_complexity` | Expected algorithmic complexity |

## What's in examples/

- `generate_code_benchmark.py` — Generate an execution-verified coding benchmark from domain and function specifications using the `domain-code-eval` flow
- `evaluate_code_models.py` — Evaluate one or more coding models against a generated benchmark and report pass@1 scores
