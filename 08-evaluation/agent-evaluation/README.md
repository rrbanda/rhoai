# Agent / MCP Evaluation

Generate synthetic MCP tool-use benchmarks and evaluate agent performance across different LLMs using LLM-as-judge scoring.

## Overview

This section provides two complementary workflows:

1. **Benchmark Generation** — Create synthetic evaluation tasks from MCP server tool schemas using a frontier model to explore tools, generate grounded questions, and produce expert-quality gold-standard trajectories
2. **Model Evaluation** — Evaluate multiple LLMs on the generated benchmark by swapping the model behind the same agent, then scoring tool-use quality with an LLM judge

### Benchmark Generation Pipeline

```
MCP Server Tool Schemas
  → Frontier model explores tools and discovers capabilities
  → Generates grounded questions at varying complexity levels
  → Expert model solves each task (gold-standard trajectories)
  → benchmark_tasks.jsonl
```

### Evaluation Pipeline

```
benchmark_tasks.jsonl
  → Agent (model swapped via configurable) solves each task
  → Programmatic metrics: tool recall, precision, order match, parameter match
  → LLM-as-judge scores: task fulfillment, grounding, tool appropriateness,
    parameter accuracy, dependency awareness, parallelism/efficiency
  → Rankings and comparison across models
```

The key idea: the **same agent** used for generation is used for evaluation — only the underlying LLM is swapped. This evaluates the full agent stack (tools, guardrails, orchestration), not just the bare model.

## Prerequisites

- Python 3.10+
- SDG Hub installed: `pip install sdg-hub[examples]`
- MCP servers running (for benchmark generation)
- LangGraph agents running (connected to MCP servers)
- An LLM endpoint for the teacher model and judge

## Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-api-key"

# Benchmark generation
export TEACHER_MODEL="openai/gpt-5.2"         # Frontier model for exploration + question generation

# Evaluation
export JUDGE_MODEL="openai/gpt-4o"             # LLM-as-judge model

# Agent URLs (one per MCP server)
export LANGGRAPH_URL_WEATHER_DATA="http://localhost:2024"
export LANGGRAPH_URL_MEDICAL_CALCULATOR="http://localhost:2025"
export LANGGRAPH_URL_WIKIPEDIA="http://localhost:2026"
# ... add more as needed
```

## Usage

### Step 1: Generate a Benchmark

```bash
python examples/generate_agent_benchmark.py \
  --mcp-servers '{"Weather Data": "http://localhost:8001/mcp", "Wikipedia": "http://localhost:8003/mcp"}' \
  --agent-urls '{"Weather Data": "http://localhost:2024", "Wikipedia": "http://localhost:2026"}' \
  --output benchmark_tasks.jsonl
```

### Step 2: Evaluate Models

```bash
python examples/evaluate_agent_models.py \
  --benchmark benchmark_tasks.jsonl \
  --agent-urls '{"Weather Data": "http://localhost:2024", "Wikipedia": "http://localhost:2026"}' \
  --models gpt-4o gpt-4o-mini \
  --output evaluation_results.jsonl
```

## Benchmark Format

Each record in `benchmark_tasks.jsonl` contains:

| Field | Description |
|-------|-------------|
| `server` | MCP server name |
| `question` | Generated evaluation question |
| `expert_answer` | Gold-standard answer from the frontier model |
| `expert_tools` | Ordered list of tools used by the expert |
| `expert_tool_trace` | Full tool call trace with inputs and outputs |
| `question_quality_rating` | Quality score of the generated question |
| `completeness_rating` | Completeness score of the expert answer |

## Evaluation Metrics

### Programmatic Metrics

| Metric | Description |
|--------|-------------|
| `tool_recall` | Fraction of expert tools also used by the model |
| `tool_precision` | Fraction of model tools that were in the expert set |
| `order_match` | LCS-based match of tool call ordering |
| `param_match` | Similarity of tool call parameters |

### LLM-as-Judge Dimensions (scored 0-10)

| Dimension | Description |
|-----------|-------------|
| `task_fulfillment` | Did the model fully answer the question? |
| `grounding` | Is the answer grounded in tool outputs? |
| `tool_appropriateness` | Were the right tools selected? |
| `parameter_accuracy` | Were tool parameters correct? |
| `dependency_awareness` | Did the model respect tool dependencies? |
| `parallelism_and_efficiency` | Were independent tools called efficiently? |

## What's in examples/

- `generate_agent_benchmark.py` — Generate synthetic benchmark tasks from MCP server tool schemas using SDG Hub's MCP Server Distillation flow
- `evaluate_agent_models.py` — Evaluate multiple models on a generated benchmark using the same agent with LLM-as-judge scoring
