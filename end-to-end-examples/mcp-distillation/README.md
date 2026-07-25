# MCP Distillation End-to-End

**Status:** GA

Teach a small language model to use MCP server tools through synthetic data generation and reinforcement learning. Combines SDG Hub's MCP Distillation flow for tool-use data generation with Training Hub's LoRA GRPO method for efficient fine-tuning.

## Start Here

| Approach | File | Best for |
|----------|------|----------|
| **Interactive notebook** | [`mcp_distillation_e2e.ipynb`](mcp_distillation_e2e.ipynb) | Learning the full pipeline step-by-step, prototyping |
| **CLI scripts** | [`examples/`](examples/) | Production pipelines, automation, CI/CD integration |

The notebook walks through the **complete lifecycle** — from exploring a demo MCP server through data generation, formatting, training, and evaluation — all in one place.

The CLI scripts break the same workflow into composable steps that can be run independently.

## Demo MCP Server

The [`demo_server/`](demo_server/) directory contains a standalone **ShopInsights Analytics Platform** — a FastMCP e-commerce server with 15 tools organized into ambiguity clusters:

```bash
cd demo_server/
pip install fastmcp
python server.py
# Server starts on http://localhost:8008
```

The tools are deliberately designed with overlapping functionality (e.g., `search_products` vs `browse_catalog` vs `get_trending_products`) to test the student model's ability to select the right tool for a given query. See [`demo_server/README.md`](demo_server/README.md) for details.

## Architecture

```
  MCP Server                SDG Hub                          Training Hub
  (your tools)         MCP Distillation Flow                  LoRA GRPO
 +-----------+    +---------------------------+    +------+    +---------+
 |  15 tools |    | Explore -> Synthesize Qs  |    |Format|    | Train   |
 | (FastMCP) |--->| -> Quality Filter         |--->| JSONL|--->| Student |
 |           |    | -> Expert Trajectories    |    |      |    | Model   |
 +-----------+    | -> Response Filter        |    +------+    +---------+
       ^          +---------------------------+                     |
       |                    |                                       v
  Langflow Agent      Teacher LLM                            Tool-capable
  (frontier model     (question gen +                        small model
   + MCP server)       quality scoring)                      (e.g. Qwen3-4B)
```

**Pipeline stages:**

1. **Generate** (01) -- SDG Hub's MCP distillation flow uses a frontier model to explore your MCP server, synthesize realistic questions, and produce expert tool-use trajectories
2. **Format** (02) -- Raw tool traces are converted into the function-calling conversation format (system/user/assistant/function messages) used for SFT
3. **Train** (03) -- Training Hub's LoRA GRPO fine-tunes a small student model on the formatted data

## What's Covered

- Generating tool-use training data with SDG Hub MCP Distillation flow
- Configuring MCP server connections for trace collection
- Formatting Langflow tool traces into function-calling JSONL
- Training with LoRA GRPO via Training Hub
- Scaling data generation with runtime parameter overrides
- Evaluating the fine-tuned model on tool-use tasks

## Prerequisites

- Python 3.10+
- A running MCP server (the example uses a demo e-commerce server with 15 tools)
- Langflow with a frontier model agent (e.g., GPT-5.2) connected to your MCP server
- An API key for the teacher LLM (OpenAI or compatible)
- GPU(s) for GRPO training (step 03)

## Quick Start

### 1. Start the demo MCP server

```bash
cd demo_server/
pip install fastmcp
python server.py
```

### 2. Install dependencies

```bash
cd examples/
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys, Langflow URL, and model preferences
```

### 4. Generate training data

```bash
python 01_generate_tool_data.py --num-samples 10
```

This runs the full MCP distillation pipeline:
- The frontier model (via Langflow) explores your MCP server and calls every tool
- A teacher LLM synthesizes realistic questions grounded in exploration findings
- The frontier model solves each question via actual MCP tool calls
- Quality filters remove low-quality questions and incomplete trajectories
- Output is saved as `generated_data/distillation_output.parquet`

### 5. Format for training

```bash
python 02_format_training_data.py
```

Converts tool traces into structured function-calling conversations:

```json
{"messages": [
  {"role": "system", "content": "<tool declarations>"},
  {"role": "user", "content": "Find trending products and check inventory"},
  {"role": "assistant", "content": "", "function_call": {"name": "get_trending_products", "arguments": "..."}},
  {"role": "function", "content": "{...}", "name": "get_trending_products"},
  {"role": "assistant", "content": "", "function_call": {"name": "get_inventory_status", "arguments": "..."}},
  {"role": "function", "content": "{...}", "name": "get_inventory_status"},
  {"role": "assistant", "content": "The top trending product is..."}
]}
```

Output is saved as `generated_data/training_data.jsonl`.

### 6. Train with GRPO

```bash
python 03_train_grpo.py --backend art --num-iterations 15
```

For multi-GPU training:

```bash
python 03_train_grpo.py --backend verl --n-gpus 4
```

## What's in This Directory

| Path | Description |
|------|-------------|
| `mcp_distillation_e2e.ipynb` | Full end-to-end tutorial notebook (interactive, all steps in one place) |
| `demo_server/` | Standalone demo MCP server (FastMCP, 15 e-commerce tools) |
| `demo_server/server.py` | FastMCP server with 15 tool definitions |
| `demo_server/data.py` | Deterministic data generation for the demo server |
| `examples/` | CLI scripts for production use |
| `examples/01_generate_tool_data.py` | Runs SDG Hub's MCP distillation flow |
| `examples/02_format_training_data.py` | Converts tool traces to function-calling JSONL |
| `examples/03_train_grpo.py` | Trains student model with LoRA GRPO |
| `examples/.env.example` | Template for API keys and configuration |
| `examples/requirements.txt` | Python dependencies |

## CLI Reference

### 01_generate_tool_data.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `./generated_data` | Directory to save generated data |
| `--num-samples` | `10` | Question candidates per MCP server row |
| `--tools-per-question` | `2` | Number of tools each question should require |
| `--checkpoint-dir` | `./checkpoints` | Directory for pipeline checkpoints |

### 02_format_training_data.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-file` | `OUTPUT_DIR/distillation_output.parquet` | Path to distillation output |
| `--output-file` | `OUTPUT_DIR/training_data.jsonl` | Path for formatted JSONL |
| `--output-dir` | `./generated_data` | Base directory for input/output files |

### 03_train_grpo.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | `Qwen/Qwen3-4B` | Student model to fine-tune |
| `--data-path` | `OUTPUT_DIR/training_data.jsonl` | Path to training JSONL |
| `--ckpt-output-dir` | `./checkpoints` | Directory for training checkpoints |
| `--backend` | `art` | `art` (single-GPU) or `verl` (multi-GPU) |
| `--n-gpus` | `1` | Number of GPUs (verl backend only) |
| `--num-iterations` | `15` | GRPO training iterations |
| `--lora-r` | `32` | LoRA rank |
| `--lora-alpha` | `64` | LoRA alpha scaling factor |
| `--group-size` | `8` | GRPO group size for relative scoring |
| `--learning-rate` | `1e-5` | Learning rate |
| `--prompt-batch-size` | `100` | Prompts per batch |

## Scaling Up

Increase volume and question complexity via runtime overrides in step 01:

```bash
# 50 candidates with 3-tool questions (harder)
python 01_generate_tool_data.py --num-samples 50 --tools-per-question 3
```

For multiple MCP servers, modify `build_input_dataset()` in `01_generate_tool_data.py` to return a multi-row DataFrame -- one row per server. The pipeline explores each independently.

## Official Documentation

- [SDG Hub MCP Distillation Examples](https://github.com/red-hat-data-services/sdg_hub/tree/main/examples/agentic/mcp_distillation_training)
