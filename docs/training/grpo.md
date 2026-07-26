# Group Relative Policy Optimization (GRPO)

GRPO is a reinforcement learning algorithm that trains models to use tools effectively. Instead of learning from labeled examples (like SFT), GRPO learns from **verifiable rewards** — did the tool call succeed? Did the response match the expected output?

## When to Use GRPO

- You're training a model to **call tools** (MCP servers, REST APIs, function calls)
- You have **tool-use traces** with verifiable success/failure signals
- You want the model to learn **which tool to use** and **how to construct arguments**
- You're building a **tool-calling model** that interacts with external systems

!!! note "Install the GRPO extra"
    GRPO requires additional dependencies not included in the base package:
    ```bash
    pip install training-hub[grpo,lora]
    ```

!!! warning "Not available in RHOAI 3.4.2 ClusterTrainingRuntime"
    The `training-hub` package pre-installed in the RHOAI 3.4.2 ClusterTrainingRuntime does **not** include `lora_grpo`. To use GRPO on-cluster, build a custom training image with `training-hub[grpo,lora]` installed. For on-cluster training via TrainJob, use **LoRA SFT** (`lora_sft`) instead — see the [MCP Distillation Pipeline](../end-to-end/mcp-distillation.md).

## Quick Start

```python
from training_hub import lora_grpo

lora_grpo(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    data_path="tool_traces.jsonl",
    ckpt_output_dir="./grpo-output",
    num_iterations=15,
    backend="art",
)
```

## How GRPO Works

```mermaid
graph LR
    A[Model generates<br/>N tool-call attempts] --> B[Execute against<br/>real tools]
    B --> C[Score: success/failure<br/>+ output quality]
    C --> D[Compute relative<br/>advantage within group]
    D --> E[Update model to favor<br/>higher-scoring attempts]
    E --> A
```

1. For each prompt, the model generates **N candidate responses** (tool calls)
2. Each response is **executed** against the actual tool/API
3. Responses are **scored** based on whether they succeeded
4. The model is **updated** to favor responses that scored higher relative to the group

This is more data-efficient than SFT because the model learns *why* certain tool calls work, not just to imitate examples.

## Data Format

GRPO expects tool-use traces in messages format with tool calls and results:

```json
{
  "messages": [
    {"role": "user", "content": "Find products under $20"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "type": "function",
        "function": {
          "name": "search_products",
          "arguments": "{\"max_price\": 20}"
        }
      }]
    },
    {
      "role": "tool",
      "content": "[{\"name\": \"Widget\", \"price\": 15.99}]"
    },
    {"role": "assistant", "content": "I found Widget at $15.99."}
  ]
}
```

## Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | str | required | HuggingFace model ID or local path |
| `ckpt_output_dir` | str | required | Where to save the trained adapter |
| `data_path` | str | None | Path to JSONL tool traces |
| `num_iterations` | int | 15 | Number of GRPO iterations |
| `group_size` | int | 8 | Number of candidate responses per prompt |
| `prompt_batch_size` | int | 100 | Number of prompts per iteration |
| `learning_rate` | float | 1e-5 | Learning rate |
| `lora_r` | int | 16 | LoRA adapter rank |
| `lora_alpha` | int | 8 | LoRA scaling factor |
| `temperature` | float | 0.7 | Sampling temperature for candidate generation |
| `max_tokens` | int | 512 | Max tokens per generated response |
| `max_prompt_length` | int | 16384 | Max prompt length |
| `n_gpus` | int | 1 | Number of GPUs |
| `gpu_memory_utilization` | float | 0.45 | vLLM GPU memory fraction |

## GPU Requirements

| Model Size | Min GPUs |
|-----------|----------|
| 3B | 1x A100 80GB |
| 7-8B | 2x A100 80GB |
| 20B | 4x A100 80GB |

!!! note "Training Time"
    GRPO is slower than SFT because it generates multiple candidate responses per example and needs to execute tool calls for scoring. Plan for 2-4x the wall-clock time of SFT on the same data.

## Related

- [MCP Distillation Pipeline](../end-to-end/mcp-distillation.md) — Generate tool-use training data, then train with GRPO
- [LoRA](lora.md) — GRPO uses LoRA adapters internally
- [Tool-Use Evaluation](../evaluation/agent-evaluation.md) — Evaluate tool-calling model quality
