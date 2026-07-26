# Agent Evaluation

Evaluate tool-use models using LLM-as-judge scoring. An evaluator model assesses whether the agent correctly selects tools, constructs arguments, and uses results to answer user queries. Use this after [GRPO training](../training/grpo.md) or [MCP distillation](../end-to-end/mcp-distillation.md) to measure tool-use quality before deployment.

## When to Use Agent Evaluation

- You've trained a model with [GRPO](../training/grpo.md) for tool use
- You've completed an [MCP distillation](../end-to-end/mcp-distillation.md) pipeline
- You need to **compare tool-use accuracy** across model versions
- You want to validate **multi-step reasoning** (chaining multiple tool calls)

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **SDG Hub** | `pip install sdg-hub` |
| **Judge model API key** | GPT-4o or Claude recommended for reliable scoring |
| **MCP server** | The tools your model was trained to use |

## Generate Agent Benchmark

Create evaluation scenarios for your MCP tools:

```python
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()

seed_data = Dataset.from_dict({
    "tool_descriptions": [
        "search_products(query, max_price) - Search products by name or category",
        "get_order(order_id) - Get order details and status",
        "create_return(order_id, reason) - Initiate a product return",
    ],
    "complexity": ["single-tool", "multi-tool", "multi-tool"],
})

agent_flows = FlowRegistry.search_flows(tag="agent-evaluation")
flow = Flow.from_yaml(FlowRegistry.get_flow_path(agent_flows[0]["name"]))
flow.set_model_config(model="gpt-4o")

benchmark = flow.generate(seed_data)
benchmark.to_json("agent_benchmark.jsonl", orient="records", lines=True)
print(f"Generated {len(benchmark)} evaluation scenarios")
```

!!! tip "Cover All Complexity Levels"
    Include single-tool, multi-tool, and edge-case scenarios in your benchmark. Models often succeed on simple calls but fail when chaining tools or handling errors.

## Evaluation Dimensions

| Dimension | What it measures | Weight | Score range |
|-----------|-----------------|--------|-------------|
| **Tool selection** | Correct tool chosen for the task | High | 0-5 |
| **Argument quality** | Parameters are well-formed and appropriate | High | 0-5 |
| **Multi-step reasoning** | Can chain multiple tool calls logically | Medium | 0-5 |
| **Response synthesis** | Uses tool results to answer the user | Medium | 0-5 |
| **Error handling** | Gracefully handles tool failures or unexpected results | Low | 0-5 |

## Evaluate with LLM-as-Judge

Use a frontier model to judge tool-use quality across all dimensions:

```python
import pandas as pd
import json
from litellm import completion

benchmark = pd.read_json("agent_benchmark.jsonl", lines=True)

def judge_tool_use(prompt, tools, expected, model_response):
    """Score a model's tool-use response using LLM-as-judge."""
    judge_response = completion(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Evaluate this agent's tool-use response.

User Query: {prompt}
Available Tools: {json.dumps(tools)}
Expected Behavior: {expected}
Model Response: {json.dumps(model_response)}

Score each dimension (0=completely wrong, 5=perfect):
1. Tool selection: Did it call the right tool(s)?
2. Argument quality: Were arguments correct and well-formed?
3. Multi-step reasoning: Did it chain calls logically (if needed)?
4. Response synthesis: Did it use tool results to answer the user?
5. Error handling: Did it handle unexpected results gracefully?

Return JSON: {{"tool_selection": N, "argument_quality": N,
"multi_step": N, "response_synthesis": N, "error_handling": N,
"total": N, "reasoning": "brief explanation"}}"""
        }],
    )
    return json.loads(judge_response.choices[0].message.content)

scores = []
for _, row in benchmark.iterrows():
    model_response = run_agent(row["prompt"], available_tools=row["tools"])
    score = judge_tool_use(
        row["prompt"], row["tools"], row["expected"], model_response
    )
    scores.append(score)

scores_df = pd.DataFrame(scores)
print(f"Average total score: {scores_df['total'].mean():.1f}/25")
for dim in ["tool_selection", "argument_quality", "multi_step",
            "response_synthesis", "error_handling"]:
    print(f"  {dim}: {scores_df[dim].mean():.1f}/5")
```

## Comparing Models

Evaluate before and after GRPO training to measure improvement:

=== "Before vs After Training"

    ```python
    models = {
        "base": "meta-llama/Llama-3.1-8B-Instruct",
        "grpo-trained": "./tool-use-model/hf_format/samples_0",
    }

    comparison = {}
    for name, model_path in models.items():
        model_scores = []
        for _, row in benchmark.iterrows():
            response = run_agent(
                row["prompt"], available_tools=row["tools"],
                model_path=model_path,
            )
            score = judge_tool_use(
                row["prompt"], row["tools"], row["expected"], response
            )
            model_scores.append(score["total"])

        comparison[name] = sum(model_scores) / len(model_scores)
        print(f"{name}: {comparison[name]:.1f}/25")

    improvement = comparison["grpo-trained"] - comparison["base"]
    print(f"\nImprovement: {improvement:+.1f} points")
    ```

=== "Across Iterations"

    ```python
    iterations = [5, 10, 15, 20]
    for n_iter in iterations:
        model_path = f"./grpo-iter{n_iter}/hf_format/samples_0"
        model_scores = []
        for _, row in benchmark.iterrows():
            response = run_agent(
                row["prompt"], available_tools=row["tools"],
                model_path=model_path,
            )
            score = judge_tool_use(
                row["prompt"], row["tools"], row["expected"], response
            )
            model_scores.append(score["total"])

        avg = sum(model_scores) / len(model_scores)
        print(f"Iteration {n_iter}: {avg:.1f}/25")
    ```

## Interpreting Results

| Total Score | Interpretation | Action |
|-------------|---------------|--------|
| **20-25** | Excellent — model reliably selects and uses tools | Ready for production |
| **15-19** | Good — works for most cases, struggles with complex chains | More training iterations or data |
| **10-14** | Fair — basic tool calls work but multi-step fails | Review training data quality and diversity |
| **0-9** | Poor — model doesn't understand tool-use patterns | Check data format, increase GRPO iterations |

## Tips and Troubleshooting

!!! tip "Test with Real MCP Servers"
    Run evaluation against the actual MCP servers your model will use in production. Mock servers may not surface argument formatting issues that real APIs expose.

!!! tip "Separate Simple and Complex Scenarios"
    Report pass rates separately for single-tool and multi-tool scenarios. A model can score 90% on single-tool calls but only 40% on multi-step chains — the aggregate masks this gap.

!!! warning "Judge Model Consistency"
    LLM-as-judge scoring can vary between runs. Use temperature 0 for the judge model and run the evaluation 2-3 times to ensure stable metrics.

## Related

- [Evaluation Overview](index.md) — All evaluation approaches
- [MCP Distillation](../end-to-end/mcp-distillation.md) — Train tool-use models
- [GRPO](../training/grpo.md) — The RL algorithm for tool-use training
- [RAG Evaluation](rag-evaluation.md) — Evaluate retrieval quality instead
- [Code Evaluation](code-evaluation.md) — Evaluate code generation
