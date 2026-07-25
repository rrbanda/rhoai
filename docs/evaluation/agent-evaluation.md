# Agent Evaluation

Evaluate tool-use models using LLM-as-judge scoring. An evaluator model assesses whether the agent correctly selects tools, constructs arguments, and uses results to answer user queries.

## Generate Agent Benchmark

Create evaluation scenarios for your MCP tools:

```python
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()

seed_data = Dataset.from_dict({
    "tool_descriptions": [
        "search_products(query, max_price) - Search products by name",
        "get_order(order_id) - Get order status",
    ],
    "complexity": ["single-tool", "multi-tool"],
})

agent_flows = FlowRegistry.search_flows(tag="agent-evaluation")
flow = Flow.from_yaml(FlowRegistry.get_flow_path(agent_flows[0]["name"]))
flow.set_model_config(model="gpt-4o")

benchmark = flow.generate(seed_data)
benchmark.to_json("agent_benchmark.jsonl", orient="records", lines=True)
```

## Evaluate with LLM-as-Judge

Use a frontier model to judge tool-use quality:

```python
import pandas as pd

benchmark = pd.read_json("agent_benchmark.jsonl", lines=True)

scores = []
for _, row in benchmark.iterrows():
    # Get model's response (including tool calls)
    model_response = run_agent(row["prompt"], available_tools=row["tools"])

    # Judge with a frontier model
    judge_prompt = f"""
    User Query: {row['prompt']}
    Available Tools: {row['tools']}
    Expected Behavior: {row['expected']}
    Model Response: {model_response}

    Score the response on:
    1. Tool selection (0-5): Did it call the right tool?
    2. Argument quality (0-5): Were arguments correct?
    3. Response quality (0-5): Did it use tool results well?
    """

    score = judge_model(judge_prompt)
    scores.append(score)

avg_score = sum(s["total"] for s in scores) / len(scores)
print(f"Average agent score: {avg_score:.1f}/15")
```

## Evaluation Dimensions

| Dimension | What it measures | Weight |
|-----------|-----------------|--------|
| **Tool selection** | Correct tool chosen for the task | High |
| **Argument quality** | Parameters are well-formed and appropriate | High |
| **Multi-step reasoning** | Can chain multiple tool calls logically | Medium |
| **Response synthesis** | Uses tool results to answer the user | Medium |
| **Error handling** | Gracefully handles tool failures | Low |

## Related

- [Evaluation Overview](index.md) — All evaluation approaches
- [MCP Distillation](../end-to-end/mcp-distillation.md) — Train tool-use models
- [GRPO](../training/grpo.md) — The RL algorithm for tool-use training
