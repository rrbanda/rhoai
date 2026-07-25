# Code Evaluation

Evaluate code generation models using pass@1 benchmarks. SDG Hub generates coding challenges, and the evaluation harness checks whether the model produces correct, executable code.

## Generate Code Benchmark

```python
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()

# Define coding domains to evaluate
seed_data = Dataset.from_dict({
    "domain": ["python", "sql", "bash"],
    "difficulty": ["medium", "medium", "easy"],
    "topic": [
        "data manipulation with pandas",
        "aggregation queries",
        "file system operations",
    ],
})

code_flows = FlowRegistry.search_flows(tag="code-evaluation")
flow = Flow.from_yaml(FlowRegistry.get_flow_path(code_flows[0]["name"]))
flow.set_model_config(model="gpt-4o")

benchmark = flow.generate(seed_data)
benchmark.to_json("code_benchmark.jsonl", orient="records", lines=True)
```

## Evaluate Models

Run the benchmark against your fine-tuned model and measure pass@1:

```python
import pandas as pd
import subprocess
import tempfile

benchmark = pd.read_json("code_benchmark.jsonl", lines=True)

results = []
for _, row in benchmark.iterrows():
    model_code = generate_with_model(row["prompt"])

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(model_code)
        f.write("\n")
        f.write(row["test_cases"])

    try:
        subprocess.run(["python", f.name], timeout=30, check=True,
                       capture_output=True)
        passed = True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        passed = False

    results.append({"prompt": row["prompt"], "passed": passed})

df = pd.DataFrame(results)
pass_at_1 = df["passed"].mean()
print(f"pass@1: {pass_at_1:.2%}")
```

## Key Metric: pass@1

**pass@1** measures the probability that a single model generation passes all test cases. Higher is better.

| Model | Typical pass@1 |
|-------|---------------|
| Base 7B model | 30-40% |
| Fine-tuned 7B | 50-65% |
| Frontier model | 80-90% |

## Related

- [Evaluation Overview](index.md) — All evaluation approaches
- [Agent Evaluation](agent-evaluation.md) — Evaluate tool-use instead
- [RAG Evaluation](rag-evaluation.md) — Evaluate RAG quality
