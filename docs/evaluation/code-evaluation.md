# Code Evaluation

Evaluate code generation models using pass@1 benchmarks. SDG Hub generates coding challenges with test cases, and the evaluation harness checks whether the model produces correct, executable code. Use this to measure whether fine-tuning improved your model's coding ability on domain-specific tasks.

## When to Use Code Evaluation

- You've **fine-tuned a model** for code generation (SQL, Python, Bash, etc.)
- You want to measure **pass@1** — the probability a single generation is correct
- You need **domain-specific benchmarks** beyond generic coding tests (e.g., your internal APIs)
- You're comparing **multiple models or training runs** on the same benchmark

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **SDG Hub** | `pip install sdg_hub` |
| **LLM API key** | For benchmark generation (GPT-4o recommended for high-quality test cases) |
| **Python runtime** | For executing and validating generated code |

!!! warning "Sandbox Execution"
    Always run model-generated code in an isolated environment (container, VM, or restricted subprocess). Never execute untrusted code with full system access.

## Generate Code Benchmark

Create domain-specific coding challenges with test cases:

```python
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()

seed_data = Dataset.from_dict({
    "domain": ["python", "sql", "bash"],
    "difficulty": ["medium", "medium", "easy"],
    "topic": [
        "data manipulation with pandas",
        "aggregation queries with GROUP BY",
        "file system operations and text processing",
    ],
})

code_flows = FlowRegistry.search_flows(tag="code-evaluation")
flow = Flow.from_yaml(FlowRegistry.get_flow_path(code_flows[0]["name"]))
flow.set_model_config(model="gpt-4o")

benchmark = flow.generate(seed_data)
benchmark.to_json("code_benchmark.jsonl", orient="records", lines=True)
print(f"Generated {len(benchmark)} coding challenges")
```

!!! tip "Use GPT-4o for Benchmark Generation"
    Higher-quality teacher models produce more reliable test cases. Using a weaker model for benchmark generation can result in incorrect test cases that penalize correct solutions.

## Benchmark Output Format

Each benchmark example contains a prompt and executable test cases:

```json
{
  "prompt": "Write a function `merge_sorted(a, b)` that merges two sorted lists into one sorted list without using the built-in `sorted()` function.",
  "test_cases": "assert merge_sorted([1,3,5], [2,4,6]) == [1,2,3,4,5,6]\nassert merge_sorted([], [1,2]) == [1,2]\nassert merge_sorted([1], []) == [1]\nassert merge_sorted([], []) == []",
  "domain": "python",
  "difficulty": "medium"
}
```

## Evaluate Models

Run the benchmark against your fine-tuned model and measure pass@1:

```python
import pandas as pd
import subprocess
import tempfile
import os

benchmark = pd.read_json("code_benchmark.jsonl", lines=True)

def evaluate_code(model_code: str, test_cases: str, timeout: int = 30) -> bool:
    """Execute model code with test cases and return pass/fail."""
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False
    ) as f:
        f.write(model_code)
        f.write("\n\n")
        f.write(test_cases)
        temp_path = f.name

    try:
        subprocess.run(
            ["python", temp_path],
            timeout=timeout,
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    finally:
        os.unlink(temp_path)

results = []
for _, row in benchmark.iterrows():
    model_code = generate_with_model(row["prompt"])
    passed = evaluate_code(model_code, row["test_cases"])
    results.append({
        "prompt": row["prompt"],
        "domain": row["domain"],
        "difficulty": row["difficulty"],
        "passed": passed,
    })

df = pd.DataFrame(results)
pass_at_1 = df["passed"].mean()
print(f"Overall pass@1: {pass_at_1:.2%}")
```

## Key Metric: pass@1

**pass@1** measures the probability that a single model generation passes all test cases. Higher is better.

| Model | Typical pass@1 |
|-------|---------------|
| Base 7B model | 30-40% |
| Fine-tuned 7B (domain-specific) | 50-65% |
| Fine-tuned 7B (general code) | 45-55% |
| Frontier model (GPT-4o, Claude) | 80-90% |

## Breaking Down Results

Analyze pass@1 by domain and difficulty for actionable insights:

```python
for domain in df["domain"].unique():
    subset = df[df["domain"] == domain]
    print(f"{domain}: {subset['passed'].mean():.2%} "
          f"({subset['passed'].sum()}/{len(subset)})")

for difficulty in ["easy", "medium", "hard"]:
    subset = df[df["difficulty"] == difficulty]
    if len(subset) > 0:
        print(f"{difficulty}: {subset['passed'].mean():.2%}")
```

## Comparing Models

Evaluate a base model and fine-tuned model on the same benchmark:

```python
models = {
    "base": "meta-llama/Llama-3.1-8B-Instruct",
    "fine-tuned": "./code-model/hf_format/samples_0",
}

comparison = {}
for name, model_path in models.items():
    results = []
    for _, row in benchmark.iterrows():
        code = generate_with_model(row["prompt"], model_path=model_path)
        passed = evaluate_code(code, row["test_cases"])
        results.append({"passed": passed, "domain": row["domain"]})

    model_df = pd.DataFrame(results)
    comparison[name] = model_df["passed"].mean()
    print(f"{name}: pass@1 = {comparison[name]:.2%}")

improvement = comparison["fine-tuned"] - comparison["base"]
print(f"\nImprovement: {improvement:+.2%}")
```

## Tips and Troubleshooting

!!! tip "Increase Benchmark Size for Reliable Metrics"
    With fewer than 50 problems, pass@1 variance is high. Aim for 100+ problems per domain for stable measurements.

!!! tip "Test Multiple Temperatures"
    Run evaluation at temperature 0.0 (deterministic) for reproducibility. Then test at 0.2-0.4 to measure the model's ability to generate diverse correct solutions.

!!! warning "Watch for Test Case Quality"
    If your fine-tuned model fails many tests that seem correct by inspection, verify the test cases themselves. Ambiguous prompts or incorrect assertions can produce false negatives.

## Related

- [Evaluation Overview](index.md) — All evaluation approaches
- [Agent Evaluation](agent-evaluation.md) — Evaluate tool-use instead
- [RAG Evaluation](rag-evaluation.md) — Evaluate RAG quality
- [SFT](../training/sft.md) — Training for code generation
- [LoRA](../training/lora.md) — Memory-efficient fine-tuning for code models
