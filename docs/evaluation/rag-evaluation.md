# RAG Evaluation

Evaluate retrieval-augmented generation quality by generating a benchmark dataset with known-correct answers, then measuring how well the model answers when given the relevant context. This is essential for validating that your fine-tuned model improves RAG pipeline performance before deploying to production.

## When to Use RAG Evaluation

- You've **fine-tuned a model** for a domain and want to measure knowledge accuracy
- You're **comparing RAG pipelines** (different retrievers, chunk sizes, or prompting strategies)
- You need a **reproducible benchmark** with ground-truth answers from your own documents
- You want to detect **regressions** after retraining or updating your document corpus

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **SDG Hub** | `pip install sdg_hub` |
| **LLM API key** | OpenAI, Anthropic, or any LiteLLM-supported provider |
| **Document corpus** | The same documents used in your RAG pipeline |

## Generate RAG Evaluation Dataset

Use SDG Hub to create question-answer pairs with source context. The flow generates questions from your documents along with ground-truth answers that serve as the benchmark.

```python
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()

corpus = Dataset.from_dict({
    "document": [
        "RHOAI 3.4 supports KServe for model serving with vLLM runtime. "
        "KServe provides autoscaling, canary rollouts, and GPU sharing...",
        "AutoRAG automatically builds and tunes RAG pipelines. It evaluates "
        "different retrieval strategies and selects the best configuration...",
    ],
    "domain": ["rhoai", "rhoai"],
})

rag_flows = FlowRegistry.search_flows(tag="rag-evaluation")

flow = Flow.from_yaml(FlowRegistry.get_flow_path(rag_flows[0]["name"]))
flow.set_model_config(model="gpt-4o-mini")
eval_dataset = flow.generate(corpus)

eval_dataset.to_json("rag_eval.jsonl", orient="records", lines=True)
print(f"Generated {len(eval_dataset)} evaluation examples")
```

!!! tip "Use a Separate Eval Set"
    Generate evaluation data from a **held-out** subset of documents not used for training data generation. This prevents the model from memorizing answers and gives you a true measure of generalization.

## Evaluation Metrics

| Metric | What it measures | Scoring method |
|--------|-----------------|----------------|
| **Answer correctness** | Does the answer match the ground truth? | LLM-as-judge or exact match |
| **Faithfulness** | Is the answer grounded in the retrieved context? | LLM-as-judge (checks for hallucination) |
| **Relevance** | Is the retrieved context relevant to the question? | LLM-as-judge or embedding similarity |
| **Completeness** | Does the answer cover all aspects of the question? | LLM-as-judge with rubric |

## Running Evaluation

### Basic Evaluation Loop

```python
import pandas as pd

eval_df = pd.read_json("rag_eval.jsonl", lines=True)

results = []
for _, row in eval_df.iterrows():
    model_answer = your_rag_pipeline(row["question"])
    results.append({
        "question": row["question"],
        "expected": row["expected_answer"],
        "actual": model_answer,
        "context": row.get("source_context", ""),
    })

results_df = pd.DataFrame(results)
results_df.to_json("rag_results.jsonl", orient="records", lines=True)
```

### LLM-as-Judge Scoring

Use a frontier model to score each answer against the ground truth:

```python
from litellm import completion

def judge_answer(question, expected, actual, context):
    response = completion(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""Score this RAG response on a scale of 1-5 for each criterion.

Question: {question}
Expected Answer: {expected}
Model Answer: {actual}
Retrieved Context: {context}

Score each criterion (1=poor, 5=excellent):
1. Correctness: Does the answer match the expected answer?
2. Faithfulness: Is the answer grounded in the context (no hallucination)?
3. Completeness: Does it address all parts of the question?

Return JSON: {{"correctness": N, "faithfulness": N, "completeness": N}}"""
        }],
    )
    return response.choices[0].message.content

scores = []
for _, row in results_df.iterrows():
    score = judge_answer(
        row["question"], row["expected"], row["actual"], row["context"]
    )
    scores.append(score)
```

## Comparing Models

Evaluate multiple models on the same benchmark to find the best configuration:

=== "Before vs After Fine-Tuning"

    ```python
    import pandas as pd

    eval_df = pd.read_json("rag_eval.jsonl", lines=True)

    models = {
        "base": "meta-llama/Llama-3.1-8B-Instruct",
        "fine-tuned": "./knowledge-model/hf_format/samples_0",
    }

    all_results = {}
    for name, model_path in models.items():
        results = []
        for _, row in eval_df.iterrows():
            answer = run_rag_with_model(row["question"], model_path)
            results.append({"question": row["question"], "answer": answer})
        all_results[name] = pd.DataFrame(results)

    for name, df in all_results.items():
        print(f"{name}: {len(df)} responses generated")
    ```

=== "Different Retrieval Strategies"

    ```python
    strategies = {
        "top-3": {"top_k": 3, "chunk_size": 512},
        "top-5": {"top_k": 5, "chunk_size": 512},
        "top-3-large": {"top_k": 3, "chunk_size": 1024},
    }

    for name, config in strategies.items():
        results = []
        for _, row in eval_df.iterrows():
            answer = run_rag_pipeline(row["question"], **config)
            results.append({"question": row["question"], "answer": answer})
        pd.DataFrame(results).to_json(
            f"results_{name}.jsonl", orient="records", lines=True
        )
    ```

## Interpreting Results

| Score Range | Interpretation | Action |
|-------------|---------------|--------|
| **4.0-5.0** | Excellent — model answers accurately and completely | Ready for production |
| **3.0-3.9** | Good — mostly correct but some gaps | Review failure cases, consider more training data |
| **2.0-2.9** | Fair — significant accuracy issues | Check retrieval quality and training data coverage |
| **1.0-1.9** | Poor — model frequently hallucinates or misses answers | Revisit data generation, try more training epochs |

!!! warning "Watch for Faithfulness"
    A model can score high on correctness but low on faithfulness — meaning it gives the right answer but not from the provided context. This indicates the model memorized answers rather than learning to extract from context, which is a sign of overfitting.

## Tips and Troubleshooting

!!! tip "Scale Your Benchmark"
    Aim for at least 100 evaluation examples across your key document topics. Fewer than 50 examples can produce noisy metrics that don't reflect real-world performance.

!!! tip "Version Your Benchmarks"
    Save both the evaluation dataset and the model checkpoint together. This lets you reproduce results and track improvements across training iterations.

## Related

- [Evaluation Overview](index.md) — All evaluation approaches
- [Knowledge Tuning](../data-generation/knowledge-tuning.md) — Generate training data from the same documents
- [Code Evaluation](code-evaluation.md) — Evaluate code generation
- [Agent Evaluation](agent-evaluation.md) — Evaluate tool-use quality
- [Plot Loss](../utilities/plot-loss.md) — Verify training convergence before evaluating
