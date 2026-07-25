# RAG Evaluation

Evaluate retrieval-augmented generation quality by generating a benchmark dataset with known-correct answers, then measuring how well the model answers when given the relevant context.

## Generate RAG Evaluation Dataset

Use SDG Hub to create question-answer pairs with source context:

```python
from datasets import Dataset
from sdg_hub import Flow, FlowRegistry

FlowRegistry.discover_flows()

# Your document corpus (same documents used in the RAG pipeline)
corpus = Dataset.from_dict({
    "document": [
        "RHOAI 3.4 supports KServe for model serving with vLLM runtime...",
        "AutoRAG automatically builds and tunes RAG pipelines...",
    ],
    "domain": ["rhoai", "rhoai"],
})

# Search for RAG evaluation flows by tag
rag_flows = FlowRegistry.search_flows(tag="rag-evaluation")

flow = Flow.from_yaml(FlowRegistry.get_flow_path(rag_flows[0]["name"]))
flow.set_model_config(model="gpt-4o-mini")
eval_dataset = flow.generate(corpus)

eval_dataset.to_json("rag_eval.jsonl", orient="records", lines=True)
print(f"Generated {len(eval_dataset)} evaluation examples")
```

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **Answer correctness** | Does the answer match the ground truth? |
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Relevance** | Is the retrieved context relevant to the question? |
| **Completeness** | Does the answer cover all aspects of the question? |

## Running Evaluation

```python
import pandas as pd

eval_df = pd.read_json("rag_eval.jsonl", lines=True)

# For each question, run through your RAG pipeline and compare
results = []
for _, row in eval_df.iterrows():
    model_answer = your_rag_pipeline(row["question"])
    results.append({
        "question": row["question"],
        "expected": row["expected_answer"],
        "actual": model_answer,
    })

results_df = pd.DataFrame(results)
results_df.to_json("rag_results.jsonl", orient="records", lines=True)
```

## Related

- [Evaluation Overview](index.md) — All evaluation approaches
- [Knowledge Tuning](../data-generation/knowledge-tuning.md) — Generate training data from the same documents
- [Code Evaluation](code-evaluation.md) — Evaluate code generation
