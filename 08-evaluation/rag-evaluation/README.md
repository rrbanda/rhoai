# RAG Evaluation Dataset Generation

Generate high-quality question-answer pairs with ground truth context for evaluating Retrieval-Augmented Generation (RAG) systems. The generated datasets are compatible with evaluation frameworks like [RAGAS](https://docs.ragas.io/).

## Overview

Evaluating a RAG pipeline requires a dataset of questions paired with ground truth answers and the context they were derived from. Manually creating these datasets is expensive and error-prone. This example uses SDG Hub's **RAG Evaluation Dataset Flow** to automatically generate evaluation data from your source documents.

### Pipeline

```
Input Documents (text/PDF)
  → Topic Extraction
  → Conceptual Question Generation
  → Question Evolution (quality improvement)
  → Answer Generation with Grounding
  → Groundedness Scoring & Filtering
  → Ground Truth Context Extraction
  → RAGAS-compatible evaluation dataset
```

Each generated record contains:

| Field | Description |
|-------|-------------|
| `question` | A conceptual question derived from the document |
| `answer` | A grounded answer generated from the source context |
| `contexts` | The source document chunk(s) used to generate the answer |
| `ground_truth` | The extracted ground truth context for evaluation |

## Prerequisites

- Python 3.10+
- SDG Hub installed: `pip install sdg-hub[examples]`
- An LLM endpoint (OpenAI, vLLM, or any LiteLLM-compatible provider)

## Environment Variables

```bash
export INFERENCE_MODEL="openai/gpt-4o"       # Model name (prefix with openai/ for OpenAI-compatible endpoints)
export URL="https://your-endpoint/v1"          # API base URL (optional for native OpenAI)
export API_KEY="your-api-key"                  # API key
export MAX_CONCURRENCY="10"                    # Max parallel LLM calls (default: 10)
```

## Usage

```bash
# Generate from a text file
python examples/generate_rag_eval_dataset.py \
  --input-file /path/to/document.txt \
  --document-title "My Document Summary" \
  --output rag_eval_dataset.jsonl

# Generate from a PDF
python examples/generate_rag_eval_dataset.py \
  --input-file /path/to/report.pdf \
  --document-title "Annual Report 2024" \
  --max-pages 20 \
  --output rag_eval_dataset.jsonl

# Customize chunking
python examples/generate_rag_eval_dataset.py \
  --input-file /path/to/document.txt \
  --document-title "Technical Spec" \
  --chunk-size 4000 \
  --chunk-overlap 500 \
  --output rag_eval_dataset.jsonl
```

## Output Format

The script produces a JSONL file where each line is a JSON object in RAGAS-compatible format:

```json
{
  "question": "What are the key performance indicators mentioned in the report?",
  "answer": "The report highlights revenue growth, customer retention rate, and ...",
  "contexts": ["The company reported a 15% increase in revenue..."],
  "ground_truth": "Key performance indicators include revenue growth of 15%, ..."
}
```

## Using with RAGAS

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import load_dataset

dataset = load_dataset("json", data_files="rag_eval_dataset.jsonl", split="train")
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(results)
```

## What's in examples/

- `generate_rag_eval_dataset.py` — Standalone script to generate a RAG evaluation dataset from text or PDF documents using SDG Hub's RAG Evaluation Dataset Flow
