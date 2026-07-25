# Knowledge Tuning End-to-End

**Status:** GA

Full pipeline for injecting domain-specific knowledge into a language model. Covers the complete workflow from document processing through evaluation: Synthetic Data Generation (SDG Hub) -> Data Mixing -> Training (Training Hub SFT/OSFT) -> Evaluation.

## What's Covered

- Generating training data with SDG Hub knowledge flows (extractive summary, detailed summary, key facts, document-based QA)
- Converting Q&A pairs into the Training Hub messages format with knowledge-specific `unmask` flag
- Fine-tuning with Training Hub (SFT or OSFT)
- Evaluating the tuned model against the base model

## Prerequisites

- Python 3.10+
- Access to a teacher LLM endpoint (for data generation)
- GPU(s) for training and evaluation
- Pre-processed source documents in JSONL format

## Quick Start

### 1. Setup

```bash
cd examples/
cp .env.example .env
# Edit .env with your model endpoint and API key

pip install -r requirements.txt
```

### 2. Prepare Documents

Place your pre-processed JSONL documents in the `documents/` directory.
Each line should contain at minimum:

```json
{"document": "...", "domain": "...", "document_outline": "...", "icl_document": "...", "icl_query_1": "...", "icl_query_2": "...", "icl_query_3": "..."}
```

The `document_outline` and `icl_*` columns are required by most flow
variants. See the SDG Hub flow metadata for full column requirements.

### 3. Generate Synthetic Data

```bash
python 01_data_generation.py
```

This runs four SDG Hub knowledge flows against your documents, producing
JSONL files in `generated_output_data/`:

| Flow | Description |
|------|-------------|
| `extractive_summary` | Extracts key segments and generates QA from them |
| `detailed_summary` | Creates high-level summaries and derives QA pairs |
| `key_facts` | Decomposes documents into atomic facts with QA pairs |
| `doc_direct_qa` | Generates QA pairs directly from the raw document |

Run a subset of flows with `--flows extractive_summary key_facts`.

### 4. Mix Data for Training

```bash
python 02_data_mixing.py
```

Converts all generated Q&A pairs into the Training Hub `messages` format
and writes `generated_output_data/training_mix/knowledge_train.jsonl`.
Each sample is marked with `"unmask": true` for knowledge training.

Skip tokenization statistics (if the student model isn't downloaded
locally) with `--skip-tokenize`.

### 5. Train the Model

```bash
# OSFT (recommended for knowledge tuning)
python 03_model_training.py --algorithm osft

# Standard SFT
python 03_model_training.py --algorithm sft

# Custom parameters
python 03_model_training.py --algorithm osft \
    --num-epochs 5 \
    --learning-rate 1e-5 \
    --nproc-per-node 4
```

OSFT (Orthogonal Subspace Fine-Tuning) is recommended for knowledge
tuning because it preserves the base model's existing capabilities while
absorbing new domain knowledge.

### 6. Evaluate

```bash
python 04_evaluation.py --tuned-model ./checkpoints/final

# With custom test questions
python 04_evaluation.py \
    --tuned-model ./checkpoints/final \
    --test-questions test_questions.jsonl
```

Compares the base model and fine-tuned model responses side by side on
a set of test questions. Provide a JSONL file with `{"question": "..."}`
per line, or use the built-in fallback questions.

## File Reference

| File | Purpose |
|------|---------|
| `.env.example` | Template for environment variables |
| `requirements.txt` | Python dependencies |
| `01_data_generation.py` | SDG Hub synthetic data generation |
| `02_data_mixing.py` | Convert to Training Hub format |
| `03_model_training.py` | SFT / OSFT fine-tuning |
| `04_evaluation.py` | Base vs tuned model comparison |

## Official Documentation

- [Knowledge Tuning Example](https://github.com/red-hat-data-services/red-hat-ai-examples/tree/main/examples/knowledge-tuning)
