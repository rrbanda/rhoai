# Continued Pretraining

Inject new domain knowledge into an already-trained instruct model by running
additional pretraining on domain-specific documents.

## Overview

These examples use the
[SpreadsheetBench](https://huggingface.co/datasets/KAKA22/SpreadsheetBench)
dataset to teach a Granite 3.3 8B Instruct model to understand Excel
spreadsheet data.  The scripts handle the full pipeline: downloading the
dataset, converting `.xlsx` files to markdown text, and training.

| Algorithm | Script | Key benefit |
|-----------|--------|-------------|
| **OSFT** | `examples/cpt_spreadsheet_osft.py` | Preserves original capabilities via orthogonal subspace constraints |
| **SFT** | `examples/cpt_spreadsheet_sft.py` | Standard pretraining with full parameter updates |

Both scripts use `is_pretraining=True` with `{"document": "..."}` JSONL
entries -- a distinct mode from instruction-tuning which uses `{"messages":
[...]}`.

## Prerequisites

```bash
pip install 'markitdown[xlsx]' huggingface_hub
```

## Quick Start

```bash
# OSFT continued pretraining (preserves base capabilities)
python examples/cpt_spreadsheet_osft.py \
    --ckpt-output-dir ./checkpoints/cpt-osft

# SFT continued pretraining
python examples/cpt_spreadsheet_sft.py \
    --ckpt-output-dir ./checkpoints/cpt-sft

# Prepare data only (no training)
python examples/cpt_spreadsheet_osft.py --prepare-only

# Use pre-prepared data
python examples/cpt_spreadsheet_osft.py \
    --data-path ./spreadsheet_pretraining_data.jsonl \
    --ckpt-output-dir ./checkpoints/cpt-osft
```

## OSFT vs SFT for Continued Pretraining

**OSFT** constrains weight updates to an orthogonal subspace, so the model
absorbs spreadsheet knowledge without losing its instruction-following and
reasoning abilities.  Use a slightly higher learning rate (5e-6 vs 2e-6)
since the orthogonal constraint prevents destructive interference.

**SFT** applies standard pretraining with full parameter updates.  Simpler but
risks degrading the model's existing capabilities, especially with small
or narrow datasets.  Uses a lower learning rate (2e-6) and higher token
throughput (25k tokens/GPU vs 10k) since there is no OSFT decomposition
overhead.

## Hardware Requirements

Both scripts default to auto-detecting available GPUs.  At least 2x A100
40 GB GPUs are recommended for Granite 3.3 8B.
