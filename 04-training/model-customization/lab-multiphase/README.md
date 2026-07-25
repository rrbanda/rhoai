# LAB Multi-Phase Training

**Status:** GA

LAB (Large-scale Alignment for chatBots) multi-phase training is a two-phase fine-tuning strategy designed for instruction-tuned models. It first establishes knowledge foundations, then adds task-specific skills while preserving all prior capabilities through comprehensive replay.

## What's Covered

- Two-phase training pipeline: knowledge tuning followed by skills + replay
- Comprehensive replay strategy to prevent catastrophic forgetting
- Automatic checkpoint discovery between phases
- Single-node and multi-node distributed training

## Training Phases

### Phase 1 — Knowledge Tuning (Phase07)

Trains the base model on knowledge-heavy data (facts, domain knowledge, core concepts). Uses a smaller batch size appropriate for focused knowledge datasets.

### Phase 2 — Skills + Replay (Phase10)

Continues from the Phase07 checkpoint on a combined dataset containing:

- **New skills data** — task instructions, problem-solving examples
- **Phase07 knowledge replay** — prevents forgetting of newly acquired knowledge
- **Base model instruction replay** — preserves original instruction-following capabilities

Uses a larger batch size to match the larger combined dataset.

## Quick Start

```python
from training_hub import sft

# Phase07: Knowledge tuning
sft(model_path="path/to/base-model", data_path="knowledge.jsonl",
    ckpt_output_dir="./ckpt/phase07", num_epochs=7,
    effective_batch_size=128, learning_rate=2e-5, ...)

# Phase10: Skills + replay (continues from Phase07 checkpoint)
sft(model_path="./ckpt/phase07/hf_format/<latest>",
    data_path="skills_plus_replay.jsonl",
    ckpt_output_dir="./ckpt/phase10", num_epochs=7,
    effective_batch_size=3840, learning_rate=2e-5, ...)
```

## When to Use LAB Multi-Phase

- You need to inject domain knowledge *and* teach new skills
- You want to preserve the base model's instruction-following ability
- You have separate knowledge and skills datasets
- You need a reproducible, multi-stage training pipeline

## What's in examples/

| File | Description |
|------|-------------|
| `lab_multiphase_training.py` | End-to-end CLI script for both phases with checkpoint discovery |
| `lab_multiphase_tutorial.ipynb` | Interactive notebook walking through configuration and execution |
