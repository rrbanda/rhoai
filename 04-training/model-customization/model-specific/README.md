# Model-Specific Training Scripts

Pre-configured training scripts for popular model families.  Each script
supports both SFT and OSFT via a ``--algorithm`` flag, with hyperparameters
tuned for the target architecture.

## Available Models

| Script | Model | Size | Default GPUs | Default tokens/GPU | Notes |
|--------|-------|------|:------------:|:------------------:|-------|
| `train_qwen.py` | Qwen 2.5 7B Instruct | 7B | 8 | 20,000 (SFT) / 16,384 (OSFT) | Long-context support (16k default) |
| `train_llama.py` | Llama 3.1 8B Instruct | 8B | 8 | 18,000 (SFT) / 16,384 (OSFT) | Supports up to 128k context |
| `train_phi.py` | Phi 4 Mini Instruct | ~3.8B | 8 | 25,000 | Dense, efficient architecture |
| `train_granite.py` | Granite 3.3 8B / 4.0 variants | 8B+ | 2+ | 25,000 (SFT) / 10,000 (OSFT) | Post-training model interpolation |
| `train_gpt_oss.py` | GPT-OSS 20B | 20B | 8 | 12,000 (SFT) / 8,192 (OSFT) | Needs 8x A100 80 GB minimum |

## Quick Start

```bash
# SFT on Qwen 2.5 7B
python train_qwen.py --algorithm sft \
    --data-path ./data.jsonl \
    --ckpt-output-dir ./checkpoints/qwen-sft

# OSFT on Llama 3.1 8B (preserves base capabilities)
python train_llama.py --algorithm osft \
    --data-path ./data.jsonl \
    --ckpt-output-dir ./checkpoints/llama-osft

# SFT on Granite 4.0
python train_granite.py --algorithm sft --granite-version 4.0-h-small \
    --data-path ./data.jsonl \
    --ckpt-output-dir ./checkpoints/granite4-sft
```

## GPU Requirements

| Model | Minimum | Recommended |
|-------|---------|-------------|
| Phi 4 Mini (~3.8B) | 2x A100 40 GB | 8x A100 40 GB |
| Qwen 2.5 7B | 2x A100 40 GB | 8x A100 80 GB |
| Llama 3.1 8B | 2x A100 40 GB | 8x A100 80 GB |
| Granite 3.3 8B | 2x A100 40 GB | 8x A100 80 GB |
| GPT-OSS 20B | 8x A100 80 GB | 8x A100 80 GB |
| Granite 4.0 H-Small | 8x A100 80 GB | 8x A100 80 GB |

## Data Format

All scripts expect JSONL with chat messages:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```
