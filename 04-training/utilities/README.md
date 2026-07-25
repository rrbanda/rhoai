# Training Utilities

Helper scripts for tasks that complement the core training algorithms (SFT, OSFT, LoRA, GRPO).

## Available Utilities

| Script | Purpose |
|--------|---------|
| `memory_estimator.py` | Estimate per-GPU VRAM usage before launching a training job |
| `plot_loss.py` | Visualize training loss curves from checkpoint directories |
| `model_interpolation.py` | Merge a base model and fine-tuned checkpoint via linear interpolation |

## Memory Estimator

Runs `training_hub.estimate()` to predict memory consumption for SFT, OSFT, LoRA, and QLoRA configurations without starting a training job.

```bash
python memory_estimator.py --model Qwen/Qwen2.5-7B-Instruct --method sft
python memory_estimator.py --model Qwen/Qwen2.5-7B-Instruct --method osft --unfreeze-rank-ratio 0.25
python memory_estimator.py --model Qwen/Qwen2.5-7B-Instruct --method lora --lora-r 16
```

## Plot Loss

Reads metrics files written during training and generates loss-curve plots. Supports comparing multiple experiments and EMA smoothing.

```bash
python plot_loss.py /path/to/checkpoints
python plot_loss.py ./run1 ./run2 --labels "lr=1e-5" "lr=5e-6" --ema
```

## Model Interpolation

Creates a new checkpoint whose weights are a linear blend of a base model and a fine-tuned model. Useful for balancing domain-specific improvements against general capability retention.

```bash
python model_interpolation.py \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --trained-model-path ./checkpoints/sft/epoch_3 \
    --trained-model-weight 0.7
```
