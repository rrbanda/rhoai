# Plot Training Loss

Visualize training loss curves to verify that training converged successfully. This is the first check you should run after any training job completes — a quick loss plot can reveal overfitting, underfitting, or data issues before you spend time on detailed evaluation.

## Quick Start

```python
from training_hub import plot_loss

plot_path = plot_loss("./my-trained-model")
print(f"Plot saved to: {plot_path}")
```

The function reads training logs from the checkpoint directory and saves a loss curve plot. It returns the **path to the saved plot file** (default: `loss_plot.png` in the checkpoint directory).

## Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ckpt_output_dirs` | str or list[str] | required | Checkpoint directory (or list of directories for comparison) |
| `metrics_file` | str | None | Custom metrics file name |
| `output_path` | str | None | Custom output path for the plot |
| `labels` | list[str] | None | Labels for each run (multi-run comparison) |
| `ema` | bool | False | Apply exponential moving average smoothing |
| `ema_span` | int | 30 | EMA span for smoothing |
| `metric_keys` | list[str] | None | Specific metrics to plot |
| `show` | bool | False | Display plot interactively |

## Interpreting Loss Curves

### Healthy Training

A good loss curve should:

- **Decrease steadily** through the first 50-80% of training
- **Plateau** towards the end (learning rate has decayed)
- **Not spike** (spikes indicate data issues or learning rate too high)

### Common Problems

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss doesn't decrease | Learning rate too low | Increase `learning_rate` by 2-5x |
| Loss spikes randomly | Bad data or LR too high | Check data quality, reduce `learning_rate` |
| Loss increases after initial drop | Overfitting | Reduce `num_epochs`, add more data |
| Loss oscillates | Batch size too small | Increase `effective_batch_size` |
| Loss plateaus very early | Learning rate too low | Increase `learning_rate` |
| Loss drops then rises steadily | Catastrophic forgetting | Switch to [OSFT](../training/osft.md) |

## Comparing Runs

Compare multiple training runs to find the best hyperparameters:

```python
from training_hub import plot_loss

plot_loss(
    ["./run-lr2e5", "./run-lr5e5", "./run-lr1e4"],
    labels=["lr=2e-5", "lr=5e-5", "lr=1e-4"],
    output_path="lr_comparison.png",
)
```

### Compare Algorithms

Plot SFT, OSFT, and LoRA runs side by side:

```python
from training_hub import plot_loss

plot_loss(
    ["./sft-output", "./osft-output", "./lora-output"],
    labels=["SFT", "OSFT", "LoRA"],
    output_path="algorithm_comparison.png",
)
```

## With EMA Smoothing

For noisy loss curves (common with small batch sizes), apply exponential moving average smoothing to reveal the underlying trend:

```python
from training_hub import plot_loss

plot_loss("./my-trained-model", ema=True, ema_span=50)
```

| EMA Span | Effect |
|----------|--------|
| 10 | Light smoothing — still shows short-term fluctuations |
| 30 | Default — good balance of detail and trend visibility |
| 50-100 | Heavy smoothing — shows only the long-term trend |

## Command-Line Usage

The `plot_loss` utility is also available as a standalone script:

```bash
python 04-training/utilities/plot_loss.py ./my-trained-model

python 04-training/utilities/plot_loss.py ./run1 ./run2 \
    --labels "lr=2e-5" "lr=5e-5" \
    --ema --ema-span 50 \
    -o ./reports/loss_comparison.png
```

See [`04-training/utilities/plot_loss.py`](https://github.com/rrbanda/rhoai/blob/main/04-training/utilities/plot_loss.py) for the full script.

## Tips and Troubleshooting

!!! tip "Always Plot Before Evaluating"
    A diverging loss curve means the model didn't learn — skip detailed evaluation and fix the training configuration first. This saves hours of wasted evaluation time.

!!! tip "Save Plots with Your Checkpoints"
    Use `output_path` to save plots alongside your model checkpoints. This makes it easy to review training quality weeks later when comparing model versions.

!!! warning "Flat Loss ≠ Bad Training (for LoRA)"
    LoRA training on small datasets can show a nearly flat loss curve while still improving the model. If LoRA loss looks flat, evaluate the model directly before adjusting hyperparameters.

## Related

- [Evaluation Overview](../evaluation/index.md) — Beyond loss: domain-specific evaluation
- [Memory Estimator](memory-estimator.md) — Ensure you have enough VRAM before training
- [Model Interpolation](model-interpolation.md) — Blend models based on loss comparison
- [SFT](../training/sft.md) — Training algorithm reference
- [GPU Requirements](../reference/gpu-requirements.md) — Hardware planning
