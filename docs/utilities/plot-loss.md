# Plot Training Loss

Visualize training loss curves to verify that training converged successfully. This is the first check you should run after any training job completes.

## Usage

```python
from training_hub import plot_loss

plot_path = plot_loss("./my-trained-model")
print(f"Plot saved to: {plot_path}")
```

The function reads training logs from the checkpoint directory and saves a loss curve plot. It returns the **path to the saved plot file** (default: `loss_plot.png` in the checkpoint directory).

## Parameters

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

## Comparing Runs

Compare multiple training runs to find the best hyperparameters:

```python
from training_hub import plot_loss

plot_loss(
    ["./run-lr2e5", "./run-lr5e5", "./run-lr1e4"],
    labels=["lr=2e-5", "lr=5e-5", "lr=1e-4"],
    output_path="comparison.png",
)
```

## With EMA Smoothing

For noisy loss curves, apply exponential moving average smoothing:

```python
from training_hub import plot_loss

plot_loss("./my-trained-model", ema=True, ema_span=50)
```

## Related

- [Evaluation Overview](../evaluation/index.md) — Beyond loss: domain-specific evaluation
- [Memory Estimator](memory-estimator.md) — Ensure you have enough VRAM
- [SFT](../training/sft.md) — Training algorithm reference
