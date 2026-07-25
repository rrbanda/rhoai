# Plot Training Loss

Visualize training loss curves to verify that training converged successfully. This is the first check you should run after any training job completes.

## Usage

```python
from training_hub import plot_loss

plot_loss("./my-trained-model")
```

This reads the training logs from the output directory and generates a loss curve plot.

## Interpreting Loss Curves

### Healthy Training

A good loss curve should:

- **Decrease steadily** through the first 50-80% of training
- **Plateau** towards the end (learning rate has decayed)
- **Not spike** (spikes indicate data issues or learning rate too high)

### Common Problems

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss doesn't decrease | Learning rate too low | Increase `lr` by 2-5x |
| Loss spikes randomly | Bad data or LR too high | Check data quality, reduce `lr` |
| Loss increases after initial drop | Overfitting | Reduce `num_epochs`, add more data |
| Loss oscillates | Batch size too small | Increase `batch_size` |
| Loss plateaus very early | Learning rate too low | Increase `lr` |

## Comparing Runs

Compare multiple training runs to find the best hyperparameters:

```python
from training_hub import plot_loss

# Plot multiple runs on the same chart
plot_loss(
    "./run-lr2e5",
    "./run-lr5e5",
    "./run-lr1e4",
    labels=["lr=2e-5", "lr=5e-5", "lr=1e-4"],
)
```

## Related

- [Evaluation Overview](../evaluation/index.md) — Beyond loss: domain-specific evaluation
- [Memory Estimator](memory-estimator.md) — Ensure you have enough VRAM
- [SFT](../training/sft.md) — Training algorithm reference
