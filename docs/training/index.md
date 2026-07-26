# Training Algorithms

Training Hub provides multiple fine-tuning algorithms, each optimized for different constraints. Pick the one that matches your use case, GPU budget, and data type.

Not sure which to pick? Use the [decision flowchart](../getting-started/choosing-an-algorithm.md).

## Algorithm Overview

| Algorithm | Parameters Trained | Min GPU | Best For | Output |
|-----------|-------------------|---------|----------|--------|
| [SFT](sft.md) | All (100%) | 2x A100 80GB | Maximum learning from abundant data | Full model |
| [OSFT](osft.md) | All (constrained) | 2x A100 80GB | Adding knowledge without forgetting | Full model |
| [LoRA](lora.md) | ~1% (adapters) | 1x L4 24GB (QLoRA) | Single-GPU, tool-calling agents, multi-adapter serving | Adapter |
| [GRPO](grpo.md) | ~1% (LoRA) | 1-4x A100 | Reward-based tool-use learning | Adapter |

## Which Algorithm for Which Track?

=== "Knowledge Track"

    Teaching a model domain knowledge (financial regulations, medical literature, product docs):

    - **OSFT** (recommended) — Adds knowledge while preserving general capabilities
    - **SFT** — Maximum learning capacity when you have abundant data and don't need base knowledge retention
    - **LoRA** — Memory-efficient option when GPU resources are limited

=== "Tool-Calling Track"

    Fine-tuning a model for tool-calling (MCP servers, APIs):

    - **LoRA SFT** (recommended, [validated on RHOAI](../end-to-end/financial-agent.md)) — Train on expert demonstrations from MCP distillation
    - **GRPO** — Learn from rewards when expert traces are unavailable

## After Training: Next Steps

All training algorithms produce model artifacts in your `ckpt_output_dir`. To deploy your model on RHOAI, you need to make those artifacts accessible to KServe:

=== "Upload to S3"

    ```bash
    # Full model (SFT, OSFT) or merged LoRA
    aws s3 sync ./my-model s3://my-bucket/models/my-model/

    # Then use storageUri: s3://my-bucket/models/my-model in your InferenceService
    ```

=== "Copy to PVC"

    ```bash
    # Create a PVC in your namespace, then copy via a helper pod
    oc cp ./my-model $(oc get pod -l app=model-copy -o name):/mnt/models/my-model

    # Then use storageUri: pvc://model-storage/my-model in your InferenceService
    ```

=== "LoRA adapter (no merge)"

    If you trained with LoRA, vLLM can serve the adapter directly without merging. Mount the adapter PVC alongside the base model — see the [Tool-Calling Model Pipeline Step 4](../end-to-end/financial-agent.md#step-4-deploy-the-fine-tuned-model-on-rhoai) for a worked example.

See the [Serving Guide](../serving/index.md) for full KServe + vLLM deployment instructions and YAML manifests.

## Advanced Algorithms

| Algorithm | Use Case | Guide |
|-----------|----------|-------|
| [LAB Multi-Phase](lab-multiphase.md) | InstructLab's phased training pipeline | [Guide](lab-multiphase.md) |
| [Continual Learning](continual-learning.md) | Incrementally add knowledge without retraining | [Guide](continual-learning.md) |
| [Continued Pretraining](continued-pretraining.md) | Extend base model with large-scale domain text | [Guide](continued-pretraining.md) |
