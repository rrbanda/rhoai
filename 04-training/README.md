# Training and Model Customization

**Status:** GA (components vary; see subfolders)

Training in RHOAI 3.4 covers the full spectrum of model customization, from supervised fine-tuning to reinforcement learning, along with infrastructure for distributing workloads across multiple nodes and GPUs. AutoML capabilities are available as a Developer Preview.

## What's Covered

- **Model Customization** -- Training Hub provides a unified API for LLM post-training algorithms: SFT, OSFT, LoRA, and GRPO
- **Distributed Workloads** -- Scale training across nodes using Kubeflow Training Operator or KubeRay
- **AutoML** -- Automated model selection and hyperparameter tuning (Developer Preview)

## Directory Structure

| Folder | Description |
|--------|-------------|
| `model-customization/` | Training Hub algorithms (SFT, OSFT, LoRA, GRPO) |
| `distributed-workloads/` | Kubeflow Trainer and KubeRay operators |
| `automl/` | AutoML for automated tuning |

## Official Documentation

- [Customize models for Gen AI and Agentic AI](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/customize_models_for_gen_ai_and_agentic_ai_applications)
- [Distributed workloads](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/accelerate_data_processing_and_training_with_distributed_workloads)
- [AutoML](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/working_with_automl)

## What's in examples/

Examples in this section demonstrate end-to-end training workflows: preparing datasets, configuring training jobs, launching distributed runs, and evaluating results.
