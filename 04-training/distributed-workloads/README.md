# Distributed Workloads

**Status:** GA

Distributed workloads in RHOAI 3.4 enable training and data processing across multiple nodes and GPUs for faster time-to-results. Two operators are supported: Kubeflow Training Operator for PyTorch/JAX/DeepSpeed jobs, and KubeRay for Ray-based distributed compute.

## What's Covered

- **Kubeflow Training Operator** -- Distributed PyTorch, JAX, and DeepSpeed training via TrainJob CRDs
- **KubeRay** -- Ray-based distributed compute with RayJob and ManagedClusterConfig
- Multi-node GPU scheduling and resource management
- PVC-based storage for checkpoints and data

## Official Documentation

- [Accelerate data processing and training with distributed workloads](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/accelerate_data_processing_and_training_with_distributed_workloads)

## What's in examples/

Examples cover configuring multi-node training jobs, setting up GPU resource quotas, launching distributed runs with both operators, and monitoring job progress.
