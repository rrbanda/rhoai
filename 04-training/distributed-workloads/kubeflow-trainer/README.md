# Kubeflow Training Operator

**Status:** GA

The Kubeflow Training Operator manages distributed training jobs on OpenShift. It supports PyTorch, JAX, and DeepSpeed frameworks through the TrainJob custom resource, handling multi-node GPU scheduling, fault tolerance, and job lifecycle management.

## What's Covered

- TrainJob CRD for declarative distributed training configuration
- PyTorch DistributedDataParallel (DDP) and FSDP strategies
- JAX multi-host training
- DeepSpeed ZeRO stages for large model training
- Multi-node GPU scheduling and placement

## Key Concepts

- **TrainJob** -- Custom resource defining the training workload, framework, and resource requirements
- **Worker replicas** -- Number of distributed training processes across nodes
- **Gang scheduling** -- Ensures all workers are co-scheduled before training begins

## Official Documentation

- [Accelerate data processing and training with distributed workloads](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/accelerate_data_processing_and_training_with_distributed_workloads)

## What's in examples/

Examples show TrainJob YAML manifests for PyTorch DDP, DeepSpeed ZeRO-3 configurations, and multi-node GPU scheduling across heterogeneous clusters.
