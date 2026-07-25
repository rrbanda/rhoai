# KubeRay Operator

**Status:** GA

KubeRay manages Ray clusters on OpenShift for distributed compute workloads. It uses RayJob resources with ManagedClusterConfig for automatic cluster provisioning and PVC-based storage for data and checkpoints. KubeRay supports both data-parallel and model-parallel training strategies.

## What's Covered

- RayJob custom resource for submitting distributed workloads
- ManagedClusterConfig for automatic Ray cluster lifecycle management
- PVC-based storage integration for datasets and checkpoints
- Data-parallel training (Ray Train) and model-parallel inference (Ray Serve)
- Autoscaling worker nodes based on workload demand

## Key Concepts

- **RayJob** -- Submits a Ray application to a managed or existing Ray cluster
- **ManagedClusterConfig** -- Defines cluster shape (head + worker nodes, GPUs per worker)
- **PVC storage** -- Persistent volumes for sharing data between Ray workers

## Official Documentation

- [Accelerate data processing and training with distributed workloads](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/accelerate_data_processing_and_training_with_distributed_workloads)

## What's in examples/

Examples demonstrate RayJob YAML configurations, PVC-backed data pipelines, multi-GPU training with Ray Train, and autoscaling cluster setups.
