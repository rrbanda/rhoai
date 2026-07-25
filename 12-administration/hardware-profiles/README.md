# Hardware Profiles

**Status:** Technology Preview (TP)

Hardware profiles enable targeting specific worker nodes, accelerator types, or CPU-only nodes for workbenches and model serving in RHOAI 3.4. Leverage Dynamic Resource Allocation (DRA) for GPU scheduling to efficiently assign hardware resources to AI workloads.

## What's Covered

- Creating and managing hardware profiles
- Targeting specific accelerator types (NVIDIA, AMD, Intel)
- Configuring CPU-only profiles for non-GPU workloads
- GPU scheduling with Dynamic Resource Allocation (DRA)
- Assigning profiles to workbenches and serving runtimes

## Official Documentation

- [Provision Hardware Configurations and Resources for Projects](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/provision_hardware_configurations_and_resources_for_projects)

## What's in examples/

Examples will include hardware profile CRD definitions, DRA configuration templates, GPU scheduling policies, and node selector configurations for mixed-hardware clusters.
