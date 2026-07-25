# SDG Pipeline

**Status:** GA

A Kubeflow Pipeline for running SDG Hub flows at scale on OpenShift AI. Automates synthetic data generation as a reusable pipeline component, enabling integration with broader ML workflows including training and evaluation stages.

## What's Covered

- Packaging SDG Hub flows as KFP components
- Configuring data generation parameters at pipeline level
- Scaling generation across cluster resources
- Chaining SDG output into downstream training pipelines
- Monitoring generation progress and outputs

## Official Documentation

- [Build, Schedule, and Track Machine Learning Pipelines](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/build_schedule_and_track_machine_learning_pipelines)

## What's in examples/

Examples will include KFP pipeline definitions with SDG Hub components, parameterized flow configurations, and integration patterns connecting SDG output to training stages.
