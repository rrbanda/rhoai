# SDG Hub on Kubeflow Pipelines

**Status:** GA (General Availability)

SDG Hub flows can be run as Kubeflow Pipeline (KFP) components on OpenShift AI, enabling automated synthetic data generation at scale. This integrates SDG into broader ML pipelines alongside data processing, training, and evaluation steps.

## What's Covered

- Packaging SDG Hub flows as KFP components
- Defining and submitting KFP pipelines on OpenShift AI
- Configuring compute resources and model endpoints for pipeline steps
- Chaining SDG components with upstream data processing and downstream training
- Monitoring pipeline runs and retrieving generated datasets

## Official Documentation

- [Generate Synthetic Data](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/customize_models_for_gen_ai_and_agentic_ai_applications/generate-synthetic-data_custom-models)

## What's in examples/

Examples show how to define KFP pipelines that run SDG Hub flows on OpenShift AI, including resource configuration, model endpoint setup, and integration with other pipeline stages.
