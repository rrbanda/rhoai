# Model Management

## Status

**GA** (General Availability) — RHOAI 3.4

## Overview

Model management in Red Hat OpenShift AI provides tools for discovering, versioning, tracking, and promoting machine learning models across their lifecycle. It brings together the model catalog, model registry, and MLflow into a unified workflow accessible from the RHOAI dashboard.

## What's Covered

- **Model Catalog** — Browse and deploy a curated set of validated third-party models. Evaluate models before committing to deployment or register them to the model registry.
- **Model Registry** — Store, version, and promote models with metadata and lineage tracking. Share models across projects with RBAC-scoped access.
- **MLflow** — Track experiments, compare runs, and version models. Red Hat-built deployment managed through the MLflow Operator.

## Structure

```
05-model-management/
├── model-catalog/   # Model discovery, evaluation, and deployment
├── model-registry/  # Model versioning, promotion, and cross-project sharing
└── mlflow/          # Experiment tracking and run comparison
```

## Official Documentation

- [Model Catalog](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/discover_evaluate_register_and_deploy_models_from_the_model_catalog)
- [Model Registry](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/register_version_and_promote_models_with_the_model_registry)
- [MLflow](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/working_with_mlflow)

## What's in examples/

Examples will include end-to-end workflows: registering a model from the catalog, promoting a model through the registry to serving, and tracking training experiments with MLflow.
