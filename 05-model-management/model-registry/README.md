# Model Registry

## Status

**GA** (General Availability) — RHOAI 3.4

## Overview

The model registry provides centralized storage for versioning, promoting, and sharing models with full metadata and lineage tracking. It uses a PostgreSQL backend and supports RBAC-scoped access so teams can collaborate across projects with proper access controls. Administrators create registries through the RHOAI dashboard; users register models and promote them to serving.

## What's Covered

- Creating and configuring model registries (admin workflow)
- Registering models with metadata and version history
- Promoting registered models to a serving runtime
- Cross-project model sharing with RBAC-scoped access
- PostgreSQL backend configuration
- Model lineage and traceability

## Official Documentation

- [Register, Version, and Promote Models with the Model Registry](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/register_version_and_promote_models_with_the_model_registry)

## What's in examples/

Examples will demonstrate creating a model registry, registering a trained model with version metadata, promoting a model version to a serving endpoint, and querying the registry programmatically.
