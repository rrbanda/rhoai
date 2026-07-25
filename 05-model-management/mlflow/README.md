# MLflow

## Status

**GA** (General Availability) — New in RHOAI 3.4

## Overview

MLflow provides experiment tracking, run comparison, and model versioning within Red Hat OpenShift AI. It is deployed and managed through the Red Hat-built MLflow Operator. MLflow is integrated into the RHOAI dashboard and included in all workbench images, with support for W&B, MLflow tracking, and TensorBoard loggers out of the box.

## What's Covered

- Deploying MLflow via the MLflow Operator
- Tracking experiments and logging metrics, parameters, and artifacts
- Comparing runs across experiments
- Model versioning through the MLflow model registry
- Dashboard integration for viewing experiment results
- Configuring experiment tracking loggers (W&B, MLflow, TensorBoard)
- Using MLflow from workbench notebooks

## Official Documentation

- [Working with MLflow](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/working_with_mlflow)

## What's in examples/

Examples will cover deploying an MLflow instance, logging training runs from a workbench notebook, comparing experiment results in the dashboard, and configuring different tracking backends.
