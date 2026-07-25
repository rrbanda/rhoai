# Platform Setup

**Status:** GA

Install and configure the Red Hat OpenShift AI Operator on your OpenShift cluster. This guide covers both CLI and web console installation methods, the DataScienceCluster custom resource, and how to enable or disable individual components.

## Prerequisites

- OpenShift 4.x cluster
- `cluster-admin` access

## What's Covered

- Installing the RHOAI Operator via CLI (`oc`) and the OpenShift web console
- Creating and configuring the `DataScienceCluster` custom resource
- Enabling components by setting `managementState: Managed`:
  - Dashboard
  - Workbenches
  - KServe / ModelMesh
  - Model Registry
  - Ray
  - Training Operator
  - Data Science Pipelines
- Disabling components with `managementState: Removed`
- Verifying the installation and component health

## Official Documentation

- [Installing and Uninstalling OpenShift AI Self-Managed](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/installing_and_uninstalling_openshift_ai_self-managed)

## What's in examples/

Sample `DataScienceCluster` CR manifests for common configurations (minimal install, full install, GPU-enabled) and shell scripts to automate operator installation.
