# KServe RawDeployment

**Status:** GA

KServe RawDeployment is the primary model serving platform in RHOAI 3.4, replacing ModelMesh. It deploys trained models as individual inference endpoints with support for both RawDeployment and Knative modes. Endpoints can be secured with authentication and configured with custom resource limits and serving runtimes.

## What's Covered

- Choosing between RawDeployment and Knative serving modes
- Configuring resource requests, limits, and GPU allocation
- Selecting and customizing serving runtimes (vLLM, MLServer)
- Exposing authenticated inference endpoints
- Setting up model storage (S3, PVC, OCI)
- Autoscaling and replica configuration

## Official Documentation

- [Deploy Large Models Using KServe RawDeployment](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/deploy_large_models_using_the_single-model_serving_platform_kserve_rawdeployment)

## What's in examples/

- Sample `InferenceService` manifests for RawDeployment and Knative modes
- Runtime configuration examples for vLLM and MLServer
- Scripts for testing inference endpoints with authentication
