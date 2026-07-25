# MLServer Serving Runtime

**Status:** GA (new in 3.4)

MLServer is a serving runtime for deploying classical ML models on RHOAI 3.4. It supports scikit-learn, XGBoost, and LightGBM models in their native serialization formats, eliminating the need for ONNX conversion. MLServer integrates with KServe RawDeployment as a `ServingRuntime` resource.

## What's Covered

- Configuring MLServer as a `ServingRuntime` resource
- Deploying scikit-learn models (pickle, joblib)
- Deploying XGBoost and LightGBM models
- Native format serving without ONNX conversion
- Inference request and response schemas (V2 protocol)
- Resource sizing for classical ML workloads

## Official Documentation

- [Deploy Large Models Using KServe RawDeployment](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/deploy_large_models_using_the_single-model_serving_platform_kserve_rawdeployment)

## What's in examples/

- `ServingRuntime` and `InferenceService` manifests for MLServer
- Sample scikit-learn and XGBoost model deployment workflows
- Inference scripts demonstrating the V2 prediction protocol
