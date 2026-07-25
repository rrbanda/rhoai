# Serving Runtimes

**Status:** GA

RHOAI 3.4 ships with two primary serving runtimes for model inference. vLLM handles large language model workloads with high-performance batching and GPU acceleration, while MLServer supports classical ML models in their native formats.

## What's Covered

- **vLLM** -- Default runtime for LLM inference with continuous batching, PagedAttention, and tensor parallelism
- **MLServer** -- Runtime for classical ML models (scikit-learn, XGBoost, LightGBM) without ONNX conversion

## Official Documentation

- [Deploy Large Models Using KServe RawDeployment](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/deploy_large_models_using_the_single-model_serving_platform_kserve_rawdeployment)
