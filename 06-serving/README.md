# Model Serving

**Status:** GA

Model serving in RHOAI 3.4 provides multiple pathways for deploying trained models into production. From single-model endpoints to distributed LLM inference and centralized governance, the serving stack covers the full spectrum of deployment needs.

## What's Covered

- **KServe RawDeployment** -- The primary single-model serving platform, supporting both RawDeployment and Knative modes with authenticated endpoints
- **Distributed Inference with llm-d** -- Kubernetes-native framework for serving LLMs at scale with intelligent routing and disaggregated serving
- **Models-as-a-Service (MaaS)** -- Centralized governance layer for LLM access with subscription-based quotas and self-service API key management
- **Serving Runtimes** -- Runtime engines including vLLM (LLM inference) and MLServer (classical ML models)

## Key Changes in 3.4

- ModelMesh is replaced by KServe RawDeployment as the primary serving platform
- llm-d introduces `LLMInferenceService` CR, replacing `InferenceService` for distributed LLM workloads
- Models-as-a-Service is new in 3.4, providing centralized LLM governance
- MLServer is now GA for classical ML model serving without ONNX conversion

## Official Documentation

- [Deploy Large Models Using KServe RawDeployment](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/deploy_large_models_using_the_single-model_serving_platform_kserve_rawdeployment)
- [Deploy Models Using Distributed Inference with llm-d](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/deploy_models_using_distributed_inference_with_llm-d)
- [Govern LLM Access with Models-as-a-Service](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/govern_llm_access_with_models-as-a-service)
