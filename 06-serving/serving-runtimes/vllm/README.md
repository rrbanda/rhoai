# vLLM Serving Runtime

**Status:** GA

vLLM is the default serving runtime for LLM inference in RHOAI 3.4. It provides high-performance inference with continuous batching, PagedAttention for efficient GPU memory management, and tensor parallelism for distributing models across multiple GPUs. vLLM powers both KServe RawDeployment and llm-d serving paths.

## What's Covered

- Configuring vLLM as a `ServingRuntime` resource
- Continuous batching for maximizing throughput
- PagedAttention and GPU memory optimization
- Tensor parallelism for multi-GPU model distribution
- Model format compatibility (Hugging Face, safetensors)
- Tuning inference parameters (max tokens, temperature, top-p)

## Official Documentation

- [Deploy Large Models Using KServe RawDeployment](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/deploy_large_models_using_the_single-model_serving_platform_kserve_rawdeployment)

## What's in examples/

- `ServingRuntime` manifests with vLLM configuration options
- Multi-GPU tensor parallelism setup examples
- Inference request scripts for chat and completion endpoints
