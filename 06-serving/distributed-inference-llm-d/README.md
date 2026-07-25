# Distributed Inference with llm-d

**Status:** GA

llm-d is a Kubernetes-native framework for serving LLMs at scale on RHOAI 3.4. It introduces the `LLMInferenceService` custom resource (replacing `InferenceService`) and provides intelligent scheduling, prefix-cache-aware routing, and disaggregated serving for high-throughput LLM workloads.

## What's Covered

- Creating and configuring `LLMInferenceService` resources
- Prefix-cache-aware routing for improved latency on repeated prompts
- Disaggregated serving (separate prefill and decode phases)
- Multi-node deployment for models that exceed single-node GPU memory
- Intelligent scheduling and load balancing across replicas
- Integration with vLLM as the inference backend

## Official Documentation

- [Deploy Models Using Distributed Inference with llm-d](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/deploy_models_using_distributed_inference_with_llm-d)
- [llm-d Project](https://llm-d.ai)

## What's in examples/

- `LLMInferenceService` manifests for single-node and multi-node deployments
- Configuration examples for prefix caching and disaggregated serving
- Load testing scripts demonstrating routing behavior
