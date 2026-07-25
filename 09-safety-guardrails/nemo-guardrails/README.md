# NeMo Guardrails

**Status:** GA (new in 3.4)

NVIDIA NeMo Guardrails provides programmable control over LLM conversations in RHOAI 3.4. Define rails to detect sensitive data, filter inappropriate content, enforce topical boundaries, and apply custom validation logic. Orchestrate detectors to filter both LLM inputs and outputs in real time.

## What's Covered

- Deploying NeMo Guardrails on OpenShift AI
- Defining rails for sensitive data detection and content filtering
- Configuring topical guardrails to keep conversations on-topic
- Writing custom validation rules
- Orchestrating multiple detectors for input/output filtering
- Integrating guardrails with model serving endpoints

## Official Documentation

- [Ensuring AI Safety with Guardrails](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/ensuring_ai_safety_with_guardrails)

## What's in examples/

Examples will include Colang rail definitions, detector configurations, integration patterns with vLLM serving endpoints, and sample policies for common enterprise safety requirements.
