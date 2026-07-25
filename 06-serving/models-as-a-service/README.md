# Models-as-a-Service (MaaS)

**Status:** GA (new in 3.4)

Models-as-a-Service provides centralized governance for LLM access in RHOAI 3.4. It enables platform administrators to manage model subscriptions with token quotas, while giving users self-service API key management and a unified interface for both locally-deployed and external provider models.

## What's Covered

- Subscription-based model access with configurable token quotas
- Self-service API key generation and management
- Role-based access control (RBAC) for model endpoints
- Usage tracking and quota enforcement
- External provider routing (OpenAI, Anthropic configured as Technology Preview)
- Integrating locally-served models into the MaaS catalog

## Official Documentation

- [Govern LLM Access with Models-as-a-Service](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/govern_llm_access_with_models-as-a-service)

## What's in examples/

- MaaS subscription and quota configuration manifests
- API key provisioning workflows
- Scripts for querying models through the MaaS gateway
