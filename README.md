# Red Hat OpenShift AI (RHOAI) 3.4 -- Feature Reference

A hands-on reference repository organized by RHOAI 3.4 features and capabilities. Each folder maps to a lifecycle phase of the AI/ML workflow, with documentation and runnable examples.

> **Scope note:** The `docs/` site and [GitHub Pages](https://rrbanda.github.io/rhoai/) focus on the **Model Customization** track (data generation, training, evaluation, serving). Other numbered folders below provide reference configurations for additional RHOAI capabilities.

## Start Here

| Use case | Notebook | What it covers |
|----------|----------|---------------|
| **Customize a model with your own data** | [model_customization_e2e.ipynb](end-to-end-examples/knowledge-tuning/model_customization_e2e.ipynb) | Document prep -> SDG Hub data generation -> Training Hub (SFT/OSFT) -> Evaluation -> Serving |
| **Teach a model to use your tools** | [mcp_distillation_e2e.ipynb](end-to-end-examples/mcp-distillation/mcp_distillation_e2e.ipynb) | MCP server exploration -> Synthetic tool-use data -> LoRA GRPO training -> Evaluation |
| **Build a financial tool-calling model** | [tool-calling-financial/](end-to-end-examples/tool-calling-financial/) | MCP distillation -> LoRA SFT -> KServe + vLLM -> NeMo Guardrails |

## Feature Maturity Matrix

| # | Area | Capability | Status |
|---|------|-----------|--------|
| 01 | Getting Started | Platform Setup, First Project, Fraud Detection Tutorial | GA |
| 02 | Workbenches | IDE Images, Custom Images, Data Connections | GA |
| 03 | Data Preparation | SDG Hub (Knowledge Tuning, MCP Distillation, KFP Pipeline) | GA |
| 03 | Data Preparation | Data Processing (Docling) | GA |
| 03 | Data Preparation | Feature Store | **TP** |
| 04 | Training | Model Customization (SFT, OSFT, LoRA, GRPO) | GA |
| 04 | Training | Distributed Workloads (Kubeflow Trainer, KubeRay) | GA |
| 04 | Training | AutoML | **DP** |
| 05 | Model Management | Model Catalog | GA |
| 05 | Model Management | Model Registry | GA |
| 05 | Model Management | MLflow | GA |
| 06 | Serving | KServe RawDeployment | GA |
| 06 | Serving | Distributed Inference (llm-d) | GA |
| 06 | Serving | Models-as-a-Service (MaaS) | GA |
| 06 | Serving | Serving Runtimes (vLLM, MLServer) | GA |
| 07 | AI Applications | Llama Stack (RAG, Agents) | **TP** |
| 07 | AI Applications | AutoRAG | **DP** |
| 07 | AI Applications | Gen AI Playground | **TP** |
| 07 | AI Applications | MCP Servers / AI Available Assets | **TP** |
| 08 | Evaluation | LM-Eval | GA |
| 09 | Safety | NeMo Guardrails | GA |
| 09 | Safety | TrustyAI | GA |
| 10 | Monitoring | Model Monitoring (Bias, Drift) | GA |
| 10 | Monitoring | Platform Observability | **TP** |
| 10 | Monitoring | Usage Telemetry | GA |
| 11 | Pipelines | KFP Pipelines (SDG, Training) | GA |
| 11 | Pipelines | Spark Operator | GA |
| 12 | Administration | User Access / RBAC | GA |
| 12 | Administration | Hardware Profiles | **TP** |
| 12 | Administration | Certificates | GA |
| 12 | Administration | Backup / Restore | GA |

**Legend:** GA = Generally Available | TP = Technology Preview | DP = Developer Preview

## Repository Structure

```
rhoai/
├── 01-getting-started/        Platform setup, first project, fraud detection tutorial
├── 02-workbenches/            IDE images, custom images, S3 data connections
├── 03-data-preparation/       SDG Hub, data processing (Docling), Feature Store
├── 04-training/               Model customization (SFT/OSFT/LoRA/GRPO), distributed workloads, AutoML
├── 05-model-management/       Model catalog, model registry, MLflow
├── 06-serving/                KServe, llm-d distributed inference, MaaS, serving runtimes
├── 07-ai-applications/        Llama Stack (RAG/Agents), AutoRAG, Gen AI Playground, MCP
├── 08-evaluation/             LM-Eval benchmarking
├── 09-safety-guardrails/      NeMo Guardrails, TrustyAI
├── 10-monitoring/             Model monitoring, platform observability, telemetry
├── 11-pipelines/              KFP pipelines (SDG + training), Spark Operator
├── 12-administration/         RBAC, hardware profiles, certificates, backup/restore
└── end-to-end-examples/       Knowledge tuning, MCP distillation, tool-calling financial, RAG application
```

## RHOAI Lifecycle Flow

```
Getting Started → Workbenches → Data Preparation → Training → Model Management
       ↓                                                           ↓
  Evaluation ← Safety/Guardrails ← AI Applications ← Serving ←────┘
       ↓
  Monitoring → Pipelines → Administration
```

## Prerequisites

- OpenShift Container Platform 4.x cluster
- Red Hat OpenShift AI Operator 3.4 installed
- `oc` CLI configured with cluster access
- GPU-enabled worker nodes (for training and inference workloads)
- S3-compatible object storage (for pipelines and model artifacts)

## Official Documentation

| Resource | URL |
|----------|-----|
| RHOAI 3.4 Docs | https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4 |
| Release Notes | https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/release_notes |
| Red Hat AI Examples | https://github.com/red-hat-data-services/red-hat-ai-examples |
| SDG Hub | https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub |
| Training Hub | https://github.com/Red-Hat-AI-Innovation-Team/training_hub |
| llm-d | https://llm-d.ai |
| Supported Configurations | https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/supported_configurations |

## License

Apache-2.0
