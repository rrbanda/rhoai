# Platform Reference

The RHOAI platform includes components and concepts beyond model customization. The repository organizes these into numbered lifecycle folders that map to the full RHOAI workflow.

## Lifecycle Folders

These folders in the [repository root](https://github.com/rrbanda/rhoai) contain examples, READMEs, and configurations for each RHOAI platform area:

| Folder | Topic | Relevance to Model Customization |
|--------|-------|----------------------------------|
| `01-getting-started/` | Platform setup and first steps | Cluster access, project creation |
| `02-workbenches/` | JupyterLab and VS Code workbenches | Development environment for training scripts |
| `03-data-preparation/` | Data ingestion and preprocessing | Preparing documents for SDG Hub |
| `04-training/` | Training jobs and distributed training | Running Training Hub on-cluster |
| `05-model-management/` | Model registry and versioning | Registering fine-tuned models |
| `06-serving/` | Model serving and inference | KServe + vLLM deployment |
| `07-ai-applications/` | AI-powered applications | Integrating served models into apps |
| `08-evaluation/` | Model evaluation and benchmarking | LM-Eval harness, agent evaluation |
| `09-safety-guardrails/` | Safety rails and content filtering | NeMo Guardrails for production |
| `10-monitoring/` | Observability and metrics | Monitoring served model health |
| `11-pipelines/` | Data Science Pipelines (KFP) | Automating the training pipeline |
| `12-administration/` | RBAC, quotas, cluster admin | Platform governance |

## Relationship to These Guides

The GH Pages documentation (`docs/`) focuses on the **model customization workflow** — data generation, training, evaluation, serving, and guardrails. The numbered folders cover the broader RHOAI platform surface.

Where topics overlap (serving, training, evaluation), the GH Pages docs provide the model-customization-specific guidance while the numbered folders contain general-purpose RHOAI platform examples.

## Official Documentation

For comprehensive RHOAI platform documentation, refer to the [Red Hat OpenShift AI product docs](https://docs.redhat.com/en/documentation/red_hat_openshift_ai/latest).
