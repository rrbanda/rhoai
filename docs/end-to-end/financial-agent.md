# End-to-End Financial Agent Pipeline

Build a tool-calling agent for financial services by fine-tuning Qwen3-4B with LoRA SFT on domain-specific tool-calling demonstrations, then deploying it behind NeMo Guardrails for compliance. This pipeline uses SDG Hub's MCP distillation to generate training data from a financial MCP server with 15 tools.

## RHOAI Feature Matrix

| Feature | RHOAI Version | Status |
|---------|--------------|--------|
| SDG Hub MCP Distillation | 3.4+ | GA |
| Training Hub LoRA SFT | 3.4+ | GA |
| LoRA KFP Pipeline | 3.4+ | GA |
| KServe RawDeployment + vLLM | 3.4+ | GA |
| NeMo Guardrails | 3.4+ | GA |
| LM-Eval | 3.4+ | GA |
| NeMo + MCP Gateway | 3.5 EA2 | **TP** |
| Validated Tool-Calling Config | 3.5 EA2 | **TP** |

The core pipeline (Steps 0-7) runs fully on RHOAI 3.4. RHOAI 3.5 features are additive enhancements.

## Pipeline Overview

![Financial Agent Architecture](images/financial-agent-architecture.png)

## Prerequisites

- RHOAI 3.4+ cluster (3.5 EA2 for MCP Gateway)
- GPU: 1x NVIDIA L4 24GB (with QLoRA 4-bit) or 1x L40/A100 for full-precision training
- Teacher model API key (Gemini 3.6 Flash recommended for cost, or GPT-4o) for data generation
- Langflow instance with a frontier model agent connected to the financial MCP server
- Python 3.11+, `oc` CLI authenticated to your cluster

## Step 0: Start the Financial MCP Server

The demo server provides 15 financial tools organized into domains — groups of tools covering portfolio management, market data, risk analysis, and trade execution.

```bash
cd end-to-end-examples/financial-agent/demo_server/
pip install fastmcp
python server.py
# Server starts on http://localhost:8009
```

| Domain | Tools |
|--------|-------|
| Market Data | `get_stock_quote`, `get_market_summary`, `get_historical_prices`, `screen_stocks` |
| Portfolio Mgmt | `get_portfolio_positions`, `get_portfolio_performance`, `get_account_summary`, `get_transaction_history` |
| Risk & Analytics | `calculate_portfolio_risk`, `get_sector_exposure`, `run_stress_test`, `analyze_stock` |
| Trading & Compliance | `submit_trade_order`, `check_compliance`, `get_regulatory_status` |

## Step 1: Generate Tool-Call Training Data

**RHOAI Feature:** SDG Hub MCP Distillation (GA)

```bash
cd end-to-end-examples/financial-agent/examples/
cp .env.example .env
# Edit .env with your API keys and Langflow URL

python 01_generate_tool_data.py --num-samples 10
```

The pipeline:

1. A frontier model (via Langflow) **explores** all 15 financial tools
2. A teacher LLM **synthesizes** realistic financial questions grounded in the exploration
3. The frontier model **solves** each question via actual MCP tool calls (expert trajectories)
4. Quality filters **remove** incomplete or low-quality examples
5. Output: `generated_data/distillation_output.parquet`

## Step 2: Format Training Data

Convert raw traces into chat-format JSONL:

```bash
python 02_format_training_data.py
```

Output format:

```json
{"messages": [
  {"role": "system", "content": "<tool declarations>"},
  {"role": "user", "content": "What's the risk-adjusted return on my tech portfolio?"},
  {"role": "assistant", "content": "", "function_call": {"name": "get_portfolio_positions", "arguments": "..."}},
  {"role": "function", "content": "{...}", "name": "get_portfolio_positions"},
  {"role": "assistant", "content": "Your tech portfolio has a Sharpe ratio of..."}
]}
```

## Step 3: Fine-Tune with LoRA SFT

**RHOAI Feature:** Training Hub LoRA SFT (GA) + Kubeflow Trainer (GA)

Three training paths are available. **Option B (TrainJob)** is the recommended approach — it runs directly on-cluster via the Kubeflow Trainer and was validated end-to-end on RHOAI 3.4.2.

### Option A: Local Training (for development)

```bash
# Single L4 24GB with QLoRA (default settings)
python 03_train_lora_sft.py

# A100 80GB with higher capacity
python 03_train_lora_sft.py --lora-r 32 --lora-alpha 64 --max-seq-len 8192 --no-4bit

# Multi-GPU data-parallel
torchrun --nproc-per-node=2 03_train_lora_sft.py
```

### Option B: Direct TrainJob on RHOAI (recommended, validated)

This approach creates a Kubeflow `TrainJob` custom resource that uses the `training-hub` ClusterTrainingRuntime. It runs directly on the GPU node with no pipeline server or RWX storage required.

**Prerequisite:** The `trainer` component must be enabled in your DataScienceCluster, and the `training-hub` ClusterTrainingRuntime must exist:

```bash
oc get clustertrainingruntimes training-hub
```

#### Step 3B.1: Create the workspace PVC and training script

```bash
cat <<'YAML' | oc apply -n financial-agent -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-workspace
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: gp3-csi
  resources:
    requests:
      storage: 50Gi
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-data-script
data:
  prepare_and_train.py: |
    """Download dataset and run LoRA SFT via Training Hub."""
    import os, json

    WORKSPACE = os.environ.get("WORKSPACE_PATH", "/workspace")
    DATA_DIR = os.path.join(WORKSPACE, "data")
    OUTPUT_DIR = os.path.join(WORKSPACE, "output")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Step 1: Downloading dataset from HuggingFace...")
    print("=" * 60)
    from datasets import load_dataset
    ds = load_dataset("LipengCS/Table-GPT", "All", split="train[:50]")

    jsonl_path = os.path.join(DATA_DIR, "training_data.jsonl")
    with open(jsonl_path, "w") as f:
        for row in ds:
            f.write(json.dumps({
                "messages": row.get("messages", row.get("conversations", []))
            }) + "\n")
    print(f"  Saved {len(ds)} examples to {jsonl_path}")

    print("=" * 60)
    print("Step 2: Running LoRA SFT training...")
    print("=" * 60)
    from training_hub import lora_sft
    result = lora_sft(
        model_path="Qwen/Qwen3-4B",
        data_path=jsonl_path,
        ckpt_output_dir=OUTPUT_DIR,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        num_epochs=1,
        learning_rate=2e-4,
        effective_batch_size=32,
        micro_batch_size=2,
        max_seq_len=2048,
        load_in_4bit=True,
        lr_scheduler="cosine",
        seed=42,
    )
    print("=" * 60)
    print("Training complete!")
    if isinstance(result, dict):
        for k, v in result.items():
            if k not in ("model", "tokenizer"):
                print(f"  {k}: {v}")
    print(f"  Output saved to: {OUTPUT_DIR}")
    print("=" * 60)
YAML
```

#### Step 3B.2: Create the TrainJob

```bash
cat <<'YAML' | oc apply -n financial-agent -f -
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: financial-agent-lora-sft
spec:
  runtimeRef:
    name: training-hub
    apiGroup: trainer.kubeflow.org
    kind: ClusterTrainingRuntime
  trainer:
    command:
      - python
      - /scripts/prepare_and_train.py
    numNodes: 1
    resourcesPerNode:
      requests:
        cpu: "2"
        memory: "10Gi"
        nvidia.com/gpu: "1"
      limits:
        cpu: "2"
        memory: "10Gi"
        nvidia.com/gpu: "1"
    env:
      - name: WORKSPACE_PATH
        value: /workspace
  podTemplateOverrides:
    - targetJobs:
        - name: node
      spec:
        tolerations:
          - key: nvidia.com/gpu
            operator: Exists
            effect: NoSchedule
        volumes:
          - name: workspace
            persistentVolumeClaim:
              claimName: training-workspace
          - name: scripts
            configMap:
              name: training-data-script
        containers:
          - name: node
            volumeMounts:
              - name: workspace
                mountPath: /workspace
              - name: scripts
                mountPath: /scripts
YAML
```

!!! warning "Key details in the TrainJob YAML"
    - **`targetJobs` name must be `node`** — the `training-hub` ClusterTrainingRuntime uses `node` as the job name, not `Trainer` or `Initializer`
    - **Container name must be `node`** — matches the runtime's container name
    - **GPU toleration is required** if GPU nodes have the `nvidia.com/gpu=True:NoSchedule` taint
    - **Resource requests** must fit within the GPU node's allocatable capacity (e.g., `g6.xlarge` has ~3.5 CPU and ~14GB RAM)

#### Step 3B.3: Monitor training

```bash
# Watch the pod start
oc get pods -n financial-agent -l job-name=financial-agent-lora-sft-node-0 -w

# Once Running, stream the logs
oc logs -f -l job-name=financial-agent-lora-sft-node-0 -n financial-agent
```

Expected output:

```
============================================================
Step 1: Downloading dataset from HuggingFace...
============================================================
  Saved 50 examples to /workspace/data/training_data.jsonl
============================================================
Step 2: Running LoRA SFT training...
============================================================
🦥 Unsloth 2026.4.5 patched 36 layers with 36 QLoRA layers
Trainable parameters = 16,544,768 / 4,025,356,288 (0.41%)
{'loss': 1.817, 'learning_rate': 2e-04, 'epoch': 0.5}
{'loss': 1.511, 'learning_rate': 0.0, 'epoch': 1.0}
{'train_runtime': 38.5, 'train_samples_per_second': 1.3}
============================================================
Training complete!
  Output saved to: /workspace/output
============================================================
```

#### Step 3B.4: Verify and retrieve the checkpoint

```bash
# Verify TrainJob completed
oc get trainjob financial-agent-lora-sft -n financial-agent

# The LoRA adapter is saved in the PVC at /workspace/output
# To use it for deployment, copy to S3 or a model registry
```

### Option C: LoRA KFP Pipeline on RHOAI (for multi-stage automation)

**Source:** [pipelines-components/pipelines/training/finetuning/lora](https://github.com/red-hat-data-services/pipelines-components/tree/main/pipelines/training/finetuning/lora)

The KFP pipeline wraps the same LoRA SFT algorithm in a four-stage automated pipeline. Use this when you need automatic dataset download, evaluation, and model registry as a repeatable workflow.

```bash
# Compile the pipeline YAML
python 03b_train_kfp_pipeline.py --compile-only

# Upload pipeline.yaml to RHOAI Dashboard > Pipelines > Import Pipeline
# Create a run with parameters:
#   dataset_uri = hf://LipengCS/Table-GPT:All  (or your S3 dataset)
#   base_model = Qwen/Qwen3-4B
#   lora_r = 16, lora_alpha = 32
```

The four pipeline stages:

1. **Dataset Download** — fetches and validates training data
2. **LoRA Training** — fine-tunes via Kubeflow Trainer + Training Hub Unsloth backend
3. **Evaluation** — LM-Eval harness benchmarks
4. **Model Registry** — registers the fine-tuned model (optional)

Prerequisites for KFP pipeline:

- `dashboard`, `trainer`, `aipipelines` components enabled in RHOAI
- `argoWorkflowsControllers` set to `Managed` in the DataScienceCluster
- Pipeline server (DSPA) running in your namespace
- RWX storage class (default: `nfs-csi`), or switch to RWO (see [cluster notes](#important-cluster-notes))
- `kubernetes-credentials` secret with cluster API access

Create the required secret:

```bash
oc create secret generic kubernetes-credentials \
  --from-literal=KUBERNETES_SERVER_URL="https://api.your-cluster.com:6443" \
  --from-literal=KUBERNETES_AUTH_TOKEN="$(oc whoami -t)"
```

## Step 4: Deploy the Fine-Tuned Model

**RHOAI Feature:** KServe RawDeployment (GA)

```bash
python 04_deploy_model.py
```

Creates a KServe InferenceService with vLLM serving the fine-tuned model. Tool-calling is enabled via `--enable-auto-tool-choice` and `--tool-call-parser hermes` (validated for Qwen3).

## Step 5: Evaluate

**RHOAI Feature:** LM-Eval (GA) + Agent Evaluation

```bash
python 05_evaluate_agent.py \
  --model-endpoint http://financial-agent.rhoai-serving.svc:8080 \
  --output evaluation_results.json
```

Measures tool recall, tool precision, argument correctness, multi-step success rate, and LLM-as-judge scoring.

## Step 6: Configure Guardrails

**RHOAI Feature:** NeMo Guardrails (GA) + MCP Gateway (3.5 TP)

```bash
python 06_configure_guardrails.py \
  --model-endpoint https://financial-agent.apps.cluster.example.com/v1
```

- **Tier 1 (GA):** PII detection, jailbreak protection, financial disclaimers
- **Tier 2 (3.5 TP):** MCP Gateway auto-enforcement on all tool calls

## Step 7: Run the Deep Agent Harness

```bash
pip install deepagents langchain-openai langgraph-cli[inmem]
export MODEL_ENDPOINT=https://financial-agent-predictor.apps.your-cluster.com/v1

cd end-to-end-examples/financial-agent/examples/
langgraph dev

# Or headless:
python 07_deep_agent.py "What is the risk-adjusted performance of portfolio PORT-0001?"
```

The Deep Agent wraps the fine-tuned Qwen3-4B with task planning, tool orchestration, and long-term memory via the `deepagents` library.

## Resource Estimates

| Phase | Resource | Time |
|-------|----------|------|
| Data Generation | 1x CPU + Teacher API | 2-4 hours |
| Training (local, L4) | 1x L4 24GB | 1-3 hours |
| Training (KFP, L40) | 1x L40 48GB | 1-2 hours |
| Serving | 1x L4 24GB+ | Ongoing |
| Evaluation | 1x GPU + Model endpoint | 30-60 min |

## Troubleshooting

| Symptom | Resolution |
|---------|------------|
| "MCP Server Distillation flow not found" | `pip install sdg_hub[dev]` |
| "CUDA out of memory during training" | Enable QLoRA: `--load-in-4bit` (default), or reduce `--max-seq-len` |
| "Tool call parser error" during inference | Verify `--tool-call-parser` matches model (`hermes` for Qwen3) |
| KFP pipeline: "storageclass not found" | Set `--storage-class` to your cluster's RWX class |
| KFP pipeline: "kubernetes-credentials not found" | Create the secret (see Step 3B prerequisites) |

## Validated on RHOAI

!!! success "End-to-End Validated"
    This pipeline has been validated on **RHOAI 3.4.2** with **OpenShift 4.18.21** on a **g6.xlarge** instance (1x NVIDIA L4 24GB GPU).

| Component | What Was Tested | Result |
|-----------|----------------|--------|
| MCP Server | 15 financial tools via FastMCP | All tools callable |
| SDG Hub | MCP distillation flow loads, teacher model (Gemini 3.6 Flash) connects | Flow discovers 22 blocks, input dataset builds |
| Data Formatting | Parquet → chat-format JSONL | Function-call structure validated |
| LoRA SFT Training | Kubeflow Trainer (`training-hub` runtime), Qwen3-4B, 50 examples, 1 epoch | Loss 1.817→1.511 in 38 seconds |
| KFP Pipeline | Compile → upload to DSPA pipeline server | 2,367-line YAML, pipeline visible in dashboard |
| Model Deployment | KServe RawDeployment + vLLM | Manifest generates correctly |
| Guardrails | NemoGuardrails CR | CR generates correctly |

### Important Cluster Notes

!!! warning "GPU Node Taint"
    GPU nodes with taint `nvidia.com/gpu=True:NoSchedule` require a toleration in the TrainJob's `podTemplateOverrides`. Without this, training pods stay `Pending`.

!!! tip "Resource Sizing"
    On `g6.xlarge` (3.5 CPU allocatable, ~14GB RAM), use `cpu: 2, memory: 10Gi` for training pod requests. Scale down the model predictor deployment before training if the GPU node is shared.

!!! tip "KFP Storage"
    The KFP pipeline creates PVCs that default to `ReadWriteMany` (RWX). If your cluster only has RWO storage (e.g., `gp3-csi`), either switch to `ReadWriteOnce` in the pipeline parameters or use the direct TrainJob approach with `03_train_lora_sft.py`.

!!! info "Argo Workflows"
    If KFP pipeline runs stay in pending state, enable the Argo workflow controller: `oc patch dsc default-dsc --type=merge -p '{"spec":{"components":{"aipipelines":{"argoWorkflowsControllers":{"managementState":"Managed"}}}}}'`

## Source Code

- [Financial Agent Example](https://github.com/rrbanda/rhoai/tree/main/end-to-end-examples/financial-agent)

## Related

- [Financial Domain Guide](../domains/financial.md)
- [Training Hub LoRA docs](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/docs/algorithms/lora.md)
- [LoRA KFP Pipeline](https://github.com/red-hat-data-services/pipelines-components/tree/main/pipelines/training/finetuning/lora)
- [Serving](../serving/index.md)
- [Guardrails](../guardrails/index.md)
