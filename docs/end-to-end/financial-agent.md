# Tool-Calling Model Pipeline (Financial Example)

Fine-tune a model to make accurate tool calls by training on domain-specific demonstrations, then deploy it with vLLM tool-calling support behind NeMo Guardrails. This pipeline uses SDG Hub's **MCP distillation for data generation**, then trains with **LoRA SFT** (not GRPO). The same technique works with any MCP server and any supported base model — financial services is the validated example.

!!! note "MCP distillation = data generation technique"
    "MCP distillation" refers to the **data generation step** — a teacher model explores your MCP server and produces tool-use traces. This pipeline trains on those traces with **LoRA SFT**. For a GRPO-based training variant, see the [MCP Distillation (GRPO) Pipeline](mcp-distillation.md).

## RHOAI Feature Matrix

| Feature | RHOAI Version | Status |
|---------|--------------|--------|
| SDG Hub MCP Distillation | 3.4+ | GA |
| Training Hub LoRA SFT | 3.4+ | GA |
| LoRA KFP Pipeline | 3.4+ | GA |
| KServe RawDeployment + vLLM | 3.4+ | GA |
| NeMo Guardrails | 3.4+ | GA |
| Agent Evaluation | 3.4+ | GA |
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

First, create the namespace where all resources will be deployed:

```bash
oc new-project financial-agent
```

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

!!! warning "This ConfigMap uses validation data, NOT your financial data"
    The ConfigMap below uses a small public HuggingFace dataset (`LipengCS/Table-GPT`) to validate the training infrastructure. **This is not your financial tool-calling data from Steps 1-2.** It lets you verify that the TrainJob, GPU scheduling, and LoRA SFT work correctly before investing time in data generation.

    **For production, you must replace this dataset** with your own financial tool-calling JSONL from Steps 1-2. Upload your `training_data.jsonl` to the PVC first:

    ```bash
    # Copy your Step 2 output to the training PVC
    oc cp generated_data/training_data.jsonl \
      $(oc get pod -l job-name=copy-data -o name -n financial-agent):/workspace/data/training_data.jsonl \
      -n financial-agent
    ```

    Then modify the ConfigMap script to skip the HuggingFace download and read directly from `/workspace/data/training_data.jsonl`.

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
    """Download dataset and run LoRA SFT via Training Hub.

    This script uses a public HuggingFace dataset for infrastructure
    validation. For production, replace the download section with your
    own financial tool-calling JSONL from Steps 1-2.
    """
    import os, json

    WORKSPACE = os.environ.get("WORKSPACE_PATH", "/workspace")
    DATA_DIR = os.path.join(WORKSPACE, "data")
    OUTPUT_DIR = os.path.join(WORKSPACE, "output")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    jsonl_path = os.path.join(DATA_DIR, "training_data.jsonl")

    if os.path.exists(jsonl_path):
        print("=" * 60)
        print("Using existing training data from PVC...")
        with open(jsonl_path) as f:
            count = sum(1 for _ in f)
        print(f"  Found {count} examples at {jsonl_path}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("Step 1: Downloading validation dataset from HuggingFace...")
        print("  (Replace this with your Step 2 JSONL for production)")
        print("=" * 60)
        from datasets import load_dataset
        ds = load_dataset("LipengCS/Table-GPT", "All", split="train[:50]")
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
# Create a run with parameters (use these exact KFP parameter names):
#   phase_01_dataset_man_data_uri = hf://LipengCS/Table-GPT:All  (or your S3 dataset)
#   phase_02_train_man_train_model = Qwen/Qwen3-4B
#   phase_02_train_man_lora_r = 16
#   phase_02_train_man_lora_alpha = 32
#   phase_02_train_man_train_epochs = 2

# Or run directly from CLI (the script maps CLI flags to KFP params):
python 03b_train_kfp_pipeline.py \
  --dataset-uri hf://LipengCS/Table-GPT:All \
  --base-model Qwen/Qwen3-4B \
  --lora-r 16 --lora-alpha 32 --num-epochs 2
```

The four pipeline stages:

1. **Dataset Download** — fetches and validates training data
2. **LoRA Training** — fine-tunes via Kubeflow Trainer + Training Hub Unsloth backend
3. **Evaluation** — LM-Eval harness benchmarks (arc_easy by default)
4. **Model Registry** — registers the fine-tuned model (optional)

!!! note "KFP pipeline vs standalone evaluation"
    The KFP pipeline's evaluation stage uses LM-Eval (general benchmarks like arc_easy). This is different from the standalone `05_evaluate_agent.py` in Step 5, which evaluates tool-calling quality specifically. Both are complementary — use the KFP eval to check for capability regression and Step 5 to measure tool-use accuracy.

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

## Step 4: Deploy the Fine-Tuned Model on RHOAI

**RHOAI Feature:** KServe RawDeployment (GA) + vLLM ServingRuntime (GA)

After training completes, the LoRA adapter weights are on the PVC at `/workspace/output`. This step deploys the model on RHOAI using YAML manifests applied with `oc apply`. All manifests are in the [`serving/`](https://github.com/rrbanda/rhoai/tree/main/end-to-end-examples/financial-agent/serving) directory.

```
serving/
├── 01-serving-runtime.yaml          vLLM ServingRuntime (only if not preinstalled)
├── 02-inferenceservice-lora.yaml    Option A: LoRA adapter from PVC (recommended)
├── 03-inferenceservice-merged.yaml  Option B: Merged model from S3
├── 04-s3-data-connection.yaml       S3 credentials secret (Option B only)
└── 05-route.yaml                    OpenShift Route for external access
```

### 4.1 Prerequisites — verify the serving stack

```bash
# Verify KServe is enabled (kserve component should be "Managed")
oc get dsc default-dsc -o jsonpath='{.spec.components.kserve.managementState}' && echo
# Expected: Managed

# Verify the vLLM ServingRuntime exists
oc get servingruntimes -n redhat-ods-applications | grep vllm
# Expected: vllm-runtime-...   (preinstalled by RHOAI)
```

!!! info "No vLLM runtime?"
    The vLLM ServingRuntime is preinstalled by RHOAI 3.4+. If missing, enable it via **RHOAI Dashboard → Settings → Serving runtimes → vLLM ServingRuntime for KServe**, or apply the provided manifest: `oc apply -f serving/01-serving-runtime.yaml -n financial-agent`

### 4.2 Option A: Serve the LoRA adapter directly (recommended)

vLLM natively supports LoRA adapters — no merging required. The base model (Qwen3-4B) is pulled from HuggingFace and the adapter (~50MB) is mounted from the training PVC.

**Deploy:**

```bash
# Apply the InferenceService
oc apply -f serving/02-inferenceservice-lora.yaml -n financial-agent

# Expose the endpoint externally
oc apply -f serving/05-route.yaml -n financial-agent
```

**Monitor readiness:**

```bash
# Watch until READY=True (typically 2-5 minutes for initial model download)
oc get inferenceservice financial-agent-lora -n financial-agent -w

# Check the predictor pod
oc get pods -n financial-agent -l serving.kserve.io/inferenceservice=financial-agent-lora

# Stream vLLM startup logs
oc logs -f deployment/financial-agent-lora-predictor -n financial-agent
```

Model is ready when logs show:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080
```

**Get the inference endpoint:**

```bash
ROUTE_URL=$(oc get route financial-agent -n financial-agent -o jsonpath='{.spec.host}')
echo "Inference endpoint: https://${ROUTE_URL}"
```

**Verify with a tool-calling request:**

```bash
curl -sk "https://${ROUTE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "financial-agent",
    "messages": [
      {"role": "user", "content": "What is the current price of AAPL?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_stock_quote",
        "description": "Get a real-time stock quote",
        "parameters": {
          "type": "object",
          "properties": {
            "ticker": {"type": "string", "description": "Stock ticker symbol"}
          },
          "required": ["ticker"]
        }
      }
    }],
    "tool_choice": "auto",
    "max_tokens": 256
  }'
```

Expected response:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "tool_calls": [{
        "function": {
          "name": "get_stock_quote",
          "arguments": "{\"ticker\": \"AAPL\"}"
        }
      }]
    }
  }]
}
```

!!! tip "Why serve the adapter directly?"
    - **No merge step** — deploy immediately after training completes
    - **Storage efficient** — adapter is ~50MB vs 8GB+ for a merged model
    - **Multi-adapter support** — serve financial, medical, and other adapters from the same base model simultaneously
    - **Per-request switching** — select the adapter at inference time via the `model` field

!!! warning "Key YAML details (see `02-inferenceservice-lora.yaml`)"
    - `EXTRA_ARGS` is the env var used by the RHOAI vLLM ServingRuntime for additional CLI flags
    - `containers[].name` must be `kserve-container` to override the ServingRuntime default
    - `tolerations` go at the `predictor` level (not inside `model`)
    - The PVC `training-workspace` is the same one created during training in Step 3

### 4.3 Option B: Merge the adapter and deploy as a standalone model

If you prefer a single merged model (simpler deployment, slightly lower inference latency):

**Step 4B.1:** Merge LoRA weights into the base model (run in a workbench or notebook with GPU):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
model = PeftModel.from_pretrained(base, "/workspace/output")
merged = model.merge_and_unload()

merged.save_pretrained("/workspace/merged-model")
AutoTokenizer.from_pretrained("Qwen/Qwen3-4B").save_pretrained("/workspace/merged-model")
```

**Step 4B.2:** Upload to S3:

```bash
aws s3 sync /workspace/merged-model s3://your-bucket/models/financial-agent-merged/
```

**Step 4B.3:** Create the data connection and deploy:

```bash
# Edit 04-s3-data-connection.yaml with your S3 credentials, then apply
oc apply -f serving/04-s3-data-connection.yaml -n financial-agent

# Edit storage.path in 03-inferenceservice-merged.yaml to match your S3 path
oc apply -f serving/03-inferenceservice-merged.yaml -n financial-agent

# Expose externally (update service name in 05-route.yaml to financial-agent-merged-predictor)
oc apply -f serving/05-route.yaml -n financial-agent
```

**Step 4B.4:** Monitor and verify using the same commands from Option A (replace `financial-agent-lora` with `financial-agent-merged`).

## Step 5: Evaluate

**RHOAI Feature:** Agent Evaluation (GA)

```bash
python 05_evaluate_agent.py \
  --model-endpoint http://financial-agent-lora-predictor.financial-agent.svc.cluster.local:8080 \
  --output evaluation_results.json
```

The evaluation script measures:

- **Tool recall** — Did the model call the correct tools?
- **Tool precision** — Did the model avoid calling unnecessary tools?
- **Argument correctness** — Were function arguments well-formed and accurate?
- **Order match** — Did the model chain tools in the correct sequence?
- **LLM-as-judge scoring** — An LLM judges task fulfillment, grounding, tool appropriateness, parameter accuracy, dependency awareness, and parallelism efficiency on a 1-10 scale

## Step 6: Configure Guardrails

**RHOAI Feature:** NeMo Guardrails (GA) + MCP Gateway (3.5 TP)

```bash
python 06_configure_guardrails.py \
  --model-endpoint https://financial-agent.apps.cluster.example.com/v1
```

- **Tier 1 (GA):** PII detection, jailbreak protection, financial disclaimers
- **Tier 2 (3.5 TP):** MCP Gateway auto-enforcement on all tool calls

## Step 7: Run the Deep Agent Harness (Optional)

!!! warning "Not yet validated on RHOAI"
    This step is optional and has not been validated on-cluster. The `deepagents` library has known compatibility issues with vLLM-served Qwen models (specifically `tool_call_id` handling). The model customization pipeline is complete at Step 6 — Steps 1-6 produce a fine-tuned, served, and guarded model ready for integration into any agent framework.

```bash
pip install deepagents langchain-openai langgraph-cli[inmem]
export MODEL_ENDPOINT=https://financial-agent-predictor.apps.your-cluster.com/v1

cd end-to-end-examples/financial-agent/examples/
langgraph dev

# Or headless:
python 07_deep_agent.py "What is the risk-adjusted performance of portfolio PORT-0001?"
```

The Deep Agent wraps the fine-tuned Qwen3-4B with task planning, tool orchestration, and long-term memory via the `deepagents` library. Alternatively, use any OpenAI-compatible agent framework (LangChain, CrewAI, etc.) since the model exposes a standard `/v1/chat/completions` endpoint with tool-calling support.

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

- [Source Code (Financial Example)](https://github.com/rrbanda/rhoai/tree/main/end-to-end-examples/financial-agent)

## Related

- [Financial Domain Guide](../domains/financial.md)
- [Training Hub LoRA docs](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/docs/algorithms/lora.md)
- [LoRA KFP Pipeline](https://github.com/red-hat-data-services/pipelines-components/tree/main/pipelines/training/finetuning/lora)
- [Serving](../serving/index.md)
- [Guardrails](../guardrails/index.md)
