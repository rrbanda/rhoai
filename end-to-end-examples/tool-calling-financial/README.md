# Tool-Calling Model Customization on RHOAI (Financial Example)

Fine-tune a small language model (Qwen3-4B) to make accurate tool calls for financial services. Uses SDG Hub's MCP distillation to generate domain-specific training data from the financial MCP server, then Training Hub's LoRA SFT to fine-tune the model on tool-calling demonstrations. The fine-tuned model is deployed on RHOAI with vLLM tool-calling support behind NeMo Guardrails. The same technique applies to any domain with any MCP server.

## RHOAI Feature Matrix

| Feature | RHOAI Version | Status | Purpose |
|---------|--------------|--------|---------|
| SDG Hub MCP Distillation | 3.4+ | GA | Generate tool-call training data from MCP servers |
| Training Hub LoRA SFT | 3.4+ | GA | Fine-tune model on tool-calling demonstrations |
| LoRA KFP Pipeline | 3.4+ | GA | End-to-end training pipeline (data → train → eval → registry) |
| KServe RawDeployment | 3.4+ | GA | Deploy fine-tuned model with vLLM runtime |
| NeMo Guardrails | 3.4+ | GA | Financial compliance rails, PII detection, disclaimers |
| Tool-Use Evaluation | 3.4+ | GA | Tool-calling quality metrics + LLM-as-judge |
| NeMo + MCP Gateway | 3.5 EA2 | **TP** | Auto-enforce guardrails on all agent tool calls |
| Validated Tool-Calling Config | 3.5 EA2 | **TP** | Pre-validated vLLM args for tool-calling models |

**Legend:** GA = Generally Available | TP = Technology Preview

## Architecture

![tool-calling financial model Architecture](docs/architecture.png)

## What You Will Build

A small, efficient language model that acts as an expert financial services agent — capable of calling portfolio management tools, querying market data, performing risk analysis, and executing trades with compliance checks. The base model (Qwen3-4B) starts with general language ability, but after fine-tuning on financial tool-calling demonstrations it learns *which* financial tool to call for a given query, how to format arguments correctly, and how to chain tools together for complex workflows.

Two training paths are provided:
- **03_train_lora_sft.py** — Run training locally via `training_hub.lora_sft()` for development and prototyping
- **03b_train_kfp_pipeline.py** — Run the official LoRA KFP pipeline on RHOAI with automatic dataset download, training, evaluation, and model registry

Both use the same algorithm (LoRA + SFT via Unsloth backend) and produce identical results. The KFP pipeline is the recommended production path because it runs entirely on-cluster and includes evaluation and model registry stages.

## Prerequisites

- RHOAI 3.4+ cluster (3.5 EA2 for MCP Gateway guardrails)
- GPU nodes: 1x NVIDIA L4 24GB (with QLoRA 4-bit) or 1x L40/A100 for full-precision
- Teacher model API key (Gemini 3.6 Flash recommended for cost, or GPT-4o) for data generation
- Langflow instance with a frontier model agent connected to the financial MCP server
- Python 3.11+
- `oc` CLI authenticated to your OpenShift cluster

**For the KFP pipeline (step 03b):**
- Pipeline server running in your Data Science Project
- Storage class with `ReadWriteOnce` support (default: `gp3-csi`)
- `kubernetes-credentials` secret with cluster API access

## Directory Structure

```
tool-calling-financial/
├── README.md                               This guide
├── demo_server/
│   ├── server.py                           FastMCP financial server (15 tools)
│   ├── data.py                             Deterministic seed data
│   └── README.md                           Server quick-start
├── serving/
│   ├── 01-serving-runtime.yaml             vLLM ServingRuntime (if not preinstalled)
│   ├── 02-inferenceservice-lora.yaml       Deploy with LoRA adapter from PVC
│   ├── 03-inferenceservice-merged.yaml     Deploy merged model from S3
│   ├── 04-s3-data-connection.yaml          S3 credentials secret
│   └── 05-route.yaml                       OpenShift Route for external access
├── guardrails/
│   ├── config.yml                          NeMo Guardrails configuration
│   ├── config.co                           Colang 2.0 financial compliance flows
│   ├── nemoguardrails_configmap.yaml       NeMo Guardrails ConfigMap
│   ├── nemoguardrails_cr.yaml              NemoGuardrails CR (GA)
│   └── mcpgateway_extension_cr.yaml        MCPGatewayExtension CR (3.5 TP)
└── examples/
    ├── .env.example                        Environment variable template
    ├── requirements.txt                    Python dependencies
    ├── langgraph.json                      LangGraph Dev Server config
    ├── 01_generate_tool_data.py            SDG Hub MCP distillation
    ├── 02_format_training_data.py          Format distillation output to JSONL
    ├── 03_train_lora_sft.py               LoRA SFT training (local, for dev)
    ├── 03b_train_kfp_pipeline.py          LoRA KFP pipeline (RHOAI production)
    ├── 04_deploy_model.py                  KServe RawDeployment + vLLM
    ├── 05_evaluate_agent.py                Tool-calling model evaluation
    ├── 06_configure_guardrails.py          NeMo Guardrails + MCP Gateway
    ├── 07_deep_agent.py                    Deep Agent harness (LangGraph)
    ├── financial_tools.py                  15 @tool wrappers for MCP server
    ├── financial_prompts.py                System prompt for Financial Insights Agent
    ├── pyproject.toml                      Agent dependencies (deepagents>=0.6.12)
    ├── AGENTS.md                           Agent identity and tool-usage guidelines
    ├── skills/
    │   ├── portfolio-analysis/SKILL.md     Multi-step portfolio analysis workflow
    │   ├── market-research/SKILL.md        Stock screening and market overview
    │   └── trade-evaluation/SKILL.md       Compliance check + order submission
    └── pipelines-components/               Auto-cloned on first KFP run (Step 3C)
```

## Step-by-Step Guide

### Step 0: Start the Financial MCP Server

First, create the namespace where all resources will be deployed:

```bash
oc new-project tool-calling-financial
```

Then start the server:

```bash
cd demo_server/
pip install fastmcp
python server.py
# Server starts on http://localhost:8009
```

The demo server exposes 15 tools organized into financial domains: portfolio management, market data, risk analysis, and trade execution.

### Step 1: Generate Tool-Call Training Data

**RHOAI Feature:** SDG Hub MCP Distillation (GA)

```bash
cd examples/
cp .env.example .env
# Edit .env with your teacher model API key and Langflow URL

python 01_generate_tool_data.py --num-samples 10
```

This runs the MCP distillation pipeline:
- A frontier model (via Langflow) explores all 15 financial tools
- A teacher LLM synthesizes realistic financial questions
- The frontier model solves each question via actual MCP tool calls (producing expert trajectories)
- Quality filters remove incomplete or low-quality results
- Output: `generated_data/distillation_output.parquet`

### Step 2: Format Training Data

```bash
python 02_format_training_data.py
```

Converts raw tool traces into chat-format JSONL for supervised fine-tuning:

```json
{"messages": [
  {"role": "system", "content": "<tool declarations>"},
  {"role": "user", "content": "What's the risk-adjusted return on my tech portfolio?"},
  {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_portfolio_positions", "arguments": "{\"portfolio_id\": \"PORT-0001\"}"}}]},
  {"role": "tool", "content": "{...}", "name": "get_portfolio_positions"},
  {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "calculate_portfolio_risk", "arguments": "{\"portfolio_id\": \"PORT-0001\"}"}}]},
  {"role": "tool", "content": "{...}", "name": "calculate_portfolio_risk"},
  {"role": "assistant", "content": "Your tech portfolio has a Sharpe ratio of..."}
]}
```

Output: `generated_data/training_data.jsonl`

### Step 3: Fine-Tune with LoRA SFT

**RHOAI Feature:** Training Hub LoRA SFT (GA) + Kubeflow Trainer (GA)

Three training paths are available. **Option B (TrainJob)** is the recommended approach — it runs directly on-cluster and was validated end-to-end on RHOAI 3.4.2.

#### Option A: Local Training (for development)

```bash
# Single L4 24GB with QLoRA (default settings)
python 03_train_lora_sft.py

# A100 80GB with higher capacity
python 03_train_lora_sft.py --lora-r 32 --lora-alpha 64 --max-seq-len 8192 --no-4bit

# Multi-GPU data-parallel
torchrun --nproc-per-node=2 03_train_lora_sft.py
```

#### Option B: Direct TrainJob on RHOAI (recommended, validated)

This creates a Kubeflow `TrainJob` CR using the `training-hub` ClusterTrainingRuntime. It runs directly on the GPU node — no pipeline server needed.

**Prerequisite:** `trainer` enabled in the DataScienceCluster and the `training-hub` ClusterTrainingRuntime exists:

```bash
oc get clustertrainingruntimes training-hub
```

> **Validation vs production data:** The ConfigMap below uses a small public dataset (`LipengCS/Table-GPT`) to validate the training infrastructure. For production, upload your Step 2 JSONL to the PVC first and the script will use it automatically (it checks for existing data before downloading).

**Step 3B.1:** Create workspace PVC and training script ConfigMap:

```bash
cat <<'YAML' | oc apply -n tool-calling-financial -f -
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

    Uses a public HuggingFace dataset for infrastructure validation.
    For production, pre-upload your Step 2 JSONL to the PVC at
    /workspace/data/training_data.jsonl and this script will use it.
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
        print("Downloading validation dataset from HuggingFace...")
        print("  (Pre-upload your Step 2 JSONL for production)")
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
    print("Running LoRA SFT training...")
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

**Step 3B.2:** Create the TrainJob:

```bash
cat <<'YAML' | oc apply -n tool-calling-financial -f -
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: tool-calling-financial-lora-sft
spec:
  runtimeRef:
    name: training-hub
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

> **Important:** The `targetJobs` name and container name must both be `node` (not `Trainer` or `Initializer`) — this matches the `training-hub` ClusterTrainingRuntime definition. GPU tolerations are required if nodes have `nvidia.com/gpu=True:NoSchedule` taint.

**Step 3B.3:** Monitor training:

```bash
# Watch the pod start
oc get pods -n tool-calling-financial -l job-name=tool-calling-financial-lora-sft-node-0 -w

# Stream logs once Running
oc logs -f -l job-name=tool-calling-financial-lora-sft-node-0 -n tool-calling-financial
```

Expected output (validated on RHOAI 3.4.2 with L4 GPU):

```
Step 1: Downloading dataset from HuggingFace...
  Saved 50 examples to /workspace/data/training_data.jsonl
Step 2: Running LoRA SFT training...
🦥 Unsloth 2026.4.5 patched 36 layers with 36 QLoRA layers
Trainable parameters = 16,544,768 / 4,025,356,288 (0.41%)
{'loss': 1.817, 'learning_rate': 2e-04, 'epoch': 0.5}
{'loss': 1.511, 'learning_rate': 0.0, 'epoch': 1.0}
{'train_runtime': 38.5, 'train_samples_per_second': 1.3}
Training complete!
  Output saved to: /workspace/output
```

**Step 3B.4:** Verify and retrieve:

```bash
oc get trainjob tool-calling-financial-lora-sft -n tool-calling-financial
# The LoRA adapter is saved in the PVC at /workspace/output
```

#### Option C: LoRA KFP Pipeline on RHOAI (for multi-stage automation)

**Source:** [pipelines-components/pipelines/training/finetuning/lora](https://github.com/red-hat-data-services/pipelines-components/tree/main/pipelines/training/finetuning/lora)

The KFP pipeline wraps the same LoRA SFT algorithm in a four-stage automated pipeline. Use this when you need repeatable dataset download, evaluation, and model registry stages.

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
3. **Evaluation** — LM-Eval harness benchmarks (arc_easy by default, checks for capability regression)
4. **Model Registry** — registers the fine-tuned model (optional)

**KFP Pipeline Prerequisites:**

| Requirement | Details |
|-------------|---------|
| RHOAI components | `dashboard`, `trainer`, `aipipelines` enabled |
| Argo Workflows | `argoWorkflowsControllers` set to `Managed` in DSC |
| Pipeline server | DSPA running in your namespace |
| Storage class | RWO with `gp3-csi` (or configure `--storage-class`) |
| Secret | `kubernetes-credentials` with `KUBERNETES_SERVER_URL` and `KUBERNETES_AUTH_TOKEN` |
| KFP SDK | `kfp==2.15.2` (for pipeline YAML compilation) |

Create the required secret:

```bash
oc create secret generic kubernetes-credentials \
  --from-literal=KUBERNETES_SERVER_URL="https://api.your-cluster.com:6443" \
  --from-literal=KUBERNETES_AUTH_TOKEN="$(oc whoami -t)"
```

### Step 4: Deploy the Fine-Tuned Model on RHOAI

**RHOAI Feature:** KServe RawDeployment (GA) + vLLM ServingRuntime (GA)

After training, the LoRA adapter is on the PVC at `/workspace/output`. Deploy using the YAML manifests in the `serving/` directory:

```
serving/
├── 01-serving-runtime.yaml          vLLM ServingRuntime (only if not preinstalled)
├── 02-inferenceservice-lora.yaml    Option A: LoRA adapter from PVC (recommended)
├── 03-inferenceservice-merged.yaml  Option B: Merged model from S3
├── 04-s3-data-connection.yaml       S3 credentials secret (Option B only)
└── 05-route.yaml                    OpenShift Route for external access
```

#### 4.1 Prerequisites — verify the serving stack

```bash
# Confirm KServe is enabled
oc get dsc default-dsc -o jsonpath='{.spec.components.kserve.managementState}' && echo
# Expected: Managed

# Confirm the vLLM ServingRuntime is available
oc get servingruntimes -n redhat-ods-applications | grep vllm
# Expected: vllm-runtime-...
```

> If the vLLM runtime is not listed, enable it via **RHOAI Dashboard → Settings → Serving runtimes → vLLM ServingRuntime for KServe**, or apply the provided manifest: `oc apply -f serving/01-serving-runtime.yaml -n tool-calling-financial`

#### 4.2 Option A: Serve the LoRA adapter directly (recommended)

vLLM natively supports LoRA adapters without merging. The base model (Qwen3-4B) is pulled from HuggingFace and the adapter (~50MB) is mounted from the training PVC.

**Deploy:**

```bash
# Apply the InferenceService
oc apply -f serving/02-inferenceservice-lora.yaml -n tool-calling-financial

# Expose the endpoint externally
oc apply -f serving/05-route.yaml -n tool-calling-financial
```

**Monitor readiness:**

```bash
# Watch until READY=True (typically 2-5 minutes for initial model download)
oc get inferenceservice tool-calling-financial-lora -n tool-calling-financial -w

# Check predictor pod
oc get pods -n tool-calling-financial -l serving.kserve.io/inferenceservice=tool-calling-financial-lora

# Stream vLLM startup logs
oc logs -f deployment/tool-calling-financial-lora-predictor -n tool-calling-financial
```

Model is ready when logs show:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080
```

**Get the inference endpoint:**

```bash
ROUTE_URL=$(oc get route tool-calling-financial -n tool-calling-financial -o jsonpath='{.spec.host}')
echo "Inference endpoint: https://${ROUTE_URL}"
```

**Verify with a tool-calling request:**

```bash
curl -sk "https://${ROUTE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tool-calling-financial",
    "messages": [{"role": "user", "content": "What is the current price of AAPL?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_stock_quote",
        "description": "Get a real-time stock quote",
        "parameters": {
          "type": "object",
          "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol"}},
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

> **Why serve directly?** No merge step, adapter is ~50MB vs 8GB+ merged, multi-adapter support from a single base model, per-request adapter switching via the `model` field.

> **Key YAML details** (see `02-inferenceservice-lora.yaml`): `EXTRA_ARGS` is the env var used by the RHOAI vLLM ServingRuntime. Container name must be `kserve-container`. Tolerations go at the `predictor` level. The PVC `training-workspace` is the same one from Step 3.

#### 4.3 Option B: Merge and deploy as a standalone model

If you prefer a single merged model (simpler deployment, slightly lower inference latency):

**Step 4B.1:** Merge the adapter (run in a workbench or notebook with GPU):

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
aws s3 sync /workspace/merged-model s3://your-bucket/models/tool-calling-financial-merged/
```

**Step 4B.3:** Create the data connection and deploy:

```bash
# Edit 04-s3-data-connection.yaml with your S3 credentials
oc apply -f serving/04-s3-data-connection.yaml -n tool-calling-financial

# Edit storage.path in 03-inferenceservice-merged.yaml to match your S3 path
oc apply -f serving/03-inferenceservice-merged.yaml -n tool-calling-financial

# Expose externally (update service name in 05-route.yaml to tool-calling-financial-merged-predictor)
oc apply -f serving/05-route.yaml -n tool-calling-financial
```

**Step 4B.4:** Monitor and verify using the same commands from Option A (replace `tool-calling-financial-lora` with `tool-calling-financial-merged`).

### Step 5: Evaluate

**RHOAI Feature:** Tool-Use Evaluation (GA)

```bash
python 05_evaluate_agent.py \
  --model-endpoint http://tool-calling-financial-lora-predictor.tool-calling-financial.svc.cluster.local:8080 \
  --output evaluation_results.json
```

Runs:
- **Tool recall / precision** — correct tool selection for each query
- **Argument correctness** — valid and complete function arguments
- **Order match** — correct tool-chaining sequence for multi-step queries
- **LLM-as-judge** — GPT-4o scores task fulfillment, grounding, tool appropriateness, parameter accuracy, dependency awareness, and parallelism on a 1-10 scale

### Step 6: Configure Guardrails

**RHOAI Feature:** NeMo Guardrails (GA)

```bash
python 06_configure_guardrails.py
```

**Tier 1 — NeMo Guardrails (GA, RHOAI 3.4+):**
- PII detection and masking on inputs/outputs
- Financial disclaimer injection on investment advice
- Jailbreak detection to prevent prompt injection

**Tier 2 — MCP Gateway Integration (TP, RHOAI 3.5):**
- Automatic guardrail enforcement on all tool calls routed through the MCP Gateway
- Policy-based tool access control

### Step 7: Deep Agent Harness (Runtime-Validated)

Wire the fine-tuned model into an autonomous agent using [LangChain Deep Agents](https://github.com/langchain-ai/deepagents). The agent adds task planning, multi-tool orchestration, persistent memory, and on-demand skills on top of the model's tool-calling ability.

> **Important:** Deep Agents requires `--max-model-len=16384` or higher on the vLLM deployment. The middleware stack adds ~3,500 tokens of system prompt.

```bash
# Increase vLLM context window (one-time)
oc patch servingruntime vllm-lora-runtime -n tool-calling-financial \
  --type='json' \
  -p='[{"op":"replace","path":"/spec/containers/0/args/3","value":"--max-model-len=16384"}]'

# Install agent dependencies
cd examples/
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .

# Create .env (do NOT commit — already in .gitignore)
cat > .env << 'EOF'
MODEL_ENDPOINT=https://<your-model-route>/v1
MODEL_NAME=tool-calling-financial
MCP_SERVER_URL=http://localhost:8009/mcp
OPENAI_API_KEY=not-needed
EOF

# Get the model Route URL for your .env
oc get route -n tool-calling-financial -l serving.kserve.io/inferenceservice \
  -o jsonpath='{.items[0].spec.host}'

# Start MCP server (separate terminal)
cd ../demo_server && source ../examples/.venv/bin/activate && python server.py

# Run the agent
python 07_deep_agent.py "What is the current price of AAPL?"
python 07_deep_agent.py "What is the risk-adjusted performance of PORT-0001?"

# Or launch the interactive LangGraph Studio UI
langgraph dev --allow-blocking
```

The agent was validated with three query types:
- **Single-tool:** `get_stock_quote(AAPL)` — correct tool selection and response synthesis
- **Multi-tool chaining:** `get_portfolio_performance` + `calculate_portfolio_risk` called in parallel
- **Compliance workflow:** `check_compliance` with proper risk tolerance reporting

The model also exposes a standard `/v1/chat/completions` endpoint, so you can integrate it with any OpenAI-compatible agent framework (LangChain, CrewAI, etc.).

## Resource Estimates

| Phase | Resource | Estimated Time |
|-------|----------|---------------|
| Data Generation | 1x CPU + Teacher API | 2-4 hours |
| Training (local, L4) | 1x L4 24GB | 1-3 hours |
| Training (KFP, L40) | 1x L40 48GB | 1-2 hours |
| Serving | 1x L4 24GB+ | Ongoing |
| Evaluation | 1x GPU + Model endpoint | 30-60 min |

## Validated on RHOAI

This pipeline has been validated end-to-end on the following environment:

| Component | Version / Config |
|-----------|-----------------|
| RHOAI | 3.4.2 |
| OpenShift | 4.18.21 |
| GPU | 1x NVIDIA L4 24GB (`g6.xlarge`) |
| Training Runtime | `training-hub` ClusterTrainingRuntime |
| Training Backend | Unsloth 2026.4.5 + Training Hub |
| Base Model | Qwen/Qwen3-4B (4-bit QLoRA) |
| Teacher Model | Gemini 3.6 Flash (via litellm) |
| SDG Hub | MCP Server Distillation flow (22 blocks) |
| Pipeline Server | DSPA with MinIO + MariaDB |

**Validated steps:**

- MCP server: 15 tools callable via FastMCP
- SDG Hub: Flow loads, teacher model configured, input dataset builds correctly
- Data formatting: Parquet → chat-format JSONL with proper tool_calls structure
- Training: LoRA SFT via Kubeflow Trainer, loss 1.817→1.511, 38s for 50 examples
- KFP pipeline: Compiles to 2,367-line YAML, uploads to pipeline server
- Model deployment: KServe manifest generates correctly (dry-run validated)
- Guardrails: NemoGuardrails CR generates correctly (dry-run validated)

**Important cluster notes:**

- GPU nodes with taint `nvidia.com/gpu=True:NoSchedule` require a toleration in TrainJob `podTemplateOverrides`
- On `g6.xlarge` (3.5 CPU, 14GB RAM), use `cpu: 2, memory: 10Gi` for training pod requests
- Scale down the model predictor before training if the GPU node is shared (only 1 GPU)
- The KFP pipeline uses RWO storage (`gp3-csi`) by default — RWX is not required. Nodes need ≥16GB free ephemeral storage for the 7.5GB training image
- Enable `argoWorkflowsControllers` in the DataScienceCluster if the pipeline workflow stays pending

## Troubleshooting

| Symptom | Resolution |
|---------|------------|
| "MCP Server Distillation flow not found" | Ensure sdg-hub is installed: `pip install sdg-hub[examples]` |
| "CUDA out of memory during training" | Enable QLoRA: `--load-in-4bit` (default), or reduce `--max-seq-len` |
| "Tool call parser error" during inference | Verify `--tool-call-parser` matches model (use `hermes` for Qwen3) |
| "Guardrails CR not ready" | Check TrustyAI operator is installed |
| KFP pipeline: "storageclass not found" | Set `--storage-class` to your cluster's storage class (default: `gp3-csi`) |
| KFP pipeline: "kubernetes-credentials not found" | Create the secret (see Step 3 Option B prerequisites) |
| Langflow agent times out | Increase `LANGFLOW_TIMEOUT` or check MCP server reachability |
| TrainJob pod stays Pending (GPU taint) | Add `tolerations` for `nvidia.com/gpu` in `podTemplateOverrides` |
| TrainJob pod stays Pending (memory) | Reduce `memory` request; `g6.xlarge` has ~14GB allocatable |
| KFP workflow stays pending (no status) | Enable `argoWorkflowsControllers` in DSC: `oc patch dsc default-dsc --type=merge -p '{"spec":{"components":{"aipipelines":{"argoWorkflowsControllers":{"managementState":"Managed"}}}}}'` |
| KFP pod evicted (ephemeral storage) | Training images are 7-15GB; use nodes with >100GB ephemeral storage or use direct TrainJob |

## RHOAI 3.4 vs 3.5

**Works fully on RHOAI 3.4 (GA features only):**
- Steps 0-3: MCP server, data generation, formatting, LoRA SFT training
- Step 4: Model deployment with KServe RawDeployment + vLLM
- Step 5: Tool-calling evaluation (tool metrics + LLM-as-judge)
- Step 6 Tier 1: NeMo Guardrails for financial compliance

**RHOAI 3.5 EA2 adds (Technology Preview):**
- Step 4: Validated tool-calling config (pre-validated vLLM args per model)
- Step 6 Tier 2: MCP Gateway integration for automatic guardrail enforcement

## References

- [Training Hub LoRA docs](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/docs/algorithms/lora.md)
- [LoRA KFP Pipeline](https://github.com/red-hat-data-services/pipelines-components/tree/main/pipelines/training/finetuning/lora)
- [KFP Pipeline Guide](https://github.com/red-hat-data-services/red-hat-ai-examples/tree/main/examples/fine-tuning/pipelines/training-hub)
- [SDG Hub MCP Distillation](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/tree/main/examples/agentic/mcp_distillation_training)
- [NeMo Guardrails on RHOAI](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/ensuring_ai_safety_with_guardrails)
- [KServe RawDeployment](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.4/html/serving_models)
