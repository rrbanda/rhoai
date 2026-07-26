# Serving Fine-Tuned Models

After training, deploy your model on RHOAI for inference using KServe with the vLLM runtime. This section covers deployment, tool-calling configuration, and integration with guardrails.

## Serving Options on RHOAI

| Mode | Use Case | Min GPU | Scale-to-Zero |
|------|----------|---------|---------------|
| **KServe RawDeployment** | Production serving with predictable load | 1x L4 24GB+ (model-dependent) | No |
| **KServe Serverless (Knative)** | Burst workloads, cost optimization | 1x L4 24GB+ (model-dependent) | Yes |
| **Distributed Inference (llm-d)** | Large models across multiple GPUs/nodes | 4x+ GPUs | No |
| **Models-as-a-Service** | Managed model endpoints | N/A | N/A |

## Deploy with KServe RawDeployment

RawDeployment is the recommended mode for fine-tuned models. It provides stable, always-on endpoints with direct GPU access.

### Step 1: Create a ServingRuntime

The ServingRuntime defines the vLLM container image, command-line flags, and volume mounts. For LoRA adapter serving, the runtime includes the PVC mount and `--enable-lora` flag. For tool-calling models, it also includes `--enable-auto-tool-choice` and `--tool-call-parser`.

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
  name: vllm-lora-runtime
  labels:
    opendatahub.io/dashboard: "true"
spec:
  multiModel: false
  supportedModelFormats:
    - name: vLLM
      autoSelect: true
  containers:
    - name: kserve-container
      image: registry.redhat.io/rhaii/vllm-cuda-rhel9@sha256:5800e12b2a465f15961fcf34b645d79ed4f91ec9161eab22b1205d12682183c8
      command: ["python", "-m", "vllm.entrypoints.openai.api_server"]
      args:
        - --port=8080
        - --model=/mnt/models
        - --served-model-name={{.Name}}
        - --max-model-len=4096
        - --enable-lora
        - --lora-modules
        - my-adapter=/mnt/lora-adapter/output
        - --max-lora-rank=16
        - --gpu-memory-utilization=0.90
      env:
        - name: HF_HUB_CACHE
          value: /tmp/hf_cache
      ports:
        - containerPort: 8080
          protocol: TCP
      volumeMounts:
        - name: lora-adapter
          mountPath: /mnt/lora-adapter
          readOnly: true
        - name: shm
          mountPath: /dev/shm
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
  volumes:
    - name: lora-adapter
      persistentVolumeClaim:
        claimName: training-workspace
    - name: shm
      emptyDir:
        medium: Memory
        sizeLimit: 4Gi
```

!!! info "Customize the runtime for your use case"
    - Change `my-adapter` in `--lora-modules` to your model name
    - Change `training-workspace` to your PVC name
    - Remove `--enable-lora` and related flags if serving a fully merged model
    - Remove `--enable-auto-tool-choice` and `--tool-call-parser` if not using tool-calling

### Step 2: Create the InferenceService

The InferenceService specifies only the base model, resource requirements, and which runtime to use. LoRA adapter mounts, tool-calling flags, and other vLLM configuration belong in the **ServingRuntime** (Step 1).

!!! warning "Do not mix `model` and `containers` in the predictor spec"
    KServe rejects InferenceService manifests that have both `model` and `containers` in the predictor. All container configuration (env vars, volume mounts, args) must go in the ServingRuntime.

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: my-finetuned-model
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
  labels:
    opendatahub.io/dashboard: "true"
spec:
  predictor:
    tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
    model:
      modelFormat:
        name: vLLM
      runtime: vllm-lora-runtime
      storageUri: hf://Qwen/Qwen3-4B
      resources:
        requests:
          cpu: "2"
          memory: "8Gi"
          nvidia.com/gpu: "1"
        limits:
          cpu: "3"
          memory: "10Gi"
          nvidia.com/gpu: "1"
```

### Deploy from Python

```python
from kubernetes import client, config

config.load_kube_config()
api = client.CustomObjectsApi()

manifest = {
    "apiVersion": "serving.kserve.io/v1beta1",
    "kind": "InferenceService",
    "metadata": {
        "name": "my-model",
        "annotations": {
            "serving.kserve.io/deploymentMode": "RawDeployment",
        },
        "labels": {
            "opendatahub.io/dashboard": "true",
        },
    },
    "spec": {
        "predictor": {
            "model": {
                "modelFormat": {"name": "vLLM"},
                "runtime": "vllm-lora-runtime",
                "storageUri": "hf://Qwen/Qwen3-4B",
                "resources": {
                    "requests": {"cpu": "2", "memory": "8Gi", "nvidia.com/gpu": "1"},
                    "limits": {"cpu": "3", "memory": "10Gi", "nvidia.com/gpu": "1"},
                },
            },
        },
    },
}

api.create_namespaced_custom_object(
    group="serving.kserve.io",
    version="v1beta1",
    namespace="my-namespace",
    plural="inferenceservices",
    body=manifest,
)
```

### Storage Options

=== "S3 / Object Storage"

    ```yaml
    metadata:
      annotations:
        serving.kserve.io/secretName: aws-connection-models
    spec:
      predictor:
        model:
          storageUri: s3://my-bucket/models/my-model
    ```

    Create the S3 secret via the RHOAI dashboard (Data Connections) or manually:

    ```yaml
    apiVersion: v1
    kind: Secret
    metadata:
      name: aws-connection-models
    type: Opaque
    stringData:
      AWS_ACCESS_KEY_ID: "..."
      AWS_SECRET_ACCESS_KEY: "..."
      AWS_S3_ENDPOINT: "https://s3.amazonaws.com"
      AWS_DEFAULT_REGION: "us-east-1"
      AWS_S3_BUCKET: "my-bucket"
    ```

=== "HuggingFace (base model)"

    ```yaml
    spec:
      predictor:
        model:
          storageUri: hf://Qwen/Qwen3-4B
    ```

=== "PVC (Persistent Volume)"

    ```yaml
    spec:
      predictor:
        model:
          storageUri: pvc://model-storage/my-model
    ```

### Verify Deployment

```bash
# Check status
oc get inferenceservice my-model -n my-namespace

# Get the endpoint URL
oc get inferenceservice my-model -n my-namespace -o jsonpath='{.status.url}'

# Test with a simple request
curl -X POST "$ENDPOINT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello, how can you help me?"}],
    "max_tokens": 128
  }'
```

## Tool-Calling Configuration

If your model was trained for tool use (via [LoRA SFT](../training/lora.md) on MCP distillation traces, or [GRPO](../training/grpo.md)), include the tool-calling flags in the **ServingRuntime** `args`:

```yaml
args:
  - --enable-auto-tool-choice
  - --tool-call-parser=hermes
```

These flags are already included in the `vllm-lora-runtime` example above. The validated configuration for RHOAI 3.4.2 uses the `hermes` parser, which works with Qwen, Granite, and Llama model families.

| vLLM Argument | Purpose | Values |
|---------------|---------|--------|
| `--enable-auto-tool-choice` | Let the model decide when to call tools | Flag (no value) |
| `--tool-call-parser` | Parser for structured tool-call output | `hermes` (Qwen, Granite, Llama), `mistral`, `llama3` |
| `--chat-template` | Custom chat template with tool definitions | Path to Jinja2 template |

!!! info "RHOAI 3.5 Technology Preview"
    RHOAI 3.5 EA2 introduces **Validated Tool-Calling Configuration** — pre-validated vLLM argument combinations for supported model architectures. This ensures correct tool-call parsing without manual configuration.

### Test Tool Calling

```python
import requests

endpoint = "https://my-model.apps.cluster.example.com"

response = requests.post(
    f"{endpoint}/v1/chat/completions",
    json={
        "model": "my-model",
        "messages": [
            {"role": "user", "content": "What is the current price of AAPL?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_stock_quote",
                    "description": "Get a real-time stock quote.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"}
                        },
                        "required": ["ticker"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
    },
)

message = response.json()["choices"][0]["message"]
if message.get("tool_calls"):
    tc = message["tool_calls"][0]["function"]
    print(f"Tool: {tc['name']}, Args: {tc['arguments']}")
else:
    print(f"Text: {message['content']}")
```

## Multi-GPU / Distributed Serving

For models too large for a single GPU, add `--tensor-parallel-size` to the ServingRuntime `args` and increase the GPU resource requests in the InferenceService:

**ServingRuntime** (add to `args`):

```yaml
args:
  - --tensor-parallel-size=4
```

**InferenceService** (increase GPU count):

```yaml
spec:
  predictor:
    model:
      resources:
        requests:
          nvidia.com/gpu: "4"
        limits:
          nvidia.com/gpu: "4"
```

For cluster-scale distributed inference, see the [llm-d documentation](https://llm-d.ai).

## Related

- [Guardrails](../guardrails/index.md) — Add safety rails before exposing the endpoint
- [Tool-Calling Model Deployment](../end-to-end/tool-calling-financial.md#step-4-deploy-the-fine-tuned-model-on-rhoai) — Worked example with validated YAML
- [GPU Requirements](../reference/gpu-requirements.md) — Hardware planning for serving
