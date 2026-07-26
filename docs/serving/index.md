# Serving Fine-Tuned Models

After training, deploy your model on RHOAI for inference using KServe with the vLLM runtime. This section covers deployment, tool-calling configuration, and integration with guardrails.

## Serving Options on RHOAI

| Mode | Use Case | Min GPU | Scale-to-Zero |
|------|----------|---------|---------------|
| **KServe RawDeployment** | Production serving with predictable load | 1x A100 40GB+ | No |
| **KServe Serverless (Knative)** | Burst workloads, cost optimization | 1x A100 40GB+ | Yes |
| **Distributed Inference (llm-d)** | Large models across multiple GPUs/nodes | 4x+ GPUs | No |
| **Models-as-a-Service** | Managed model endpoints | N/A | N/A |

## Deploy with KServe RawDeployment

RawDeployment is the recommended mode for fine-tuned models. It provides stable, always-on endpoints with direct GPU access.

### Minimal Deployment

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: my-finetuned-model
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    model:
      modelFormat:
        name: vLLM
      runtime: vllm-runtime
      storageUri: s3://my-bucket/models/my-finetuned-model
      resources:
        requests:
          nvidia.com/gpu: "1"
        limits:
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
    },
    "spec": {
        "predictor": {
            "model": {
                "modelFormat": {"name": "vLLM"},
                "runtime": "vllm-runtime",
                "storageUri": "s3://my-bucket/models/my-model",
                "resources": {
                    "requests": {"nvidia.com/gpu": "1"},
                    "limits": {"nvidia.com/gpu": "1"},
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

If your model was trained for tool use (via [LoRA SFT](../training/lora.md) on MCP distillation traces, or [GRPO](../training/grpo.md)), enable tool-calling in vLLM:

```yaml
spec:
  predictor:
    containers:
      - name: kserve-container
        env:
          - name: EXTRA_ARGS
            value: "--enable-auto-tool-choice --tool-call-parser hermes"
```

!!! warning "RHOAI-specific env var"
    The RHOAI vLLM ServingRuntime uses `EXTRA_ARGS` (not `VLLM_ARGS`) to pass additional CLI flags to vLLM. The container name must be `kserve-container` to override the default container in the ServingRuntime.

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

For models too large for a single GPU, use tensor parallelism:

```yaml
spec:
  predictor:
    model:
      resources:
        requests:
          nvidia.com/gpu: "4"
        limits:
          nvidia.com/gpu: "4"
    containers:
      - name: kserve-container
        env:
          - name: EXTRA_ARGS
            value: "--tensor-parallel-size 4"
```

For cluster-scale distributed inference, see the [llm-d documentation](https://llm-d.ai).

## Related

- [Guardrails](../guardrails/index.md) — Add safety rails before exposing the endpoint
- [Tool-Calling Model Deployment](../end-to-end/tool-calling-financial.md#step-4-deploy-the-fine-tuned-model-on-rhoai) — Worked example with financial services
- [GPU Requirements](../reference/gpu-requirements.md) — Hardware planning for serving
