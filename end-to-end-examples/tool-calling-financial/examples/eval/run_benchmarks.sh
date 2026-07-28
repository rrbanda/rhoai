#!/usr/bin/env bash
# Reproducible benchmark runner for the tool-calling financial model.
# Runs both generic (tool-eval-bench) and domain-specific evaluations.
#
# Prerequisites:
#   uv tool install 'tool-eval-bench[perf] @ git+https://github.com/SeraphimSerapis/tool-eval-bench.git'
#   pip install httpx
#   oc login <cluster>
#
# Usage:
#   ./eval/run_benchmarks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$SCRIPT_DIR"
EXAMPLES_DIR="$(dirname "$SCRIPT_DIR")"

ROUTE_HOST=$(oc get route financial-agent-model -n financial-agent -o jsonpath='{.spec.host}' 2>/dev/null || echo "")
if [[ -z "$ROUTE_HOST" ]]; then
    echo "ERROR: Could not find model route. Are you logged into OpenShift?"
    echo "  Try: oc login <cluster-url>"
    exit 1
fi

ENDPOINT="https://${ROUTE_HOST}"
BASE_MODEL="financial-agent-lora"
LORA_MODEL="financial-agent"

echo "=================================================="
echo "  Tool-Calling Model Benchmark Suite"
echo "=================================================="
echo "  Endpoint: ${ENDPOINT}"
echo "  Base model: ${BASE_MODEL}"
echo "  LoRA model: ${LORA_MODEL}"
echo "  Output dir: ${EVAL_DIR}"
echo "=================================================="
echo ""

# Verify endpoint
echo "[1/4] Probing endpoint..."
curl -sk "${ENDPOINT}/v1/models" | python3 -m json.tool > /dev/null 2>&1 || {
    echo "ERROR: Endpoint not reachable at ${ENDPOINT}/v1/models"
    exit 1
}
echo "  OK - endpoint is live"
echo ""

# Generic benchmark
echo "[2/4] Running generic benchmark (tool-eval-bench) on base model..."
echo "  This takes ~20 minutes per model (84 scenarios x 3 trials)."
tool-eval-bench --seed 42 --hardmode --trials 3 \
  --model "$BASE_MODEL" --backend vllm \
  --base-url "$ENDPOINT" --no-think \
  --json-file "${EVAL_DIR}/base-model-results.json" \
  --output-dir "$EVAL_DIR"
echo ""

echo "[3/4] Running generic benchmark (tool-eval-bench) on LoRA model..."
tool-eval-bench --seed 42 --hardmode --trials 3 \
  --model "$LORA_MODEL" --backend vllm \
  --base-url "$ENDPOINT" --no-think \
  --json-file "${EVAL_DIR}/lora-model-results.json" \
  --output-dir "$EVAL_DIR"
echo ""

# Domain-specific benchmark
echo "[4/4] Running domain-specific financial tools evaluation..."
python3 "${EVAL_DIR}/domain_eval.py" \
  --endpoint "$ENDPOINT" \
  --base-model "$BASE_MODEL" \
  --lora-model "$LORA_MODEL" \
  --output "${EVAL_DIR}/domain-eval-results.json"
echo ""

echo "=================================================="
echo "  BENCHMARKS COMPLETE"
echo "=================================================="
echo "  Results:"
echo "    ${EVAL_DIR}/base-model-results.json"
echo "    ${EVAL_DIR}/lora-model-results.json"
echo "    ${EVAL_DIR}/domain-eval-results.json"
echo ""
echo "  Compare with: tool-eval-bench --history"
echo "=================================================="
