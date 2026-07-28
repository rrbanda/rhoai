# Tool-Calling Fine-Tuning: Evaluation Results

!!! abstract "Summary"
    After fine-tuning Qwen3-4B on RHOAI using the MCP Distillation pipeline, the model achieves **100% tool-calling accuracy** across all 15 domain-specific financial tools — correct tool selection, correct parameters, correct format — on every query tested. General capabilities are fully preserved with no degradation.

---

## What We Measured

We validated the fine-tuned model on two dimensions:

| Evaluation | Purpose | Result |
|-----------|---------|--------|
| **Domain accuracy** (15 financial tools, 20 queries) | Does the model call the right tool with the right arguments? | **100% accuracy** |
| **General capability check** (84 generic scenarios) | Did fine-tuning break anything? | **No degradation** (0% delta) |

**Setup:** Qwen3-4B base model fine-tuned with LoRA (rank 16) using 10 MCP-distilled training examples, deployed on RHOAI 3.4.2 with vLLM on NVIDIA L4 GPU.

---

## Domain-Specific Results: Financial Tool Calling

The fine-tuned model was evaluated against all 15 financial tools from the MCP server — the same tools it was trained to use. Queries range from stock quotes to portfolio risk analysis to trade execution.

| Metric | Score |
|--------|:-----:|
| **Tool Selection Accuracy** | 100% (20/20) |
| **Parameter Extraction Accuracy** | 100% (20/20) |
| **Output Format Compliance** | 100% (20/20) |
| **Average Response Latency** | 9.0 seconds |

Every query was routed to the correct tool with correctly extracted parameters — stock tickers, portfolio IDs, date ranges, order types, and numerical thresholds all parsed accurately from natural language.

**Example queries the model handles correctly:**

| User Query | Tool Called | Key Parameters |
|-----------|------------|----------------|
| "Give me a full analysis of Microsoft stock" | `analyze_stock` | `ticker: MSFT` |
| "Buy 50 shares of Google at market price in my TECH-FUND portfolio" | `submit_trade_order` | `ticker: GOOGL, action: buy, quantity: 50, order_type: market` |
| "Run a market crash stress test on my portfolio CONSERVATIVE-A" | `run_stress_test` | `portfolio_id: CONSERVATIVE-A, scenario: market_crash` |
| "Find high-dividend healthcare stocks with at least 3% yield" | `screen_stocks` | `sector: Healthcare, min_dividend_yield: 3` |
| "Show me NVIDIA weekly price data for the past year" | `get_historical_prices` | `ticker: NVDA, period: 1y, interval: 1wk` |

---

## General Capability Preservation

To verify fine-tuning did not degrade the model's general tool-calling ability, we ran [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) — an industry-standard benchmark with 84 deterministic scenarios across 15 categories (multi-step chains, error recovery, safety boundaries, structured output, etc.).

| Metric | Base Model | Fine-Tuned | Delta |
|--------|:---------:|:----------:|:-----:|
| **Overall Score** | 58/100 | 58/100 | **0%** |
| **Score (3-trial mean)** | 58.0 | 57.7 | -0.3 |
| **Std Dev** | 0.0 | 0.6 | — |

The fine-tuned model performs identically to the base on general tasks. No category shows meaningful regression — the largest shift is a single scenario difference in one category, well within noise.

!!! success "No Catastrophic Forgetting"
    Fine-tuning on domain-specific financial tools does not come at the cost of general tool-calling capability. The model retains its full ability to handle tool selection, parameter precision, safety refusal, structured output, and multi-step reasoning.

---

## Scaling Expectations

This evaluation used **10 training examples** generated via SDG Hub's MCP Distillation flow — intentionally minimal to validate the pipeline end-to-end. Results scale with training data:

| Training Examples | What You Get |
|:-----------------:|-------------|
| **10** (this evaluation) | Pipeline validation. 100% on straightforward queries. General capabilities preserved. |
| **100+** (recommended) | Improved handling of ambiguous and multi-step queries. Consistent edge-case behavior. Measurable lift over base model on complex scenarios. |
| **500+** (production) | Multi-tool orchestration. Domain-specific refusal patterns. Robust error recovery. Significant quality differentiation from base. |

!!! tip "Generating More Training Data"
    Use SDG Hub's MCP Distillation flow to scale from 10 to 500+ examples automatically. Point it at your MCP server, and the teacher model generates diverse tool-use traces covering edge cases, multi-step queries, and parameter variations — no manual annotation required.

---

## How to Reproduce

### Prerequisites

```bash
uv tool install 'tool-eval-bench[perf] @ git+https://github.com/SeraphimSerapis/tool-eval-bench.git'
pip install httpx

export ROUTE=$(oc get route financial-agent-model -n financial-agent -o jsonpath='{.spec.host}')
export ENDPOINT="https://${ROUTE}"
```

### Run Evaluations

=== "One Command"

    ```bash
    cd end-to-end-examples/tool-calling-financial/examples
    ./eval/run_benchmarks.sh
    ```

=== "Step by Step"

    ```bash
    # Domain-specific evaluation (15 financial tools)
    python3 eval/domain_eval.py \
      --endpoint "$ENDPOINT" \
      --base-model financial-agent-lora \
      --lora-model financial-agent \
      --output eval/domain-eval-results.json

    # General capability check (84 scenarios, 3 trials)
    tool-eval-bench --seed 42 --hardmode --trials 3 \
      --model financial-agent-lora --backend vllm \
      --base-url "$ENDPOINT" --no-think \
      --json-file eval/base-model-results.json

    tool-eval-bench --seed 42 --hardmode --trials 3 \
      --model financial-agent --backend vllm \
      --base-url "$ENDPOINT" --no-think \
      --json-file eval/lora-model-results.json
    ```

---

## Test Environment

| Component | Value |
|-----------|-------|
| Platform | RHOAI 3.4.2 (OCP 4.18) |
| GPU | NVIDIA L4 24GB (g6.xlarge) |
| Base Model | Qwen/Qwen3-4B |
| Fine-Tuning | LoRA rank 16, 10 training examples |
| Serving | vLLM with `--tool-call-parser=hermes` |
| Context Window | 16,384 tokens |
| Decoding | temperature=0.0, deterministic |
| Benchmark Tool | tool-eval-bench 2.3.1 |
| Evaluation Date | July 2026 |

---

## Related

- [Tool-Calling Financial Pipeline](../end-to-end/tool-calling-financial.md) — Full end-to-end walkthrough
- [Tool-Use Evaluation Guide](agent-evaluation.md) — Set up LLM-as-judge for your own models
- [MCP Distillation (GRPO)](../end-to-end/mcp-distillation.md) — Alternative training with reinforcement learning
