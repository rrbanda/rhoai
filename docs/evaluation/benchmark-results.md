# Tool-Calling Fine-Tuning: Evaluation Results

!!! abstract "Summary"
    Fine-tuning Qwen3-4B with 136 targeted tool-calling examples on RHOAI demonstrates measurable improvement: the fine-tuned model scores **91% on hard multi-step queries vs 89% for the base model**, fixing sequential tool-chain failures while preserving general capabilities. On simple single-tool queries, both models achieve **100% accuracy** across all 15 financial tools.

---

## What We Measured

We evaluated the fine-tuned model on three dimensions:

| Evaluation | Purpose | Base Model | Fine-Tuned | Delta |
|-----------|---------|:----------:|:----------:|:-----:|
| **Hard queries** (35 multi-step/complex) | Does fine-tuning improve complex tool use? | 89% (31/35) | **91% (32/35)** | **+2%** |
| **Simple queries** (20 single-tool) | Baseline accuracy on straightforward calls | 100% (20/20) | **100% (20/20)** | 0% |
| **General capability** (84 generic scenarios) | Did fine-tuning break anything? | 58/100 | 58/100 | **0%** |

**Setup:** Qwen3-4B fine-tuned with LoRA (rank 16, 3 epochs) using 136 domain-specific training examples, deployed on RHOAI 3.4.2 with vLLM on NVIDIA L4 GPU.

---

## Hard Query Results: Where Fine-Tuning Helps

The hard evaluation tests 35 queries across 5 categories designed to stress the model's tool-calling capabilities beyond simple single-tool routing.

| Category | Base Model | Fine-Tuned | Delta |
|----------|:---------:|:----------:|:-----:|
| **Multi-step chains** (16 queries) | 88% | **94%** | **+6%** |
| **Parallel tool calls** (7 queries) | 100% | 100% | 0% |
| **Complex parameter inference** (5 queries) | 80% | 80% | 0% |
| **Implicit/ambiguous routing** (5 queries) | 80% | 80% | 0% |
| **No-tool (should abstain)** (2 queries) | 100% | 100% | 0% |

!!! success "Key Improvement: Sequential Tool Chains"
    The fine-tuned model correctly handles "check compliance then submit trade" patterns that the base model fails on. When asked to verify compliance **and** execute a trade, the base model stops after the compliance check — the fine-tuned model completes both steps.

**Specific queries that improved:**

| Query | Base | LoRA | What Changed |
|-------|:----:|:----:|-------------|
| "Sell 50 shares of MSFT from IRA-2024 with a stop order at $400. First verify compliance." | FAIL | **PASS** | Base only called `check_compliance`; LoRA correctly calls both `check_compliance` + `submit_trade_order` |
| "How much money am I making on BALANCED-1?" | FAIL | **PASS** | Base returned text only; LoRA correctly calls `get_portfolio_performance` |

---

## Simple Query Results: Domain Accuracy

On straightforward single-tool queries, both models achieve perfect accuracy — confirming that the base Qwen3-4B model is already strong at basic tool routing when schemas are well-designed.

| Metric | Score |
|--------|:-----:|
| **Tool Selection Accuracy** | 100% (20/20) |
| **Parameter Extraction Accuracy** | 100% (20/20) |
| **Output Format Compliance** | 100% (20/20) |

**Example queries both models handle correctly:**

| User Query | Tool Called | Key Parameters |
|-----------|------------|----------------|
| "Give me a full analysis of Microsoft stock" | `analyze_stock` | `ticker: MSFT` |
| "Buy 50 shares of Google at market price in my TECH-FUND portfolio" | `submit_trade_order` | `ticker: GOOGL, action: buy, quantity: 50, order_type: market` |
| "Run a market crash stress test on my portfolio CONSERVATIVE-A" | `run_stress_test` | `portfolio_id: CONSERVATIVE-A, scenario: market_crash` |
| "Find high-dividend healthcare stocks with at least 3% yield" | `screen_stocks` | `sector: Healthcare, min_dividend_yield: 3` |

---

## General Capability Preservation

To verify fine-tuning did not degrade the model's general tool-calling ability, we ran [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) — an industry-standard benchmark with 84 deterministic scenarios across 15 categories.

| Metric | Base Model | Fine-Tuned | Delta |
|--------|:---------:|:----------:|:-----:|
| **Overall Score** | 58/100 | 58/100 | **0%** |
| **Score (3-trial mean)** | 58.0 | 57.7 | -0.3 |
| **Std Dev** | 0.0 | 0.6 | — |

!!! success "No Catastrophic Forgetting"
    Fine-tuning on domain-specific financial tools does not come at the cost of general tool-calling capability. The model retains its full ability to handle tool selection, parameter precision, safety refusal, structured output, and multi-step reasoning.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Training examples | 136 (10 original + 126 generated) |
| Training focus | Multi-step chains (42%), temporal inference (23%), implicit routing (24%), parallel calls (11%) |
| LoRA rank | 16 |
| Epochs | 3 |
| Max sequence length | 4,096 tokens |
| Training time | ~13 minutes on NVIDIA L4 |
| Final training loss | 0.024 |

!!! tip "Generating Training Data"
    The 126 additional examples were generated programmatically using `generate_multistep_data.py` — no teacher model or Langflow required. For production deployments, use SDG Hub's MCP Distillation flow to generate 500+ diverse examples automatically from your MCP server.

---

## How to Reproduce

### Evaluation Methodology

Three complementary evaluation techniques are used, each targeting a different dimension of tool-calling quality:

**1. Hard Query Evaluation (`eval/compare_models.py`)**

Sends 35 challenging queries to both models via the OpenAI-compatible `/v1/chat/completions` endpoint with all 15 financial tool schemas provided. Each query is scored on:

- **Tool selection**: Did the model call the expected tool(s)?
- **Call count**: Did the model issue the minimum required number of tool calls? (e.g., "check compliance then submit trade" requires 2 calls)
- **Parameter accuracy**: For queries with expected parameters (like `period=ytd` for "this year"), are the arguments correct?

A query **passes** only if all three criteria are met. Queries are categorized into multi-step chains, parallel calls, complex parameter inference, implicit/ambiguous routing, and no-tool abstention. The script runs both models back-to-back with `temperature=0.0` for deterministic output and produces a JSON report with per-query results.

**2. Domain-Specific Evaluation (`eval/domain_eval.py`)**

Tests 20 straightforward single-tool queries against the 15 financial tools — one query per tool plus extras for tools with complex parameter combinations. Measures tool selection accuracy, parameter extraction accuracy, and output format compliance (valid `tool_calls` JSON structure). This confirms baseline correctness on simple cases.

**3. General Capability Check ([tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench))**

An open-source, industry-standard benchmark with 84 deterministic scenarios across 15 categories (multi-step chains, error recovery, safety boundaries, structured output, etc.). Runs 3 trials with a fixed seed for statistical stability. This detects catastrophic forgetting — if fine-tuning degraded capabilities the model previously had.

**Training Data Generation (`generate_multistep_data.py`)**

Programmatically generates tool-calling training examples using template-based randomization over the 15 financial tool schemas. Produces examples in the same JSONL format as the SDG Hub distillation pipeline (`{"messages": [system, user, assistant(tool_calls), tool(response), assistant(final_answer)]}`). Covers four patterns: multi-step chains (42%), temporal parameter inference (23%), implicit routing (24%), and parallel calls (11%).

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
    # Hard query comparison (35 multi-step queries)
    python3 eval/compare_models.py

    # Domain-specific evaluation (15 financial tools, 20 simple queries)
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
| Fine-Tuning | LoRA rank 16, 136 training examples, 3 epochs |
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
