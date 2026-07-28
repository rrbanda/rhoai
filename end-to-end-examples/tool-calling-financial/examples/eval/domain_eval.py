#!/usr/bin/env python3
"""Domain-specific tool-calling evaluation: Base vs LoRA fine-tuned model.

Evaluates both models on 20 financial tool-calling scenarios using the same
15 financial tools the model was fine-tuned on. Measures:
  - Tool selection accuracy (correct tool chosen)
  - Parameter accuracy (correct arguments passed)
  - Format compliance (valid tool_calls structure)

Usage:
    python eval/domain_eval.py --endpoint https://<route>
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import httpx

FINANCIAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "Get the current stock price and basic quote data for a given ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL, MSFT)"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_summary",
            "description": "Get a summary of major market indices and overall market conditions.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_prices",
            "description": "Get historical price data for a stock over a specified period.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "period": {"type": "string", "description": "Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 5y)"},
                    "interval": {"type": "string", "description": "Data interval (1m, 5m, 1h, 1d, 1wk)"},
                },
                "required": ["ticker", "period"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "screen_stocks",
            "description": "Screen stocks based on criteria like market cap, sector, P/E ratio, dividend yield.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {"type": "string", "description": "Market sector (Technology, Healthcare, Finance, Energy, etc.)"},
                    "min_market_cap": {"type": "number", "description": "Minimum market cap in billions"},
                    "max_pe_ratio": {"type": "number", "description": "Maximum P/E ratio"},
                    "min_dividend_yield": {"type": "number", "description": "Minimum dividend yield percentage"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_positions",
            "description": "Get all current positions in a portfolio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"}
                },
                "required": ["portfolio_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_performance",
            "description": "Get portfolio performance metrics over a specified period.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"},
                    "period": {"type": "string", "description": "Performance period (1d, 1w, 1m, 3m, 6m, 1y, ytd)"},
                },
                "required": ["portfolio_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_account_summary",
            "description": "Get account summary including total value, cash balance, and buying power.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"}
                },
                "required": ["portfolio_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transaction_history",
            "description": "Get transaction history for a portfolio with optional filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"},
                    "ticker": {"type": "string", "description": "Filter by ticker symbol"},
                    "transaction_type": {"type": "string", "description": "Filter by type: buy, sell, dividend, split"},
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                },
                "required": ["portfolio_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_portfolio_risk",
            "description": "Calculate risk metrics for a portfolio including VaR, Sharpe ratio, and beta.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"}
                },
                "required": ["portfolio_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_exposure",
            "description": "Get sector allocation breakdown for a portfolio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"}
                },
                "required": ["portfolio_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_stress_test",
            "description": "Run a stress test scenario on a portfolio to model potential losses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"},
                    "scenario": {"type": "string", "description": "Stress scenario: market_crash, rate_hike, recession, sector_rotation"},
                },
                "required": ["portfolio_id", "scenario"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_stock",
            "description": "Get comprehensive analysis of a stock including fundamentals, technicals, and analyst ratings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_trade_order",
            "description": "Submit a trade order to buy or sell a security.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"},
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "action": {"type": "string", "description": "Trade action: buy or sell"},
                    "quantity": {"type": "integer", "description": "Number of shares"},
                    "order_type": {"type": "string", "description": "Order type: market, limit, stop"},
                    "limit_price": {"type": "number", "description": "Limit price (required for limit orders)"},
                },
                "required": ["portfolio_id", "ticker", "action", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_compliance",
            "description": "Check if a proposed trade complies with portfolio constraints and regulations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"},
                    "ticker": {"type": "string", "description": "Stock ticker to check"},
                    "action": {"type": "string", "description": "Proposed action: buy or sell"},
                    "quantity": {"type": "integer", "description": "Proposed quantity"},
                },
                "required": ["portfolio_id", "ticker", "action", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_regulatory_status",
            "description": "Get regulatory compliance status and any pending issues for a portfolio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"}
                },
                "required": ["portfolio_id"],
            },
        },
    },
]

TEST_CASES = [
    {
        "query": "What's the current price of Tesla stock?",
        "expected_tools": ["get_stock_quote"],
        "expected_params": {"ticker": "TSLA"},
        "complexity": "simple",
    },
    {
        "query": "Show me Apple's stock price history for the last 3 months.",
        "expected_tools": ["get_historical_prices"],
        "expected_params": {"ticker": "AAPL", "period": "3mo"},
        "complexity": "simple",
    },
    {
        "query": "How is the overall market doing today?",
        "expected_tools": ["get_market_summary"],
        "expected_params": {},
        "complexity": "simple",
    },
    {
        "query": "Find technology stocks with market cap over 100 billion and PE ratio under 30.",
        "expected_tools": ["screen_stocks"],
        "expected_params": {"sector": "Technology", "min_market_cap": 100, "max_pe_ratio": 30},
        "complexity": "simple",
    },
    {
        "query": "What positions do I have in portfolio ACME-401K?",
        "expected_tools": ["get_portfolio_positions"],
        "expected_params": {"portfolio_id": "ACME-401K"},
        "complexity": "simple",
    },
    {
        "query": "How has my retirement portfolio RET-2024 performed year to date?",
        "expected_tools": ["get_portfolio_performance"],
        "expected_params": {"portfolio_id": "RET-2024", "period": "ytd"},
        "complexity": "simple",
    },
    {
        "query": "What's the total value and cash balance in my main-portfolio account?",
        "expected_tools": ["get_account_summary"],
        "expected_params": {"portfolio_id": "main-portfolio"},
        "complexity": "simple",
    },
    {
        "query": "Show me all buy transactions for NVDA in portfolio GROWTH-1 from January to March 2024.",
        "expected_tools": ["get_transaction_history"],
        "expected_params": {"portfolio_id": "GROWTH-1", "ticker": "NVDA", "transaction_type": "buy", "start_date": "2024-01-01", "end_date": "2024-03-31"},
        "complexity": "medium",
    },
    {
        "query": "Calculate the risk metrics for my aggressive portfolio AGG-PORT.",
        "expected_tools": ["calculate_portfolio_risk"],
        "expected_params": {"portfolio_id": "AGG-PORT"},
        "complexity": "simple",
    },
    {
        "query": "What sectors is my portfolio BALANCED-1 exposed to?",
        "expected_tools": ["get_sector_exposure"],
        "expected_params": {"portfolio_id": "BALANCED-1"},
        "complexity": "simple",
    },
    {
        "query": "Run a market crash stress test on my portfolio CONSERVATIVE-A.",
        "expected_tools": ["run_stress_test"],
        "expected_params": {"portfolio_id": "CONSERVATIVE-A", "scenario": "market_crash"},
        "complexity": "simple",
    },
    {
        "query": "Give me a full analysis of Microsoft stock.",
        "expected_tools": ["analyze_stock"],
        "expected_params": {"ticker": "MSFT"},
        "complexity": "simple",
    },
    {
        "query": "Buy 50 shares of Google at market price in my TECH-FUND portfolio.",
        "expected_tools": ["submit_trade_order"],
        "expected_params": {"portfolio_id": "TECH-FUND", "ticker": ["GOOGL", "GOOG"], "action": "buy", "quantity": 50, "order_type": "market"},
        "complexity": "medium",
    },
    {
        "query": "Can I buy 200 shares of AMD in portfolio GROWTH-1 without violating compliance rules?",
        "expected_tools": ["check_compliance"],
        "expected_params": {"portfolio_id": "GROWTH-1", "ticker": "AMD", "action": "buy", "quantity": 200},
        "complexity": "simple",
    },
    {
        "query": "Check the regulatory compliance status of portfolio INST-FUND-7.",
        "expected_tools": ["get_regulatory_status"],
        "expected_params": {"portfolio_id": "INST-FUND-7"},
        "complexity": "simple",
    },
    {
        "query": "Place a limit order to sell 100 shares of AMZN at $200 in my MAIN account.",
        "expected_tools": ["submit_trade_order"],
        "expected_params": {"portfolio_id": "MAIN", "ticker": "AMZN", "action": "sell", "quantity": 100, "order_type": "limit", "limit_price": 200},
        "complexity": "medium",
    },
    {
        "query": "What would happen to my pension fund PENSION-2024 if interest rates spike?",
        "expected_tools": ["run_stress_test"],
        "expected_params": {"portfolio_id": "PENSION-2024", "scenario": "rate_hike"},
        "complexity": "simple",
    },
    {
        "query": "Show me NVIDIA weekly price data for the past year.",
        "expected_tools": ["get_historical_prices"],
        "expected_params": {"ticker": "NVDA", "period": "1y", "interval": "1wk"},
        "complexity": "medium",
    },
    {
        "query": "Find high-dividend healthcare stocks with at least 3% yield.",
        "expected_tools": ["screen_stocks"],
        "expected_params": {"sector": "Healthcare", "min_dividend_yield": 3},
        "complexity": "simple",
    },
    {
        "query": "Show me the dividend payments received in portfolio INCOME-1 during 2024.",
        "expected_tools": ["get_transaction_history"],
        "expected_params": {"portfolio_id": "INCOME-1", "transaction_type": "dividend", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "complexity": "medium",
    },
]

HARD_TEST_CASES = [
    {
        "query": "I want a detailed analysis of NVDA including technicals and fundamentals.",
        "expected_tools": ["analyze_stock"],
        "expected_params": {"ticker": "NVDA"},
        "complexity": "hard",
    },
    {
        "query": "Tell me about Amazon stock.",
        "expected_tools": ["analyze_stock", "get_stock_quote"],
        "expected_params": {"ticker": ["AMZN", "AMAZON"]},
        "complexity": "hard",
    },
    {
        "query": "Find me cheap energy stocks.",
        "expected_tools": ["screen_stocks"],
        "expected_params": {"sector": "Energy"},
        "complexity": "hard",
    },
    {
        "query": "How much money did my portfolio TECH-1 make this quarter?",
        "expected_tools": ["get_portfolio_performance"],
        "expected_params": {"portfolio_id": "TECH-1"},
        "complexity": "hard",
    },
    {
        "query": "How risky is my portfolio AGGR-5?",
        "expected_tools": ["calculate_portfolio_risk"],
        "expected_params": {"portfolio_id": "AGGR-5"},
        "complexity": "hard",
    },
    {
        "query": "Set a stop loss: sell 75 shares of META if it drops to 450 in my IRA-2024.",
        "expected_tools": ["submit_trade_order"],
        "expected_params": {"portfolio_id": "IRA-2024", "ticker": "META", "action": "sell", "quantity": 75},
        "complexity": "hard",
    },
    {
        "query": "Before I buy 500 shares of GME in MEME-PORT, will that pass compliance?",
        "expected_tools": ["check_compliance"],
        "expected_params": {"portfolio_id": "MEME-PORT", "ticker": "GME", "action": "buy", "quantity": 500},
        "complexity": "hard",
    },
    {
        "query": "I need minute-by-minute TSLA data from today.",
        "expected_tools": ["get_historical_prices"],
        "expected_params": {"ticker": "TSLA", "period": "1d"},
        "complexity": "hard",
    },
    {
        "query": "Am I too concentrated in tech across my portfolio DIV-FUND?",
        "expected_tools": ["get_sector_exposure"],
        "expected_params": {"portfolio_id": "DIV-FUND"},
        "complexity": "hard",
    },
    {
        "query": "What if there is a recession? How would CONSERVATIVE-2 be affected?",
        "expected_tools": ["run_stress_test"],
        "expected_params": {"portfolio_id": "CONSERVATIVE-2", "scenario": "recession"},
        "complexity": "hard",
    },
    {
        "query": "Is it legal for me to dump all 1000 shares of INSIDER-STOCK from portfolio EXEC-1?",
        "expected_tools": ["check_compliance"],
        "expected_params": {"portfolio_id": "EXEC-1", "ticker": "INSIDER-STOCK", "action": "sell", "quantity": 1000},
        "complexity": "hard",
    },
    {
        "query": "Any compliance issues I should know about for HEDGE-FUND-A?",
        "expected_tools": ["get_regulatory_status"],
        "expected_params": {"portfolio_id": "HEDGE-FUND-A"},
        "complexity": "hard",
    },
    {
        "query": "Compare AAPL and MSFT stock prices.",
        "expected_tools": ["get_stock_quote"],
        "expected_params": {},
        "expect_parallel": True,
        "min_tool_calls": 2,
        "complexity": "hard",
    },
    {
        "query": "How much cash do I have available in GROWTH-2?",
        "expected_tools": ["get_account_summary"],
        "expected_params": {"portfolio_id": "GROWTH-2"},
        "complexity": "hard",
    },
    {
        "query": "What is a P/E ratio?",
        "expected_tools": [],
        "expected_params": {},
        "expect_no_tool": True,
        "complexity": "hard",
    },
]


@dataclass
class EvalResult:
    query: str
    model: str
    tool_correct: bool = False
    params_correct: bool = False
    format_valid: bool = False
    latency_ms: float = 0.0
    error: str = ""
    actual_tool: str = ""
    actual_params: dict = field(default_factory=dict)


def call_model(client: httpx.Client, endpoint: str, model: str, query: str) -> tuple[dict | None, float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "tools": FINANCIAL_TOOLS,
        "tool_choice": "auto",
        "max_tokens": 512,
        "temperature": 0.0,
    }
    t0 = time.time()
    try:
        resp = client.post(
            f"{endpoint}/v1/chat/completions",
            json=payload,
            timeout=60,
        )
        latency = (time.time() - t0) * 1000
        resp.raise_for_status()
        return resp.json(), latency
    except Exception as e:
        latency = (time.time() - t0) * 1000
        print(f"    ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return None, latency


def evaluate_response(response: dict | None, expected_tools: list, expected_params: dict) -> tuple[bool, bool, bool, str, dict]:
    if response is None:
        return False, False, False, "", {}

    choices = response.get("choices", [])
    if not choices:
        return False, False, False, "", {}

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    if not tool_calls:
        return False, False, False, "", {}

    format_valid = True
    actual_tools = []
    all_params = {}

    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        actual_tools.append(name)
        try:
            params = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            format_valid = False
            params = {}
        all_params[name] = params

    actual_tool = actual_tools[0] if actual_tools else ""
    actual_params = all_params.get(actual_tool, {})

    tool_correct = actual_tool in expected_tools

    params_correct = True
    if tool_correct and expected_params:
        for key, expected_val in expected_params.items():
            actual_val = actual_params.get(key)
            if actual_val is None:
                params_correct = False
                break
            # Support list of acceptable values (e.g., ["GOOGL", "GOOG"])
            if isinstance(expected_val, list):
                if str(actual_val).upper() not in [str(v).upper() for v in expected_val]:
                    params_correct = False
                    break
            elif isinstance(expected_val, str):
                if expected_val.lower() != str(actual_val).lower():
                    params_correct = False
                    break
            elif isinstance(expected_val, (int, float)):
                try:
                    if abs(float(actual_val) - float(expected_val)) > 0.01:
                        params_correct = False
                        break
                except (ValueError, TypeError):
                    params_correct = False
                    break

    return tool_correct, params_correct, format_valid, actual_tool, actual_params


def evaluate_test_case(response: dict | None, tc: dict) -> tuple[bool, bool, bool, str, dict]:
    """Evaluate a single test case, handling special flags like expect_no_tool and expect_parallel."""
    expected_tools = tc["expected_tools"]
    expected_params = tc["expected_params"]

    if response is None:
        return False, False, False, "", {}

    choices = response.get("choices", [])
    if not choices:
        return False, False, False, "", {}

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    if tc.get("expect_no_tool"):
        no_tool = not tool_calls
        return no_tool, no_tool, True, "(none)", {}

    if not tool_calls:
        return False, False, False, "", {}

    if tc.get("expect_parallel"):
        min_calls = tc.get("min_tool_calls", 2)
        if len(tool_calls) < min_calls:
            return False, False, True, f"{len(tool_calls)} calls (expected >={min_calls})", {}
        all_valid = all(
            tc_item.get("function", {}).get("name", "") in expected_tools
            for tc_item in tool_calls
        )
        return all_valid, all_valid, True, f"{len(tool_calls)} parallel calls", {}

    return evaluate_response(response, expected_tools, expected_params)


def run_evaluation(endpoint: str, models: list[str], test_cases: list[dict], label: str = "simple") -> dict[str, list[EvalResult]]:
    client = httpx.Client(verify=False, timeout=60)
    results: dict[str, list[EvalResult]] = {m: [] for m in models}
    total = len(test_cases)

    for model in models:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model} ({label}, {total} queries)")
        print(f"{'='*60}")
        for i, tc in enumerate(test_cases, 1):
            response, latency = call_model(client, endpoint, model, tc["query"])
            tool_ok, params_ok, fmt_ok, actual_tool, actual_params = evaluate_test_case(response, tc)
            result = EvalResult(
                query=tc["query"],
                model=model,
                tool_correct=tool_ok,
                params_correct=params_ok,
                format_valid=fmt_ok,
                latency_ms=latency,
                actual_tool=actual_tool,
                actual_params=actual_params,
            )
            results[model].append(result)
            status = "PASS" if (tool_ok and params_ok) else ("TOOL_OK" if tool_ok else "FAIL")
            expected_label = tc["expected_tools"][0] if tc["expected_tools"] else "(no-tool)"
            print(f"  [{i:2d}/{total}] {status:7s} | {expected_label:30s} | {latency:6.0f}ms | {actual_tool}")

    client.close()
    return results


def print_summary(results: dict[str, list[EvalResult]], label: str = "Simple", total_queries: int = 20):
    print(f"\n{'='*80}")
    print(f"  DOMAIN-SPECIFIC EVALUATION RESULTS ({label}, {total_queries} Queries)")
    print(f"{'='*80}\n")

    header = f"{'Metric':<35}"
    for model in results:
        header += f"  {model:>22}"
    print(header)
    print("-" * len(header))

    for metric_name, metric_fn in [
        ("Tool Selection Accuracy", lambda r: r.tool_correct),
        ("Parameter Accuracy", lambda r: r.params_correct),
        ("Format Compliance", lambda r: r.format_valid),
        ("Full Pass (tool + params)", lambda r: r.tool_correct and r.params_correct),
    ]:
        row = f"{metric_name:<35}"
        for model in results:
            score = sum(1 for r in results[model] if metric_fn(r))
            pct = score / len(results[model]) * 100
            row += f"  {pct:>18.1f}%  "
        print(row)

    print()
    row = f"{'Avg Latency (ms)':<35}"
    for model in results:
        avg = sum(r.latency_ms for r in results[model]) / len(results[model])
        row += f"  {avg:>18.0f}ms "
    print(row)

    print(f"\n{'='*80}")
    print("  PER-QUERY COMPARISON")
    print(f"{'='*80}\n")

    models = list(results.keys())
    print(f"{'#':<4} {'Expected Tool':<30} ", end="")
    for m in models:
        print(f" {m[:15]:>15}", end="")
    print()
    print("-" * (35 + 17 * len(models)))

    test_set = TEST_CASES if label == "Simple" else HARD_TEST_CASES
    for i in range(len(test_set)):
        tc = test_set[i]
        expected_label = tc["expected_tools"][0] if tc["expected_tools"] else "(no-tool)"
        print(f"{i+1:<4} {expected_label:<30} ", end="")
        for m in models:
            r = results[m][i]
            if r.tool_correct and r.params_correct:
                status = "PASS"
            elif r.tool_correct:
                status = "TOOL_OK"
            else:
                status = "FAIL"
            print(f" {status:>15}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Domain-specific financial tool-calling evaluation")
    parser.add_argument("--endpoint", required=True, help="vLLM endpoint URL (without /v1)")
    parser.add_argument("--base-model", default="financial-agent-lora", help="Base model name")
    parser.add_argument("--lora-model", default="financial-agent", help="LoRA fine-tuned model name")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--hard-only", action="store_true", help="Only run hard test cases")
    parser.add_argument("--simple-only", action="store_true", help="Only run simple test cases")
    args = parser.parse_args()

    models = [args.base_model, args.lora_model]

    all_results = {}

    if not args.hard_only:
        simple_results = run_evaluation(args.endpoint, models, TEST_CASES, label="simple")
        print_summary(simple_results, label="Simple", total_queries=len(TEST_CASES))
        all_results["simple"] = simple_results

    if not args.simple_only:
        hard_results = run_evaluation(args.endpoint, models, HARD_TEST_CASES, label="hard")
        print_summary(hard_results, label="Hard/Ambiguous", total_queries=len(HARD_TEST_CASES))
        all_results["hard"] = hard_results

    if args.output:
        output_data = {
            "benchmark": "domain-specific-financial-tools",
            "tools_count": len(FINANCIAL_TOOLS),
            "models": {},
            "test_sets": {},
        }
        for set_name, results in all_results.items():
            test_cases = TEST_CASES if set_name == "simple" else HARD_TEST_CASES
            output_data["test_sets"][set_name] = {"queries_count": len(test_cases)}
            for model in models:
                if model not in output_data["models"]:
                    output_data["models"][model] = {}
                model_results = results[model]
                output_data["models"][model][set_name] = {
                    "tool_selection_accuracy": sum(1 for r in model_results if r.tool_correct) / len(model_results) * 100,
                    "parameter_accuracy": sum(1 for r in model_results if r.params_correct) / len(model_results) * 100,
                    "format_compliance": sum(1 for r in model_results if r.format_valid) / len(model_results) * 100,
                    "full_pass_rate": sum(1 for r in model_results if r.tool_correct and r.params_correct) / len(model_results) * 100,
                    "avg_latency_ms": sum(r.latency_ms for r in model_results) / len(model_results),
                    "per_query": [
                        {
                            "query": r.query,
                            "tool_correct": r.tool_correct,
                            "params_correct": r.params_correct,
                            "format_valid": r.format_valid,
                            "actual_tool": r.actual_tool,
                            "actual_params": r.actual_params,
                            "latency_ms": r.latency_ms,
                        }
                        for r in model_results
                    ],
                }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
