#!/usr/bin/env python3
"""Side-by-side evaluation of base vs fine-tuned LoRA model on 35 hard queries.

Tests both models on the same queries and produces a comparison report.
"""

import json
import sys
import time
import httpx

ENDPOINT = "https://financial-agent-model-financial-agent.apps.cluster-4l6x6.4l6x6.sandbox1213.opentlc.com"
MODELS = {
    "base": "financial-agent-lora",
    "lora": "financial-agent",
}

FINANCIAL_TOOLS = [
    {"type": "function", "function": {"name": "get_stock_quote", "description": "Get the current stock price and basic quote data for a given ticker symbol.", "parameters": {"type": "object", "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL, MSFT)"}}, "required": ["ticker"]}}},
    {"type": "function", "function": {"name": "get_market_summary", "description": "Get a summary of major market indices and overall market conditions.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_historical_prices", "description": "Get historical price data for a stock over a specified period.", "parameters": {"type": "object", "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol"}, "period": {"type": "string", "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y"], "description": "Time period"}, "interval": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "1d", "1wk", "1mo"], "description": "Data interval"}}, "required": ["ticker"]}}},
    {"type": "function", "function": {"name": "screen_stocks", "description": "Screen stocks based on criteria like sector, market cap, PE ratio, dividend yield.", "parameters": {"type": "object", "properties": {"sector": {"type": "string", "description": "Filter by sector (e.g., Technology, Healthcare, Energy)"}, "min_market_cap": {"type": "number", "description": "Minimum market cap in billions"}, "max_pe_ratio": {"type": "number", "description": "Maximum P/E ratio"}, "min_dividend_yield": {"type": "number", "description": "Minimum dividend yield percentage"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_portfolio_positions", "description": "Get all current positions in a portfolio with quantities, cost basis, and current value.", "parameters": {"type": "object", "properties": {"portfolio_id": {"type": "string", "description": "Portfolio identifier"}}, "required": ["portfolio_id"]}}},
    {"type": "function", "function": {"name": "get_portfolio_performance", "description": "Get portfolio performance metrics including returns, alpha, beta, and Sharpe ratio.", "parameters": {"type": "object", "properties": {"portfolio_id": {"type": "string", "description": "Portfolio identifier"}, "period": {"type": "string", "enum": ["1d", "1w", "1m", "3m", "6m", "1y", "ytd"], "description": "Performance period"}}, "required": ["portfolio_id"]}}},
    {"type": "function", "function": {"name": "get_account_summary", "description": "Get account-level summary including cash balance, total equity, margin, and buying power.", "parameters": {"type": "object", "properties": {"portfolio_id": {"type": "string", "description": "Portfolio identifier"}}, "required": ["portfolio_id"]}}},
    {"type": "function", "function": {"name": "get_transaction_history", "description": "Retrieve transaction history for a portfolio with optional filtering.", "parameters": {"type": "object", "properties": {"portfolio_id": {"type": "string", "description": "Portfolio identifier"}, "transaction_type": {"type": "string", "enum": ["buy", "sell", "dividend", "fee"], "description": "Filter by type"}, "ticker": {"type": "string", "description": "Filter by ticker"}, "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"}, "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}}, "required": ["portfolio_id"]}}},
    {"type": "function", "function": {"name": "calculate_portfolio_risk", "description": "Calculate risk metrics for a portfolio including VaR, volatility, max drawdown, and beta.", "parameters": {"type": "object", "properties": {"portfolio_id": {"type": "string", "description": "Portfolio identifier"}}, "required": ["portfolio_id"]}}},
    {"type": "function", "function": {"name": "get_sector_exposure", "description": "Get portfolio allocation breakdown by sector with concentration percentages.", "parameters": {"type": "object", "properties": {"portfolio_id": {"type": "string", "description": "Portfolio identifier"}}, "required": ["portfolio_id"]}}},
    {"type": "function", "function": {"name": "run_stress_test", "description": "Run a stress test scenario on a portfolio to estimate potential losses.", "parameters": {"type": "object", "properties": {"portfolio_id": {"type": "string", "description": "Portfolio identifier"}, "scenario": {"type": "string", "enum": ["market_crash", "rate_hike", "recession", "tech_selloff", "inflation_spike"], "description": "Stress scenario"}}, "required": ["portfolio_id", "scenario"]}}},
    {"type": "function", "function": {"name": "analyze_stock", "description": "Run comprehensive analysis on a stock including fundamentals, technicals, and analyst consensus.", "parameters": {"type": "object", "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol"}}, "required": ["ticker"]}}},
    {"type": "function", "function": {"name": "submit_trade_order", "description": "Submit a buy or sell order for a stock in a portfolio.", "parameters": {"type": "object", "properties": {"portfolio_id": {"type": "string", "description": "Portfolio identifier"}, "ticker": {"type": "string", "description": "Stock ticker symbol"}, "action": {"type": "string", "enum": ["buy", "sell"], "description": "Trade direction"}, "quantity": {"type": "integer", "description": "Number of shares"}, "order_type": {"type": "string", "enum": ["market", "limit", "stop"], "description": "Order type"}, "limit_price": {"type": "number", "description": "Limit price (for limit/stop orders)"}}, "required": ["portfolio_id", "ticker", "action", "quantity"]}}},
    {"type": "function", "function": {"name": "check_compliance", "description": "Check whether a proposed trade complies with portfolio constraints and regulatory rules.", "parameters": {"type": "object", "properties": {"portfolio_id": {"type": "string", "description": "Portfolio identifier"}, "ticker": {"type": "string", "description": "Stock ticker symbol"}, "action": {"type": "string", "enum": ["buy", "sell"], "description": "Proposed trade direction"}, "quantity": {"type": "integer", "description": "Proposed number of shares"}}, "required": ["portfolio_id", "ticker", "action", "quantity"]}}},
    {"type": "function", "function": {"name": "get_regulatory_status", "description": "Get the regulatory compliance status of a portfolio including any active restrictions.", "parameters": {"type": "object", "properties": {"portfolio_id": {"type": "string", "description": "Portfolio identifier"}}, "required": ["portfolio_id"]}}},
]

HARD_QUERIES = [
    {"id": 1, "query": "I want to buy 200 shares of NVDA in my GROWTH-1 portfolio. First check if it passes compliance, and if so, submit the market order.", "expected_tools": ["check_compliance", "submit_trade_order"], "category": "multi-step", "min_calls": 2},
    {"id": 2, "query": "Get me the current price of TSLA, then analyze the stock to decide if it's a good buy.", "expected_tools": ["get_stock_quote", "analyze_stock"], "category": "multi-step", "min_calls": 2},
    {"id": 3, "query": "Check the risk of my AGGRESSIVE-1 portfolio and run a market crash stress test on it.", "expected_tools": ["calculate_portfolio_risk", "run_stress_test"], "category": "multi-step", "min_calls": 2},
    {"id": 4, "query": "Look at my IRA-2024 portfolio positions and tell me the sector exposure breakdown.", "expected_tools": ["get_portfolio_positions", "get_sector_exposure"], "category": "multi-step", "min_calls": 2},
    {"id": 5, "query": "I need to sell 100 shares of AAPL from TECH-FUND. Check compliance first, then submit the sell order at market price.", "expected_tools": ["check_compliance", "submit_trade_order"], "category": "multi-step", "min_calls": 2},
    {"id": 6, "query": "Get the performance of my BALANCED-1 portfolio for the year, and also calculate its risk metrics.", "expected_tools": ["get_portfolio_performance", "calculate_portfolio_risk"], "category": "multi-step", "min_calls": 2},
    {"id": 7, "query": "Show me AMZN historical prices for the last 3 months, and also get the current quote.", "expected_tools": ["get_historical_prices", "get_stock_quote"], "category": "multi-step", "min_calls": 2},
    {"id": 8, "query": "Before buying 500 shares of GME in MEME-PORT, check compliance and also check the account summary to see if I have enough cash.", "expected_tools": ["check_compliance", "get_account_summary"], "category": "multi-step", "min_calls": 2},
    {"id": 9, "query": "Get the regulatory status of INST-FUND-7 and also run an inflation spike stress test.", "expected_tools": ["get_regulatory_status", "run_stress_test"], "category": "multi-step", "min_calls": 2},
    {"id": 10, "query": "Analyze both AAPL and MSFT stocks for me.", "expected_tools": ["analyze_stock"], "category": "parallel", "min_calls": 2},
    {"id": 11, "query": "Compare the current stock prices of GOOGL, AMZN, and META.", "expected_tools": ["get_stock_quote"], "category": "parallel", "min_calls": 3},
    {"id": 12, "query": "Get risk metrics for both my AGGRESSIVE-1 and CONSERVATIVE-2 portfolios.", "expected_tools": ["calculate_portfolio_risk"], "category": "parallel", "min_calls": 2},
    {"id": 13, "query": "Check the sector exposure of my three portfolios: GROWTH-1, BALANCED-1, and INCOME-1.", "expected_tools": ["get_sector_exposure"], "category": "parallel", "min_calls": 3},
    {"id": 14, "query": "Run a recession stress test on GROWTH-1 and a rate hike stress test on CONSERVATIVE-2.", "expected_tools": ["run_stress_test"], "category": "parallel", "min_calls": 2},
    {"id": 15, "query": "Get quotes for the FAANG stocks: META, AAPL, AMZN, NFLX, and GOOGL.", "expected_tools": ["get_stock_quote"], "category": "parallel", "min_calls": 5},
    {"id": 16, "query": "Find me undervalued tech stocks with a PE under 20.", "expected_tools": ["screen_stocks"], "category": "complex-params", "min_calls": 1, "expected_params": {"sector": "Technology", "max_pe_ratio": 20}},
    {"id": 17, "query": "Show me how my retirement fund RET-2024 has done this year.", "expected_tools": ["get_portfolio_performance"], "category": "complex-params", "min_calls": 1, "expected_params": {"portfolio_id": "RET-2024", "period": "ytd"}},
    {"id": 18, "query": "I want to place a limit order to buy 75 shares of AMD at $150 in my TECH-FUND.", "expected_tools": ["submit_trade_order"], "category": "complex-params", "min_calls": 1, "expected_params": {"portfolio_id": "TECH-FUND", "ticker": "AMD", "action": "buy", "quantity": 75, "order_type": "limit", "limit_price": 150}},
    {"id": 19, "query": "What dividends have I received in INCOME-1 this year?", "expected_tools": ["get_transaction_history"], "category": "complex-params", "min_calls": 1, "expected_params": {"portfolio_id": "INCOME-1", "transaction_type": "dividend"}},
    {"id": 20, "query": "Screen for high-yield energy stocks paying at least 4% dividends.", "expected_tools": ["screen_stocks"], "category": "complex-params", "min_calls": 1, "expected_params": {"sector": "Energy", "min_dividend_yield": 4}},
    {"id": 21, "query": "Am I diversified enough in GROWTH-1?", "expected_tools": ["get_sector_exposure"], "category": "implicit", "min_calls": 1},
    {"id": 22, "query": "How risky is my AGGRESSIVE-1 portfolio right now?", "expected_tools": ["calculate_portfolio_risk"], "category": "implicit", "min_calls": 1},
    {"id": 23, "query": "What's my cash situation in MAIN-ACCOUNT?", "expected_tools": ["get_account_summary"], "category": "implicit", "min_calls": 1},
    {"id": 24, "query": "How much money am I making on BALANCED-1?", "expected_tools": ["get_portfolio_performance"], "category": "implicit", "min_calls": 1},
    {"id": 25, "query": "What would happen to CONSERVATIVE-2 if the market tanks?", "expected_tools": ["run_stress_test"], "category": "implicit", "min_calls": 1},
    {"id": 26, "query": "Get the market overview and then find cheap healthcare stocks.", "expected_tools": ["get_market_summary", "screen_stocks"], "category": "multi-step", "min_calls": 2},
    {"id": 27, "query": "Check my GROWTH-1 performance this quarter and also get all my positions.", "expected_tools": ["get_portfolio_performance", "get_portfolio_positions"], "category": "multi-step", "min_calls": 2},
    {"id": 28, "query": "I need a full review: get positions, performance, risk, and sector exposure for BALANCED-1.", "expected_tools": ["get_portfolio_positions", "get_portfolio_performance", "calculate_portfolio_risk", "get_sector_exposure"], "category": "multi-step", "min_calls": 4},
    {"id": 29, "query": "Sell 50 shares of MSFT from IRA-2024 with a stop order at $400. First verify compliance.", "expected_tools": ["check_compliance", "submit_trade_order"], "category": "multi-step", "min_calls": 2},
    {"id": 30, "query": "Get me transaction history and current positions for INCOME-1.", "expected_tools": ["get_transaction_history", "get_portfolio_positions"], "category": "multi-step", "min_calls": 2},
    {"id": 31, "query": "What is a Sharpe ratio and why does it matter?", "expected_tools": [], "category": "no-tool", "min_calls": 0},
    {"id": 32, "query": "Explain the difference between a market order and a limit order.", "expected_tools": [], "category": "no-tool", "min_calls": 0},
    {"id": 33, "query": "Do a complete health check on PENSION-2024: positions, performance YTD, risk analysis, sector concentration, and run both a recession and rate hike stress test.", "expected_tools": ["get_portfolio_positions", "get_portfolio_performance", "calculate_portfolio_risk", "get_sector_exposure", "run_stress_test"], "category": "multi-step", "min_calls": 5},
    {"id": 34, "query": "Screen for large-cap tech stocks, then analyze the top one (assume AAPL) in detail.", "expected_tools": ["screen_stocks", "analyze_stock"], "category": "multi-step", "min_calls": 2},
    {"id": 35, "query": "Check if GROWTH-1 can buy 1000 shares of NVDA, and also check CONSERVATIVE-2 for the same trade.", "expected_tools": ["check_compliance"], "category": "parallel", "min_calls": 2},
]


def call_model(client, model_id, query):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": query}],
        "tools": FINANCIAL_TOOLS,
        "tool_choice": "auto",
        "max_tokens": 1024,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    try:
        resp = client.post(f"{ENDPOINT}/v1/chat/completions", json=payload, timeout=120)
        latency = (time.time() - t0) * 1000
        resp.raise_for_status()
        return resp.json(), latency
    except Exception as e:
        latency = (time.time() - t0) * 1000
        print(f"    ERROR: {e}", file=sys.stderr)
        return None, latency


def evaluate_response(response, tc):
    if response is None:
        return {"pass": False, "reason": "error", "tool_calls": 0, "tools_called": []}

    message = response["choices"][0]["message"]
    tool_calls = message.get("tool_calls", [])
    tools_called = [tc_item["function"]["name"] for tc_item in tool_calls] if tool_calls else []
    num_calls = len(tool_calls)

    if tc["min_calls"] == 0:
        if num_calls == 0:
            return {"pass": True, "reason": "correctly_no_tool", "tool_calls": 0, "tools_called": []}
        else:
            return {"pass": False, "reason": "should_not_call_tool", "tool_calls": num_calls, "tools_called": tools_called}

    if num_calls < tc["min_calls"]:
        return {"pass": False, "reason": f"too_few_calls ({num_calls} < {tc['min_calls']})", "tool_calls": num_calls, "tools_called": tools_called}

    expected = set(tc["expected_tools"])
    called = set(tools_called)
    missing = expected - called
    if missing and len(called) == 0:
        return {"pass": False, "reason": "no_tool_calls", "tool_calls": 0, "tools_called": []}
    elif missing:
        return {"pass": False, "reason": f"missing_tools: {missing}", "tool_calls": num_calls, "tools_called": tools_called}

    if "expected_params" in tc and tool_calls:
        fn_args = json.loads(tool_calls[0]["function"]["arguments"])
        for key, val in tc["expected_params"].items():
            actual = fn_args.get(key)
            if actual is None:
                return {"pass": False, "reason": f"missing_param: {key}", "tool_calls": num_calls, "tools_called": tools_called}
            if isinstance(val, str) and str(actual).lower() != val.lower():
                return {"pass": False, "reason": f"wrong_param: {key}={actual} (expected {val})", "tool_calls": num_calls, "tools_called": tools_called}
            if isinstance(val, (int, float)):
                try:
                    if abs(float(actual) - float(val)) > 0.01:
                        return {"pass": False, "reason": f"wrong_param: {key}={actual} (expected {val})", "tool_calls": num_calls, "tools_called": tools_called}
                except (ValueError, TypeError):
                    return {"pass": False, "reason": f"wrong_param_type: {key}={actual}", "tool_calls": num_calls, "tools_called": tools_called}

    return {"pass": True, "reason": "all_correct", "tool_calls": num_calls, "tools_called": tools_called}


def run_model(client, model_name, model_id):
    results = []
    for tc in HARD_QUERIES:
        response, latency = call_model(client, model_id, tc["query"])
        result = evaluate_response(response, tc)
        result["id"] = tc["id"]
        result["query"] = tc["query"]
        result["category"] = tc["category"]
        result["expected_tools"] = tc["expected_tools"]
        result["latency_ms"] = latency
        results.append(result)

        status = "PASS" if result["pass"] else "FAIL"
        print(f"  [{tc['id']:2d}] {status:4s} | {tc['category']:14s} | calls={result['tool_calls']} | {result['reason'][:50]}")
    return results


def main():
    client = httpx.Client(verify=False, timeout=120)

    all_results = {}
    for model_name, model_id in MODELS.items():
        print(f"\n{'='*70}")
        print(f"  EVALUATING: {model_name} ({model_id})")
        print(f"  {len(HARD_QUERIES)} hard queries")
        print(f"{'='*70}\n")
        all_results[model_name] = run_model(client, model_name, model_id)

    client.close()

    # Comparison
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Base vs Fine-Tuned LoRA")
    print(f"{'='*70}\n")

    for model_name in MODELS:
        results = all_results[model_name]
        passed = sum(1 for r in results if r["pass"])
        total = len(results)
        print(f"  {model_name:10s}: {passed}/{total} ({passed/total*100:.0f}%)")

    # Category breakdown
    print(f"\n  {'Category':<16} {'Base':>8} {'LoRA':>8} {'Delta':>8}")
    print(f"  {'-'*44}")
    categories = sorted(set(tc["category"] for tc in HARD_QUERIES))
    for cat in categories:
        base_pass = sum(1 for r in all_results["base"] if r["category"] == cat and r["pass"])
        lora_pass = sum(1 for r in all_results["lora"] if r["category"] == cat and r["pass"])
        base_total = sum(1 for r in all_results["base"] if r["category"] == cat)
        base_pct = base_pass / base_total * 100
        lora_pct = lora_pass / base_total * 100
        delta = lora_pct - base_pct
        delta_str = f"+{delta:.0f}%" if delta > 0 else f"{delta:.0f}%"
        print(f"  {cat:<16} {base_pct:>7.0f}% {lora_pct:>7.0f}% {delta_str:>8}")

    # Per-query comparison
    print(f"\n  QUERIES WHERE MODELS DIFFER:")
    print(f"  {'ID':>4} {'Category':<16} {'Base':>6} {'LoRA':>6} {'Detail'}")
    print(f"  {'-'*70}")
    for i in range(len(HARD_QUERIES)):
        base_r = all_results["base"][i]
        lora_r = all_results["lora"][i]
        if base_r["pass"] != lora_r["pass"]:
            base_s = "PASS" if base_r["pass"] else "FAIL"
            lora_s = "PASS" if lora_r["pass"] else "FAIL"
            detail = lora_r["reason"] if not lora_r["pass"] else base_r["reason"]
            print(f"  {base_r['id']:>4} {base_r['category']:<16} {base_s:>6} {lora_s:>6} {detail[:40]}")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endpoint": ENDPOINT,
        "num_queries": len(HARD_QUERIES),
        "base_model": MODELS["base"],
        "lora_model": MODELS["lora"],
        "base_results": all_results["base"],
        "lora_results": all_results["lora"],
        "summary": {
            "base_pass": sum(1 for r in all_results["base"] if r["pass"]),
            "lora_pass": sum(1 for r in all_results["lora"] if r["pass"]),
            "total": len(HARD_QUERIES),
        }
    }
    output_path = "/Users/raghurambanda/workspace/rhoai/end-to-end-examples/tool-calling-financial/examples/eval/comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
