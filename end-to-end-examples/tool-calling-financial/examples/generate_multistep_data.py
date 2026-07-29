#!/usr/bin/env python3
"""Generate multi-step tool-calling training data programmatically.

Creates 150 training examples in the same JSONL format as the SDG Hub
distillation pipeline output, targeting patterns where the base model fails:
  - Sequential multi-tool chains (do X then Y)
  - Temporal parameter inference (this year → ytd, last quarter → 3m)
  - Implicit/ambiguous tool routing
  - Parallel multi-tool calls

No Langflow or teacher model required — uses template-based generation
with randomized parameters for diversity.
"""

import json
import random
from pathlib import Path

random.seed(42)

TOOL_DECLARATIONS = [
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

SYSTEM_MSG = "You are a helpful financial assistant with access to the following tools:\n\n" + json.dumps(TOOL_DECLARATIONS, indent=2)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "CRM", "INTC", "ORCL", "ADBE", "PYPL", "SQ"]
PORTFOLIOS = ["GROWTH-1", "BALANCED-1", "CONSERVATIVE-2", "AGGRESSIVE-1", "IRA-2024", "TECH-FUND", "INCOME-1", "PENSION-2024", "MAIN-ACCOUNT", "RET-2024", "MEME-PORT", "ESG-FUND"]
SECTORS = ["Technology", "Healthcare", "Energy", "Financial", "Consumer", "Industrial", "Real Estate", "Utilities"]
SCENARIOS = ["market_crash", "rate_hike", "recession", "tech_selloff", "inflation_spike"]
PERIODS_MAP = {"this year": "ytd", "year to date": "ytd", "this month": "1m", "this week": "1w", "today": "1d", "last 3 months": "3m", "last quarter": "3m", "last 6 months": "6m", "the past year": "1y"}


def make_tool_response(tool_name, args):
    """Generate a realistic mock tool response."""
    if tool_name == "check_compliance":
        return json.dumps({"compliant": True, "warnings": [], "message": f"Trade approved: {args.get('action', 'buy')} {args.get('quantity', 0)} shares of {args.get('ticker', 'N/A')} in {args.get('portfolio_id', 'N/A')}"})
    elif tool_name == "get_stock_quote":
        price = round(random.uniform(50, 500), 2)
        return json.dumps({"ticker": args.get("ticker", ""), "price": price, "change": round(random.uniform(-5, 5), 2), "volume": random.randint(1000000, 50000000)})
    elif tool_name == "get_portfolio_performance":
        return json.dumps({"portfolio_id": args.get("portfolio_id", ""), "period": args.get("period", "1m"), "return_pct": round(random.uniform(-5, 15), 2), "alpha": round(random.uniform(-1, 3), 2), "sharpe": round(random.uniform(0.5, 2.5), 2)})
    elif tool_name == "calculate_portfolio_risk":
        return json.dumps({"portfolio_id": args.get("portfolio_id", ""), "var_95": round(random.uniform(2, 8), 2), "volatility": round(random.uniform(10, 30), 2), "max_drawdown": round(random.uniform(5, 25), 2), "beta": round(random.uniform(0.5, 1.5), 2)})
    elif tool_name == "get_sector_exposure":
        return json.dumps({"portfolio_id": args.get("portfolio_id", ""), "sectors": {"Technology": 35, "Healthcare": 20, "Financial": 15, "Energy": 10, "Consumer": 10, "Other": 10}})
    elif tool_name == "get_portfolio_positions":
        return json.dumps({"portfolio_id": args.get("portfolio_id", ""), "positions": [{"ticker": "AAPL", "shares": 100, "value": 19500}, {"ticker": "MSFT", "shares": 50, "value": 21000}], "total_value": 125000})
    elif tool_name == "get_account_summary":
        return json.dumps({"portfolio_id": args.get("portfolio_id", ""), "cash": round(random.uniform(5000, 100000), 2), "equity": round(random.uniform(50000, 500000), 2), "buying_power": round(random.uniform(10000, 200000), 2)})
    elif tool_name == "submit_trade_order":
        return json.dumps({"order_id": f"ORD-{random.randint(10000, 99999)}", "status": "filled", "filled_price": round(random.uniform(50, 500), 2)})
    elif tool_name == "run_stress_test":
        return json.dumps({"portfolio_id": args.get("portfolio_id", ""), "scenario": args.get("scenario", ""), "estimated_loss_pct": round(random.uniform(5, 35), 2), "worst_case_value": round(random.uniform(50000, 200000), 2)})
    elif tool_name == "analyze_stock":
        return json.dumps({"ticker": args.get("ticker", ""), "rating": random.choice(["buy", "hold", "sell"]), "target_price": round(random.uniform(100, 600), 2), "pe_ratio": round(random.uniform(10, 50), 1)})
    elif tool_name == "get_historical_prices":
        return json.dumps({"ticker": args.get("ticker", ""), "period": args.get("period", "1mo"), "data_points": 30, "latest_close": round(random.uniform(50, 500), 2)})
    elif tool_name == "screen_stocks":
        return json.dumps({"results": [{"ticker": random.choice(TICKERS), "price": round(random.uniform(50, 300), 2)} for _ in range(3)], "total": 15})
    elif tool_name == "get_market_summary":
        return json.dumps({"sp500": round(random.uniform(4500, 5500), 2), "nasdaq": round(random.uniform(14000, 17000), 2), "trend": random.choice(["bullish", "neutral", "bearish"])})
    elif tool_name == "get_transaction_history":
        return json.dumps({"portfolio_id": args.get("portfolio_id", ""), "transactions": [{"date": "2026-06-15", "type": args.get("transaction_type", "buy"), "ticker": "AAPL", "amount": 5000}]})
    elif tool_name == "get_regulatory_status":
        return json.dumps({"portfolio_id": args.get("portfolio_id", ""), "status": "compliant", "restrictions": []})
    return json.dumps({"result": "ok"})


def make_example(user_query, tool_calls_sequence, final_answer):
    """Build a complete training example in the expected JSONL format."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_query},
    ]

    for calls_in_turn in tool_calls_sequence:
        tool_calls_msg = []
        for (tool_name, args) in calls_in_turn:
            tool_calls_msg.append({
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(args)},
            })
        messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls_msg})

        for (tool_name, args) in calls_in_turn:
            messages.append({"role": "tool", "content": make_tool_response(tool_name, args), "name": tool_name})

    messages.append({"role": "assistant", "content": final_answer})
    return {"messages": messages}


def generate_sequential_examples():
    """Generate examples where the model must call tool A THEN tool B."""
    examples = []

    # Pattern: check_compliance → submit_trade_order
    for _ in range(15):
        portfolio = random.choice(PORTFOLIOS)
        ticker = random.choice(TICKERS)
        action = random.choice(["buy", "sell"])
        qty = random.choice([25, 50, 75, 100, 150, 200, 500, 1000])
        order_type = random.choice(["market", "limit", "stop"])
        price = round(random.uniform(100, 500), 2) if order_type != "market" else None

        queries = [
            f"I want to {action} {qty} shares of {ticker} in {portfolio}. Check compliance first, then submit the {order_type} order.",
            f"Verify compliance and then {action} {qty} {ticker} shares in my {portfolio} portfolio at {order_type} price.",
            f"First check if I can {action} {qty} shares of {ticker} in {portfolio}, and if it's allowed, go ahead and place the order.",
            f"Please {action} {qty} shares of {ticker} from {portfolio}. Make sure to verify compliance before submitting.",
            f"Can I {action} {qty} of {ticker} in {portfolio}? If yes, submit the trade as a {order_type} order.",
        ]

        compliance_args = {"portfolio_id": portfolio, "ticker": ticker, "action": action, "quantity": qty}
        trade_args = {"portfolio_id": portfolio, "ticker": ticker, "action": action, "quantity": qty, "order_type": order_type}
        if price:
            trade_args["limit_price"] = price

        examples.append(make_example(
            random.choice(queries),
            [[("check_compliance", compliance_args), ("submit_trade_order", trade_args)]],
            f"Done! The compliance check passed and I've submitted your {order_type} order to {action} {qty} shares of {ticker} in {portfolio}."
        ))

    # Pattern: get_portfolio_positions → get_sector_exposure (related)
    for _ in range(10):
        portfolio = random.choice(PORTFOLIOS)
        queries = [
            f"Show me all positions in {portfolio} and also the sector breakdown.",
            f"I want to see what I own in {portfolio} and how it's distributed across sectors.",
            f"Get my {portfolio} holdings and sector exposure.",
            f"What's in my {portfolio} portfolio? Also show sector concentration.",
            f"List positions and sector allocation for {portfolio}.",
        ]
        examples.append(make_example(
            random.choice(queries),
            [[("get_portfolio_positions", {"portfolio_id": portfolio}), ("get_sector_exposure", {"portfolio_id": portfolio})]],
            f"Here's your {portfolio} portfolio: you hold positions across multiple sectors with the largest concentration in Technology at 35%."
        ))

    # Pattern: analyze_stock → submit_trade (research then act)
    for _ in range(10):
        portfolio = random.choice(PORTFOLIOS)
        ticker = random.choice(TICKERS)
        qty = random.choice([25, 50, 100])
        queries = [
            f"Analyze {ticker} and if it looks good, buy {qty} shares in {portfolio}.",
            f"Do a full analysis on {ticker}, then purchase {qty} shares in my {portfolio} portfolio.",
            f"Research {ticker} and then submit a buy order for {qty} shares in {portfolio}.",
        ]
        examples.append(make_example(
            random.choice(queries),
            [[("analyze_stock", {"ticker": ticker}), ("submit_trade_order", {"portfolio_id": portfolio, "ticker": ticker, "action": "buy", "quantity": qty, "order_type": "market"})]],
            f"Analysis complete: {ticker} shows positive signals. I've submitted a market order to buy {qty} shares in {portfolio}."
        ))

    # Pattern: get_performance → calculate_risk (full review)
    for _ in range(10):
        portfolio = random.choice(PORTFOLIOS)
        period = random.choice(["ytd", "1m", "3m", "6m", "1y"])
        queries = [
            f"Give me a full review of {portfolio}: performance and risk metrics.",
            f"How is {portfolio} doing? Show me returns and risk analysis.",
            f"I need the performance numbers and risk metrics for {portfolio}.",
            f"Review {portfolio} — what are the returns and how risky is it?",
        ]
        examples.append(make_example(
            random.choice(queries),
            [[("get_portfolio_performance", {"portfolio_id": portfolio, "period": period}), ("calculate_portfolio_risk", {"portfolio_id": portfolio})]],
            f"Your {portfolio} portfolio returned well this period with moderate risk levels. The Sharpe ratio indicates decent risk-adjusted returns."
        ))

    # Pattern: 4+ tool comprehensive review
    for _ in range(8):
        portfolio = random.choice(PORTFOLIOS)
        scenario = random.choice(SCENARIOS)
        queries = [
            f"Do a complete health check on {portfolio}: positions, performance, risk, sector exposure, and a {scenario} stress test.",
            f"Full portfolio review for {portfolio}: I need positions, YTD performance, risk analysis, sector breakdown, and run a {scenario} scenario.",
            f"Give me everything on {portfolio} — holdings, returns, risk, diversification, and stress test with {scenario}.",
        ]
        examples.append(make_example(
            random.choice(queries),
            [[
                ("get_portfolio_positions", {"portfolio_id": portfolio}),
                ("get_portfolio_performance", {"portfolio_id": portfolio, "period": "ytd"}),
                ("calculate_portfolio_risk", {"portfolio_id": portfolio}),
                ("get_sector_exposure", {"portfolio_id": portfolio}),
                ("run_stress_test", {"portfolio_id": portfolio, "scenario": scenario}),
            ]],
            f"Complete review of {portfolio}: Your portfolio is well-diversified with moderate risk. Under a {scenario} scenario, estimated losses would be contained."
        ))

    return examples


def generate_temporal_examples():
    """Generate examples with natural language time references → correct period params."""
    examples = []

    temporal_phrases = [
        ("this year", "ytd"), ("year to date", "ytd"), ("so far this year", "ytd"),
        ("YTD", "ytd"), ("since January", "ytd"), ("in 2026", "ytd"),
        ("this month", "1m"), ("the past month", "1m"), ("last 30 days", "1m"),
        ("this week", "1w"), ("the past week", "1w"), ("last 7 days", "1w"),
        ("today", "1d"), ("today's session", "1d"),
        ("last quarter", "3m"), ("the past 3 months", "3m"), ("past quarter", "3m"),
        ("last 6 months", "6m"), ("the past half year", "6m"),
    ]

    for phrase, period in temporal_phrases:
        portfolio = random.choice(PORTFOLIOS)
        queries = [
            f"How has {portfolio} performed {phrase}?",
            f"Show me {portfolio} returns {phrase}.",
            f"What are the performance numbers for {portfolio} {phrase}?",
            f"Get me the {phrase} performance for {portfolio}.",
        ]
        examples.append(make_example(
            random.choice(queries),
            [[("get_portfolio_performance", {"portfolio_id": portfolio, "period": period})]],
            f"Your {portfolio} portfolio has returned well over the {phrase} period."
        ))

    # Additional: "how much am I making" type queries
    for _ in range(10):
        portfolio = random.choice(PORTFOLIOS)
        queries = [
            f"How much money am I making on {portfolio}?",
            f"What's my return on {portfolio}?",
            f"Am I making or losing money in {portfolio}?",
            f"How's {portfolio} doing for me?",
            f"What's the profit/loss on my {portfolio} portfolio?",
            f"Is {portfolio} making money?",
        ]
        examples.append(make_example(
            random.choice(queries),
            [[("get_portfolio_performance", {"portfolio_id": portfolio, "period": "ytd"})]],
            f"Your {portfolio} portfolio is performing positively with solid returns year to date."
        ))

    return examples


def generate_implicit_examples():
    """Generate examples where the intent is ambiguous but should trigger a tool."""
    examples = []

    # "Am I diversified?" → get_sector_exposure
    for _ in range(8):
        portfolio = random.choice(PORTFOLIOS)
        queries = [
            f"Am I diversified enough in {portfolio}?",
            f"Is {portfolio} too concentrated in one sector?",
            f"How spread out is my {portfolio} portfolio?",
            f"Do I have concentration risk in {portfolio}?",
            f"Is my {portfolio} properly diversified?",
        ]
        examples.append(make_example(
            random.choice(queries),
            [[("get_sector_exposure", {"portfolio_id": portfolio})]],
            f"Looking at your {portfolio} sector allocation, you have reasonable diversification though Technology is the largest position at 35%."
        ))

    # "How risky?" → calculate_portfolio_risk
    for _ in range(8):
        portfolio = random.choice(PORTFOLIOS)
        queries = [
            f"How risky is {portfolio}?",
            f"Should I be worried about {portfolio}?",
            f"Is {portfolio} too aggressive?",
            f"What's the risk level of my {portfolio} portfolio?",
            f"Am I taking on too much risk with {portfolio}?",
        ]
        examples.append(make_example(
            random.choice(queries),
            [[("calculate_portfolio_risk", {"portfolio_id": portfolio})]],
            f"Your {portfolio} portfolio has moderate risk with a beta indicating correlation to market movements."
        ))

    # "What would happen if..." → run_stress_test
    for _ in range(8):
        portfolio = random.choice(PORTFOLIOS)
        scenario_phrases = [
            ("the market crashes", "market_crash"),
            ("interest rates spike", "rate_hike"),
            ("we enter a recession", "recession"),
            ("tech stocks tank", "tech_selloff"),
            ("inflation gets out of control", "inflation_spike"),
        ]
        phrase, scenario = random.choice(scenario_phrases)
        queries = [
            f"What would happen to {portfolio} if {phrase}?",
            f"How would {portfolio} hold up if {phrase}?",
            f"If {phrase}, how bad would it be for {portfolio}?",
            f"Stress test {portfolio} — what if {phrase}?",
        ]
        examples.append(make_example(
            random.choice(queries),
            [[("run_stress_test", {"portfolio_id": portfolio, "scenario": scenario})]],
            f"Under a {scenario} scenario, your {portfolio} portfolio would see estimated losses but remain viable."
        ))

    # "What's my cash situation?" → get_account_summary
    for _ in range(6):
        portfolio = random.choice(PORTFOLIOS)
        queries = [
            f"What's my cash situation in {portfolio}?",
            f"How much buying power do I have in {portfolio}?",
            f"Can I afford more trades in {portfolio}?",
            f"What's available in my {portfolio} account?",
            f"Do I have cash to deploy in {portfolio}?",
        ]
        examples.append(make_example(
            random.choice(queries),
            [[("get_account_summary", {"portfolio_id": portfolio})]],
            f"Your {portfolio} account has available cash and buying power for additional trades."
        ))

    return examples


def generate_parallel_examples():
    """Generate examples requiring parallel tool calls."""
    examples = []

    # Compare multiple stocks
    for _ in range(8):
        tickers = random.sample(TICKERS, random.randint(2, 4))
        ticker_str = ", ".join(tickers[:-1]) + f" and {tickers[-1]}"
        queries = [
            f"Get me quotes for {ticker_str}.",
            f"What are the current prices of {ticker_str}?",
            f"Compare {ticker_str} — show me current prices.",
        ]
        calls = [("get_stock_quote", {"ticker": t}) for t in tickers]
        examples.append(make_example(
            random.choice(queries),
            [calls],
            f"Here are the current prices for {ticker_str}."
        ))

    # Compare multiple portfolios
    for _ in range(6):
        portfolios = random.sample(PORTFOLIOS, 2)
        queries = [
            f"Compare risk metrics for {portfolios[0]} and {portfolios[1]}.",
            f"Which is riskier: {portfolios[0]} or {portfolios[1]}?",
            f"Get risk analysis for both {portfolios[0]} and {portfolios[1]}.",
        ]
        calls = [("calculate_portfolio_risk", {"portfolio_id": p}) for p in portfolios]
        examples.append(make_example(
            random.choice(queries),
            [calls],
            f"Comparing risk: {portfolios[0]} and {portfolios[1]} have different risk profiles."
        ))

    return examples


def main():
    all_examples = []
    all_examples.extend(generate_sequential_examples())
    all_examples.extend(generate_temporal_examples())
    all_examples.extend(generate_implicit_examples())
    all_examples.extend(generate_parallel_examples())

    random.shuffle(all_examples)

    output_path = Path("/Users/raghurambanda/workspace/rhoai/end-to-end-examples/tool-calling-financial/examples/generated_data/multistep_training_data.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {len(all_examples)} training examples")
    print(f"  Sequential/multi-step: {53} examples")
    print(f"  Temporal inference: {len(generate_temporal_examples())} examples")
    print(f"  Implicit routing: {len(generate_implicit_examples())} examples")
    print(f"  Parallel calls: {len(generate_parallel_examples())} examples")
    print(f"Saved to: {output_path}")

    # Also read existing training data and merge
    existing_path = Path("/Users/raghurambanda/workspace/rhoai/end-to-end-examples/tool-calling-financial/examples/sample_data/training_data.jsonl")
    if existing_path.exists():
        existing = []
        with open(existing_path) as f:
            for line in f:
                if line.strip():
                    existing.append(json.loads(line))
        print(f"\nExisting training data: {len(existing)} examples")

        combined = existing + all_examples
        combined_path = output_path.parent / "combined_training_data.jsonl"
        with open(combined_path, "w") as f:
            for ex in combined:
                f.write(json.dumps(ex) + "\n")
        print(f"Combined dataset: {len(combined)} examples")
        print(f"Saved to: {combined_path}")
    else:
        print(f"\nNote: No existing training data found at {existing_path}")
        combined_path = output_path
        print(f"Using generated data only: {output_path}")


if __name__ == "__main__":
    main()
