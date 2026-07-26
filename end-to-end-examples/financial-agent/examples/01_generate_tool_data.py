"""Generate tool-use training data via SDG Hub's MCP distillation flow.

This script runs the MCP Server Distillation pipeline for the FinanceInsights
Advisory Platform end-to-end:
  1. Loads the built-in distillation flow from SDG Hub
  2. Configures teacher LLM and Langflow agent connections
  3. Creates an input dataset from 15 financial MCP server tool schemas
  4. Runs the full pipeline (exploration -> question synthesis -> expert trajectories)
  5. Saves the output as a Parquet file for downstream formatting

The pipeline uses a frontier model (via Langflow + MCP server) to actively explore
portfolio management, market data, risk analysis, and trade execution tools, then
a teacher LLM to synthesize and quality-filter the training questions.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nest_asyncio
import pandas as pd
from dotenv import load_dotenv

nest_asyncio.apply()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tool-use training data with SDG Hub MCP distillation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "./generated_data"),
        help="Directory to save generated data (default: ./generated_data)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of question candidates per MCP server row (default: 10)",
    )
    parser.add_argument(
        "--tools-per-question",
        type=int,
        default=2,
        help="Number of tools each question should require (default: 2)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.environ.get("CHECKPOINT_DIR", "./checkpoints"),
        help="Directory for pipeline checkpoints (default: ./checkpoints)",
    )
    return parser.parse_args()


def build_input_dataset() -> pd.DataFrame:
    """Build the input DataFrame with FinanceInsights MCP server tool schemas.

    Contains 15 tools organized across 4 clusters:
      - Market Data: get_stock_quote, get_market_summary, get_historical_prices, screen_stocks
      - Portfolio Management: get_portfolio_positions, get_portfolio_performance,
        get_account_summary, get_transaction_history
      - Risk Analysis: calculate_portfolio_risk, get_sector_exposure, run_stress_test,
        analyze_stock
      - Trade Execution: submit_trade_order, check_compliance, get_regulatory_status
    """
    tool_list = [
        {
            "name": "get_stock_quote",
            "description": "Get a real-time quote for a stock including price, change, volume, and market cap.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. AAPL, MSFT)"},
                },
                "required": ["ticker"],
            },
        },
        {
            "name": "get_market_summary",
            "description": "Get a snapshot of broad market indices including S&P 500, DJIA, NASDAQ, and Russell 2000.",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "get_historical_prices",
            "description": "Retrieve historical price data for a stock over a specified date range and interval.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                    "interval": {
                        "type": "string",
                        "enum": ["daily", "weekly", "monthly"],
                        "description": "Price interval (default: daily)",
                    },
                },
                "required": ["ticker", "date_from", "date_to"],
            },
        },
        {
            "name": "screen_stocks",
            "description": "Screen stocks by sector, valuation, dividend yield, and volume criteria.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sector": {"type": "string", "description": "Filter by sector (e.g. Technology, Healthcare)"},
                    "min_market_cap": {"type": "integer", "description": "Minimum market cap in dollars"},
                    "max_pe_ratio": {"type": "number", "description": "Maximum P/E ratio"},
                    "min_dividend_yield": {"type": "number", "description": "Minimum dividend yield (%)"},
                    "min_volume": {"type": "integer", "description": "Minimum average daily volume"},
                    "sort_by": {
                        "type": "string",
                        "enum": ["market_cap", "pe_ratio", "dividend_yield", "volume", "price"],
                        "description": "Sort results by field",
                    },
                    "limit": {"type": "integer", "description": "Max results to return", "default": 20},
                },
            },
        },
        {
            "name": "get_portfolio_positions",
            "description": "Get all current positions in a portfolio with quantities, cost basis, and current value.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier (e.g. PORT-0001)"},
                },
                "required": ["portfolio_id"],
            },
        },
        {
            "name": "get_portfolio_performance",
            "description": "Get portfolio performance metrics including returns, alpha, beta, and Sharpe ratio.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"},
                    "period": {
                        "type": "string",
                        "enum": ["1d", "1w", "1m", "3m", "6m", "1y", "ytd"],
                        "description": "Performance period (default: 1m)",
                    },
                },
                "required": ["portfolio_id"],
            },
        },
        {
            "name": "get_account_summary",
            "description": "Get account-level summary including cash balance, total equity, margin, and buying power.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"},
                },
                "required": ["portfolio_id"],
            },
        },
        {
            "name": "get_transaction_history",
            "description": "Retrieve transaction history for a portfolio with optional filtering by date, action, or ticker.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"},
                    "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                    "action": {
                        "type": "string",
                        "enum": ["buy", "sell", "dividend", "fee"],
                        "description": "Filter by transaction type",
                    },
                    "ticker": {"type": "string", "description": "Filter by ticker symbol"},
                },
                "required": ["portfolio_id"],
            },
        },
        {
            "name": "calculate_portfolio_risk",
            "description": "Calculate risk metrics for a portfolio including VaR, volatility, max drawdown, and beta.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"},
                },
                "required": ["portfolio_id"],
            },
        },
        {
            "name": "get_sector_exposure",
            "description": "Get portfolio allocation breakdown by sector with concentration percentages.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"},
                },
                "required": ["portfolio_id"],
            },
        },
        {
            "name": "run_stress_test",
            "description": "Run a stress test scenario on a portfolio to estimate potential losses under adverse conditions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"},
                    "scenario": {
                        "type": "string",
                        "enum": ["market_crash_2008", "rate_hike_300bps", "tech_selloff_30pct", "oil_shock", "pandemic_lockdown"],
                        "description": "Stress test scenario to simulate",
                    },
                },
                "required": ["portfolio_id", "scenario"],
            },
        },
        {
            "name": "analyze_stock",
            "description": "Run comprehensive analysis on a stock including fundamentals, technicals, and analyst consensus.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                },
                "required": ["ticker"],
            },
        },
        {
            "name": "submit_trade_order",
            "description": "Submit a buy or sell order for a stock in a portfolio.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"},
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "action": {
                        "type": "string",
                        "enum": ["buy", "sell"],
                        "description": "Trade direction",
                    },
                    "shares": {"type": "integer", "description": "Number of shares"},
                    "order_type": {
                        "type": "string",
                        "enum": ["market", "limit", "stop"],
                        "description": "Order type (default: market)",
                    },
                    "limit_price": {"type": "number", "description": "Limit price (required for limit/stop_limit orders)"},
                },
                "required": ["portfolio_id", "ticker", "action", "shares"],
            },
        },
        {
            "name": "check_compliance",
            "description": "Check whether a proposed trade complies with portfolio constraints and regulatory rules.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"},
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "action": {
                        "type": "string",
                        "enum": ["buy", "sell"],
                        "description": "Proposed trade direction",
                    },
                    "shares": {"type": "integer", "description": "Proposed number of shares"},
                },
                "required": ["portfolio_id", "ticker", "action", "shares"],
            },
        },
        {
            "name": "get_regulatory_status",
            "description": "Get the regulatory compliance status of a portfolio including any active restrictions or flags.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "portfolio_id": {"type": "string", "description": "Portfolio identifier"},
                },
                "required": ["portfolio_id"],
            },
        },
    ]

    return pd.DataFrame(
        {
            "tool_list": [tool_list],
            "mcp_server_name": ["FinanceInsights Advisory Platform"],
            "mcp_server_description": [
                "Wealth management platform for financial advisors and portfolio managers. "
                "Provides portfolio management, real-time market data, risk analysis, "
                "and trade execution capabilities. "
                "Features 15 tools organized across market data, portfolio management, "
                "risk analysis, and trade execution clusters."
            ],
        }
    )


def main() -> None:
    load_dotenv()
    args = parse_args()

    teacher_model = os.environ.get("TEACHER_MODEL", "openai/gpt-5.2")
    teacher_api_key = os.environ.get("TEACHER_API_KEY")
    langflow_url = os.environ.get("LANGFLOW_URL")
    langflow_api_key = os.environ.get("LANGFLOW_API_KEY")

    if not teacher_api_key:
        print("ERROR: TEACHER_API_KEY is required. Set it in .env or as an env var.")
        sys.exit(1)
    if not langflow_url:
        print("ERROR: LANGFLOW_URL is required. Set it in .env or as an env var.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -- Load the distillation flow -------------------------------------------
    from sdg_hub import Flow, FlowRegistry

    FlowRegistry.discover_flows()
    flow_yaml = FlowRegistry.get_flow_path("MCP Server Distillation")
    if flow_yaml is None:
        print("ERROR: MCP Server Distillation flow not found in registry.")
        print("Ensure sdg-hub is installed: pip install sdg-hub[examples]")
        sys.exit(1)
    print(f"Loading flow from: {flow_yaml}")
    flow = Flow.from_yaml(flow_yaml)
    print(f"  Flow: {flow.metadata.name}")
    print(f"  Blocks: {len(flow.blocks)}")

    # -- Configure teacher model (question synthesis + quality scoring) --------
    flow.set_model_config(model=teacher_model, api_key=teacher_api_key)
    print(f"  Teacher model: {teacher_model}")

    # -- Configure Langflow agent (frontier model + MCP server) ---------------
    agent_kwargs = {
        "agent_framework": "langflow",
        "agent_url": langflow_url,
    }
    if langflow_api_key:
        agent_kwargs["agent_api_key"] = langflow_api_key

    flow.set_agent_config(**agent_kwargs)
    print(f"  Langflow URL: {langflow_url}")

    # Longer timeout for the exploration block (it calls many tools)
    flow.set_agent_config(timeout=300, blocks=["explore_server"])

    # -- Build input dataset --------------------------------------------------
    dataset = build_input_dataset()
    print(f"\nInput dataset: {len(dataset)} row(s), {len(dataset['tool_list'].iloc[0])} tools")

    # -- Run the pipeline -----------------------------------------------------
    print(f"\nRunning pipeline (num_samples={args.num_samples}, "
          f"tools_per_question={args.tools_per_question})...")
    print("-" * 60)

    result = flow.generate(
        dataset,
        runtime_params={
            "multiply_tool_rows": {"num_samples": args.num_samples},
            "sample_tools": {"num_samples": args.tools_per_question},
        },
        checkpoint_dir=str(checkpoint_dir),
    )

    # -- Save results ---------------------------------------------------------
    if hasattr(result, "to_pandas"):
        result_df = result.to_pandas()
    else:
        result_df = result

    output_path = output_dir / "distillation_output.parquet"
    result_df.to_parquet(output_path, index=False)

    print("-" * 60)
    print(f"Pipeline complete!")
    print(f"  Generated examples: {len(result_df)}")
    print(f"  Output columns: {list(result_df.columns)}")
    print(f"  Saved to: {output_path}")

    # Print quality distribution if available
    for col in ["question_quality_rating", "completeness_rating"]:
        if col in result_df.columns:
            print(f"\n  {col}:")
            for val, count in result_df[col].value_counts().items():
                print(f"    {val}: {count}")


if __name__ == "__main__":
    main()
