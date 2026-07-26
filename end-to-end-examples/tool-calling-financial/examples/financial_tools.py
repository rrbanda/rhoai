"""Financial tool wrappers for the Deep Agent.

Each function wraps a call to the FinanceInsights MCP server.
These are passed to ``create_deep_agent(tools=[...])`` and Deep Agents
infers the schema from function signatures and docstrings automatically.

The MCP server URL is read from the MCP_SERVER_URL environment variable
(default: http://localhost:8009/mcp).
"""

from __future__ import annotations

import json
import os
from typing import Optional

import httpx
from langchain_core.tools import tool

_MCP_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8009/mcp")
_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=30)
    return _client


def _mcp_session_id() -> str | None:
    return getattr(_mcp_session_id, "_sid", None)


def _call_mcp(tool_name: str, arguments: dict) -> dict:
    """Low-level call to the MCP server via Streamable HTTP.

    Handles session initialization on first call.
    """
    client = _get_client()

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    sid = _mcp_session_id()
    if sid:
        headers["Mcp-Session-Id"] = sid

    if not sid:
        init_resp = client.post(
            _MCP_URL,
            json={
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "tool-calling-financial", "version": "1.0"},
                },
            },
            headers=headers,
        )
        if init_resp.status_code == 200:
            new_sid = init_resp.headers.get("Mcp-Session-Id")
            if new_sid:
                _mcp_session_id._sid = new_sid
                headers["Mcp-Session-Id"] = new_sid

            client.post(
                _MCP_URL,
                json={
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                },
                headers=headers,
            )

    resp = client.post(
        _MCP_URL,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        },
        headers=headers,
    )
    data = resp.json()
    content = data.get("result", {}).get("content", [])
    if content and content[0].get("type") == "text":
        return json.loads(content[0]["text"])
    return data


# ---------------------------------------------------------------------------
# Cluster 1 — Market Data
# ---------------------------------------------------------------------------
@tool
def get_stock_quote(ticker: str) -> dict:
    """Get a real-time price quote for a single stock.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, JPM, MSFT)

    Returns current price, volume, daily change, and market cap.
    """
    return _call_mcp("get_stock_quote", {"ticker": ticker})


@tool
def get_market_summary() -> dict:
    """Get a snapshot of broad market indices (S&P 500, DJIA, NASDAQ, Russell 2000).

    Takes no parameters. Use for overall market conditions.
    """
    return _call_mcp("get_market_summary", {})


@tool
def get_historical_prices(
    ticker: str,
    date_from: str,
    date_to: str,
    interval: str = "daily",
) -> dict:
    """Get historical OHLCV price data for a stock over a date range.

    Args:
        ticker: Stock ticker symbol
        date_from: Start date in YYYY-MM-DD format
        date_to: End date in YYYY-MM-DD format
        interval: Price interval — daily, weekly, or monthly
    """
    return _call_mcp("get_historical_prices", {
        "ticker": ticker,
        "date_from": date_from,
        "date_to": date_to,
        "interval": interval,
    })


@tool
def screen_stocks(
    sector: Optional[str] = None,
    min_market_cap: Optional[int] = None,
    max_pe_ratio: Optional[float] = None,
    min_dividend_yield: Optional[float] = None,
    min_volume: Optional[int] = None,
    sort_by: str = "market_cap",
    limit: int = 20,
) -> dict:
    """Screen stocks by sector, valuation, dividend yield, and volume criteria.

    Args:
        sector: Filter by sector (e.g. Technology, Healthcare, Financial Services)
        min_market_cap: Minimum market cap in dollars
        max_pe_ratio: Maximum P/E ratio
        min_dividend_yield: Minimum dividend yield percentage
        min_volume: Minimum daily trading volume
        sort_by: Sort order — market_cap, pe_ratio, dividend_yield, volume, price
        limit: Max results to return (1-50)
    """
    args = {"sort_by": sort_by, "limit": limit}
    if sector:
        args["sector"] = sector
    if min_market_cap is not None:
        args["min_market_cap"] = min_market_cap
    if max_pe_ratio is not None:
        args["max_pe_ratio"] = max_pe_ratio
    if min_dividend_yield is not None:
        args["min_dividend_yield"] = min_dividend_yield
    if min_volume is not None:
        args["min_volume"] = min_volume
    return _call_mcp("screen_stocks", args)


# ---------------------------------------------------------------------------
# Cluster 2 — Portfolio Management
# ---------------------------------------------------------------------------
@tool
def get_portfolio_positions(portfolio_id: str) -> dict:
    """Get current holdings for a portfolio with unrealized P&L.

    Args:
        portfolio_id: Portfolio identifier (e.g. PORT-0001)
    """
    return _call_mcp("get_portfolio_positions", {"portfolio_id": portfolio_id})


@tool
def get_portfolio_performance(portfolio_id: str, period: str = "1m") -> dict:
    """Get time-weighted returns for a portfolio over a period.

    Args:
        portfolio_id: Portfolio identifier (e.g. PORT-0001)
        period: Performance period — 1d, 1w, 1m, 3m, 6m, 1y, ytd
    """
    return _call_mcp("get_portfolio_performance", {
        "portfolio_id": portfolio_id,
        "period": period,
    })


@tool
def get_account_summary(portfolio_id: str) -> dict:
    """Get a high-level account overview — total value, cash, margin, buying power.

    Args:
        portfolio_id: Portfolio identifier (e.g. PORT-0001)
    """
    return _call_mcp("get_account_summary", {"portfolio_id": portfolio_id})


@tool
def get_transaction_history(
    portfolio_id: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    action: Optional[str] = None,
    ticker: Optional[str] = None,
) -> dict:
    """Get past trades, dividends, and fees for a portfolio.

    Args:
        portfolio_id: Portfolio identifier (e.g. PORT-0001)
        date_from: Start date filter (YYYY-MM-DD)
        date_to: End date filter (YYYY-MM-DD)
        action: Filter by action — buy, sell, dividend, fee
        ticker: Filter by ticker symbol
    """
    args: dict = {"portfolio_id": portfolio_id}
    if date_from:
        args["date_from"] = date_from
    if date_to:
        args["date_to"] = date_to
    if action:
        args["action"] = action
    if ticker:
        args["ticker"] = ticker
    return _call_mcp("get_transaction_history", args)


# ---------------------------------------------------------------------------
# Cluster 3 — Risk & Analytics
# ---------------------------------------------------------------------------
@tool
def calculate_portfolio_risk(portfolio_id: str) -> dict:
    """Calculate risk metrics for a portfolio — VaR, Sharpe ratio, beta, volatility, max drawdown.

    Args:
        portfolio_id: Portfolio identifier (e.g. PORT-0001)
    """
    return _call_mcp("calculate_portfolio_risk", {"portfolio_id": portfolio_id})


@tool
def get_sector_exposure(portfolio_id: str) -> dict:
    """Get sector allocation breakdown for a portfolio with concentration warnings.

    Args:
        portfolio_id: Portfolio identifier (e.g. PORT-0001)
    """
    return _call_mcp("get_sector_exposure", {"portfolio_id": portfolio_id})


@tool
def run_stress_test(portfolio_id: str, scenario: str) -> dict:
    """Run a hypothetical stress test on a portfolio.

    Args:
        portfolio_id: Portfolio identifier (e.g. PORT-0001)
        scenario: Stress scenario — market_crash_2008, rate_hike_300bps,
                  tech_selloff_30pct, oil_shock, pandemic_lockdown
    """
    return _call_mcp("run_stress_test", {
        "portfolio_id": portfolio_id,
        "scenario": scenario,
    })


@tool
def analyze_stock(ticker: str) -> dict:
    """Run fundamental and technical analysis on a single stock.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, JPM)

    Returns valuation, technicals, support/resistance, and sector comparison.
    """
    return _call_mcp("analyze_stock", {"ticker": ticker})


# ---------------------------------------------------------------------------
# Cluster 4 — Trading & Compliance
# ---------------------------------------------------------------------------
@tool
def submit_trade_order(
    portfolio_id: str,
    ticker: str,
    action: str,
    shares: int,
    order_type: str = "market",
    limit_price: Optional[float] = None,
) -> dict:
    """Place a trade order with pre-trade compliance validation.

    Args:
        portfolio_id: Portfolio identifier (e.g. PORT-0001)
        ticker: Stock ticker symbol to trade
        action: Trade direction — buy or sell
        shares: Number of shares to trade
        order_type: Order type — market, limit, stop
        limit_price: Limit price (required for limit/stop orders)
    """
    args = {
        "portfolio_id": portfolio_id,
        "ticker": ticker,
        "action": action,
        "shares": shares,
        "order_type": order_type,
    }
    if limit_price is not None:
        args["limit_price"] = limit_price
    return _call_mcp("submit_trade_order", args)


@tool
def check_compliance(
    portfolio_id: str,
    ticker: str,
    action: str,
    shares: int,
) -> dict:
    """Run pre-trade compliance validation without placing an order.

    Args:
        portfolio_id: Portfolio identifier (e.g. PORT-0001)
        ticker: Stock ticker symbol to check
        action: Proposed action — buy or sell
        shares: Number of shares in proposed trade
    """
    return _call_mcp("check_compliance", {
        "portfolio_id": portfolio_id,
        "ticker": ticker,
        "action": action,
        "shares": shares,
    })


@tool
def get_regulatory_status(portfolio_id: str) -> dict:
    """Get KYC/AML and regulatory status for a portfolio's client.

    Args:
        portfolio_id: Portfolio identifier (e.g. PORT-0001)
    """
    return _call_mcp("get_regulatory_status", {"portfolio_id": portfolio_id})


ALL_TOOLS = [
    get_stock_quote,
    get_market_summary,
    get_historical_prices,
    screen_stocks,
    get_portfolio_positions,
    get_portfolio_performance,
    get_account_summary,
    get_transaction_history,
    calculate_portfolio_risk,
    get_sector_exposure,
    run_stress_test,
    analyze_stock,
    submit_trade_order,
    check_compliance,
    get_regulatory_status,
]
