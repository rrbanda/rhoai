"""FinanceInsights Advisory Platform — FastMCP server with 15 financial tools.

Start the server:
    python server.py

The server exposes a Streamable-HTTP transport on port 8009 (configurable via
the ``PORT`` environment variable) for Langflow / other MCP clients.

Tools are organized into ambiguity clusters to test small-model tool selection:
  - Market Data: get_stock_quote, get_market_summary, get_historical_prices, screen_stocks
  - Portfolio Management: get_portfolio_positions, get_portfolio_performance,
                          get_account_summary, get_transaction_history
  - Risk & Analytics: calculate_portfolio_risk, get_sector_exposure, run_stress_test,
                      analyze_stock
  - Trading & Compliance: submit_trade_order, check_compliance, get_regulatory_status
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Annotated, Any, Optional
import math
import os
import random as _rng
import statistics

from data import DataStore, create_data_store
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="FinanceInsights Advisory Platform",
    instructions=(
        "Wealth management and advisory platform for financial professionals. "
        "Provides real-time market data, portfolio management, risk analysis, "
        "and trade execution with pre-trade compliance checks."
    ),
)

store: DataStore = create_data_store()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _in_range(date_str: str, start: str, end: str) -> bool:
    d = _parse_date(date_str)
    return _parse_date(start) <= d <= _parse_date(end)


def _stock_by_ticker(ticker: str) -> dict[str, Any] | None:
    t = ticker.upper()
    for s in store.stocks:
        if s["ticker"] == t:
            return s
    return None


def _portfolio_by_id(pid: str) -> dict[str, Any] | None:
    for p in store.portfolios:
        if p["portfolio_id"] == pid:
            return p
    return None


def _format_market_cap(mc: int) -> str:
    if mc >= 1_000_000_000_000:
        return f"${mc / 1_000_000_000_000:.2f}T"
    if mc >= 1_000_000_000:
        return f"${mc / 1_000_000_000:.2f}B"
    return f"${mc / 1_000_000:.2f}M"


# =========================================================================
# Cluster 1 — Market Data
# =========================================================================
@mcp.tool()
def get_stock_quote(
    ticker: Annotated[str, "Stock ticker symbol (e.g. 'AAPL', 'JPM')"],
) -> dict[str, Any]:
    """Get real-time price quote for a single stock.

    Returns current price, volume, daily change, and percentage change.
    Use this when the user asks about a specific stock's current price or
    today's performance. For broad market direction, use get_market_summary
    instead. For filtering across many stocks, use screen_stocks.
    """
    stock = _stock_by_ticker(ticker)
    if not stock:
        return {"error": f"Ticker '{ticker}' not found"}

    change = round(stock["current_price"] - stock["prev_close"], 2)
    change_pct = round(change / stock["prev_close"] * 100, 2)

    return {
        "ticker": stock["ticker"],
        "name": stock["name"],
        "exchange": stock["exchange"],
        "current_price": stock["current_price"],
        "prev_close": stock["prev_close"],
        "change": change,
        "change_pct": change_pct,
        "day_high": stock["day_high"],
        "day_low": stock["day_low"],
        "volume": stock["volume"],
        "avg_volume": stock["avg_volume"],
        "market_cap": stock["market_cap"],
        "market_cap_formatted": _format_market_cap(stock["market_cap"]),
    }


@mcp.tool()
def get_market_summary() -> dict[str, Any]:
    """Get a snapshot of broad market indices.

    Returns current values and daily changes for S&P 500, DJIA, NASDAQ
    Composite, and Russell 2000. Takes no parameters — use this for a
    quick overview of overall market conditions. For individual stock
    prices, use get_stock_quote instead.
    """
    return {
        "indices": store.market_indices,
        "market_status": "open",
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@mcp.tool()
def get_historical_prices(
    ticker: Annotated[str, "Stock ticker symbol (e.g. 'AAPL')"],
    date_from: Annotated[str, "Start date (ISO format YYYY-MM-DD)"],
    date_to: Annotated[str, "End date (ISO format YYYY-MM-DD)"],
    interval: Annotated[str, "Price interval: daily, weekly, monthly"] = "daily",
) -> dict[str, Any]:
    """Get historical OHLCV price data for a stock over a date range.

    Returns open, high, low, close, and volume for each period. Use this
    for charting, backtesting, or historical analysis. For today's price
    only, use get_stock_quote instead.
    """
    stock = _stock_by_ticker(ticker)
    if not stock:
        return {"error": f"Ticker '{ticker}' not found"}

    try:
        start = _parse_date(date_from)
        end = _parse_date(date_to)
    except ValueError:
        return {"error": "Invalid date format, use YYYY-MM-DD"}

    if end <= start:
        return {"error": "date_to must be after date_from"}

    rng = _rng.Random(hash(ticker) + hash(date_from))
    base_price = stock["current_price"]
    prices: list[dict[str, Any]] = []
    current = start

    if interval == "weekly":
        step = timedelta(weeks=1)
    elif interval == "monthly":
        step = timedelta(days=30)
    else:
        step = timedelta(days=1)

    while current <= end:
        if current.weekday() >= 5 and interval == "daily":
            current += step
            continue
        drift = rng.gauss(0, 0.015)
        base_price = round(base_price * (1 + drift), 2)
        day_open = round(base_price * rng.uniform(0.995, 1.005), 2)
        day_high = round(max(day_open, base_price) * rng.uniform(1.001, 1.025), 2)
        day_low = round(min(day_open, base_price) * rng.uniform(0.975, 0.999), 2)
        day_volume = rng.randint(
            int(stock["avg_volume"] * 0.5), int(stock["avg_volume"] * 1.8)
        )
        prices.append(
            {
                "date": current.strftime("%Y-%m-%d"),
                "open": day_open,
                "high": day_high,
                "low": day_low,
                "close": base_price,
                "volume": day_volume,
            }
        )
        current += step

    return {
        "ticker": stock["ticker"],
        "name": stock["name"],
        "date_from": date_from,
        "date_to": date_to,
        "interval": interval,
        "data_points": len(prices),
        "prices": prices[:252],
        "note": f"Showing first 252 of {len(prices)} data points"
        if len(prices) > 252
        else None,
    }


@mcp.tool()
def screen_stocks(
    sector: Annotated[
        Optional[str],
        "Filter by sector (e.g. 'Technology', 'Financial Services')",
    ] = None,
    min_market_cap: Annotated[
        Optional[int], "Minimum market cap in dollars"
    ] = None,
    max_pe_ratio: Annotated[
        Optional[float], "Maximum P/E ratio"
    ] = None,
    min_dividend_yield: Annotated[
        Optional[float], "Minimum dividend yield percentage"
    ] = None,
    min_volume: Annotated[
        Optional[int], "Minimum daily volume"
    ] = None,
    sort_by: Annotated[
        str,
        "Sort order: market_cap, pe_ratio, dividend_yield, volume, price",
    ] = "market_cap",
    limit: Annotated[int, "Max results to return (1-50)"] = 20,
) -> dict[str, Any]:
    """Screen and filter stocks from the universe using financial criteria.

    Returns a list of stocks matching all specified filters. Use this when
    the user wants to find stocks meeting certain criteria (e.g. 'tech stocks
    with low P/E'). For a single stock's price, use get_stock_quote.
    """
    results = []
    for s in store.stocks:
        if sector and s["sector"] != sector:
            continue
        if min_market_cap is not None and s["market_cap"] < min_market_cap:
            continue
        if max_pe_ratio is not None and s["pe_ratio"] > max_pe_ratio:
            continue
        if min_dividend_yield is not None and s["dividend_yield"] < min_dividend_yield:
            continue
        if min_volume is not None and s["volume"] < min_volume:
            continue
        results.append(s)

    sort_keys: dict[str, Any] = {
        "market_cap": lambda x: -x["market_cap"],
        "pe_ratio": lambda x: x["pe_ratio"],
        "dividend_yield": lambda x: -x["dividend_yield"],
        "volume": lambda x: -x["volume"],
        "price": lambda x: -x["current_price"],
    }
    if sort_by in sort_keys:
        results.sort(key=sort_keys[sort_by])

    limit = max(1, min(limit, 50))
    return {
        "total": len(results),
        "stocks": results[:limit],
        "filters_applied": {
            "sector": sector,
            "min_market_cap": min_market_cap,
            "max_pe_ratio": max_pe_ratio,
            "min_dividend_yield": min_dividend_yield,
            "min_volume": min_volume,
        },
    }


# =========================================================================
# Cluster 2 — Portfolio Management
# =========================================================================
@mcp.tool()
def get_portfolio_positions(
    portfolio_id: Annotated[str, "Portfolio ID (e.g. 'PORT-0001')"],
) -> dict[str, Any]:
    """Get current holdings for a portfolio with unrealized P&L.

    Returns each position's ticker, shares, average cost, current value,
    and unrealized gain/loss. Use this to see what a client currently owns.
    For returns over time, use get_portfolio_performance. For a quick
    account-level summary, use get_account_summary.
    """
    portfolio = _portfolio_by_id(portfolio_id)
    if not portfolio:
        return {"error": f"Portfolio '{portfolio_id}' not found"}

    positions = []
    total_value = 0.0
    total_cost = 0.0
    for pos in portfolio["positions"]:
        stock = _stock_by_ticker(pos["ticker"])
        current_price = stock["current_price"] if stock else 0.0
        current_value = round(pos["shares"] * current_price, 2)
        cost_basis = round(pos["shares"] * pos["avg_cost"], 2)
        unrealized_pnl = round(current_value - cost_basis, 2)
        unrealized_pnl_pct = round(unrealized_pnl / cost_basis * 100, 2) if cost_basis else 0.0

        positions.append(
            {
                "ticker": pos["ticker"],
                "name": stock["name"] if stock else pos["ticker"],
                "shares": pos["shares"],
                "avg_cost": pos["avg_cost"],
                "current_price": current_price,
                "current_value": current_value,
                "cost_basis": cost_basis,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
            }
        )
        total_value += current_value
        total_cost += cost_basis

    return {
        "portfolio_id": portfolio_id,
        "client_name": portfolio["client_name"],
        "positions": positions,
        "total_market_value": round(total_value, 2),
        "total_cost_basis": round(total_cost, 2),
        "total_unrealized_pnl": round(total_value - total_cost, 2),
        "position_count": len(positions),
    }


@mcp.tool()
def get_portfolio_performance(
    portfolio_id: Annotated[str, "Portfolio ID (e.g. 'PORT-0001')"],
    period: Annotated[
        str,
        "Performance period: 1d, 1w, 1m, 3m, 6m, 1y, ytd",
    ] = "1m",
) -> dict[str, Any]:
    """Get time-weighted returns for a portfolio over a period.

    Returns percentage return, annualized return, and benchmark comparison.
    Use this to evaluate how a portfolio has performed. For current holdings
    and P&L, use get_portfolio_positions instead.
    """
    portfolio = _portfolio_by_id(portfolio_id)
    if not portfolio:
        return {"error": f"Portfolio '{portfolio_id}' not found"}

    valid_periods = {"1d", "1w", "1m", "3m", "6m", "1y", "ytd"}
    if period not in valid_periods:
        return {"error": f"Invalid period. Choose from: {sorted(valid_periods)}"}

    rng = _rng.Random(hash(portfolio_id) + hash(period))

    period_multipliers = {
        "1d": 1, "1w": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 252, "ytd": 150,
    }
    trading_days = period_multipliers[period]

    daily_return = rng.gauss(0.0004, 0.012)
    period_return = round((1 + daily_return) ** trading_days - 1, 4)
    annualized = round((1 + period_return) ** (252 / trading_days) - 1, 4)

    total_value = sum(
        pos["shares"] * (_stock_by_ticker(pos["ticker"]) or {}).get("current_price", 0)
        for pos in portfolio["positions"]
    ) + portfolio["cash_balance"]

    benchmark_return = round(rng.gauss(period_return, abs(period_return) * 0.3), 4)

    return {
        "portfolio_id": portfolio_id,
        "client_name": portfolio["client_name"],
        "period": period,
        "period_return_pct": round(period_return * 100, 2),
        "annualized_return_pct": round(annualized * 100, 2),
        "portfolio_value": round(total_value, 2),
        "benchmark_return_pct": round(benchmark_return * 100, 2),
        "alpha_pct": round((period_return - benchmark_return) * 100, 2),
        "benchmark": "S&P 500",
    }


@mcp.tool()
def get_account_summary(
    portfolio_id: Annotated[str, "Portfolio ID (e.g. 'PORT-0001')"],
) -> dict[str, Any]:
    """Get a high-level account overview for a portfolio.

    Returns total value, cash balance, margin, buying power, and account
    type. Use this for a quick snapshot. For detailed holdings, use
    get_portfolio_positions. For returns, use get_portfolio_performance.
    """
    portfolio = _portfolio_by_id(portfolio_id)
    if not portfolio:
        return {"error": f"Portfolio '{portfolio_id}' not found"}

    securities_value = sum(
        pos["shares"] * (_stock_by_ticker(pos["ticker"]) or {}).get("current_price", 0)
        for pos in portfolio["positions"]
    )
    total_value = securities_value + portfolio["cash_balance"]
    buying_power = portfolio["cash_balance"] + portfolio["margin_available"]

    return {
        "portfolio_id": portfolio_id,
        "client_name": portfolio["client_name"],
        "account_type": portfolio["account_type"],
        "risk_tolerance": portfolio["risk_tolerance"],
        "investment_objective": portfolio["investment_objective"],
        "total_value": round(total_value, 2),
        "securities_value": round(securities_value, 2),
        "cash_balance": portfolio["cash_balance"],
        "margin_available": portfolio["margin_available"],
        "buying_power": round(buying_power, 2),
        "position_count": len(portfolio["positions"]),
        "created_at": portfolio["created_at"],
    }


@mcp.tool()
def get_transaction_history(
    portfolio_id: Annotated[str, "Portfolio ID (e.g. 'PORT-0001')"],
    date_from: Annotated[
        Optional[str], "Start date filter (ISO format YYYY-MM-DD)"
    ] = None,
    date_to: Annotated[
        Optional[str], "End date filter (ISO format YYYY-MM-DD)"
    ] = None,
    action: Annotated[
        Optional[str], "Filter by action: buy, sell, dividend, fee"
    ] = None,
    ticker: Annotated[
        Optional[str], "Filter by ticker symbol"
    ] = None,
) -> dict[str, Any]:
    """Get past trades, dividends, and fees for a portfolio.

    Returns a filtered list of transactions. Use this for trade history,
    audit trails, or dividend tracking. For current positions and P&L,
    use get_portfolio_positions instead.
    """
    portfolio = _portfolio_by_id(portfolio_id)
    if not portfolio:
        return {"error": f"Portfolio '{portfolio_id}' not found"}

    results = []
    for tx in store.transactions:
        if tx["portfolio_id"] != portfolio_id:
            continue
        if date_from and tx["date"] < date_from:
            continue
        if date_to and tx["date"] > date_to:
            continue
        if action and tx["action"] != action:
            continue
        if ticker and tx["ticker"].upper() != ticker.upper():
            continue
        results.append(tx)

    results.sort(key=lambda x: x["date"], reverse=True)

    return {
        "portfolio_id": portfolio_id,
        "client_name": portfolio["client_name"],
        "total_transactions": len(results),
        "transactions": results,
        "filters_applied": {
            "date_from": date_from,
            "date_to": date_to,
            "action": action,
            "ticker": ticker,
        },
    }


# =========================================================================
# Cluster 3 — Risk & Analytics
# =========================================================================
@mcp.tool()
def calculate_portfolio_risk(
    portfolio_id: Annotated[str, "Portfolio ID (e.g. 'PORT-0001')"],
) -> dict[str, Any]:
    """Calculate risk metrics for a portfolio.

    Returns Value-at-Risk (95%), Sharpe ratio, portfolio beta, maximum
    drawdown, and annualized volatility. Use this for overall risk
    assessment. For sector-level breakdown, use get_sector_exposure.
    For scenario analysis, use run_stress_test.
    """
    portfolio = _portfolio_by_id(portfolio_id)
    if not portfolio:
        return {"error": f"Portfolio '{portfolio_id}' not found"}

    rng = _rng.Random(hash(portfolio_id) + 99)

    total_value = sum(
        pos["shares"] * (_stock_by_ticker(pos["ticker"]) or {}).get("current_price", 0)
        for pos in portfolio["positions"]
    ) + portfolio["cash_balance"]

    betas = []
    weights = []
    for pos in portfolio["positions"]:
        stock = _stock_by_ticker(pos["ticker"])
        if stock:
            val = pos["shares"] * stock["current_price"]
            weights.append(val / total_value if total_value else 0)
            betas.append(stock["beta"])

    portfolio_beta = round(
        sum(w * b for w, b in zip(weights, betas)), 2
    ) if betas else 1.0

    volatility = round(rng.uniform(0.10, 0.30), 4)
    sharpe = round(rng.uniform(0.3, 2.5), 2)
    var_95 = round(total_value * 1.65 * volatility / math.sqrt(252), 2)
    max_drawdown = round(rng.uniform(-0.35, -0.05), 4)

    return {
        "portfolio_id": portfolio_id,
        "client_name": portfolio["client_name"],
        "portfolio_value": round(total_value, 2),
        "var_95_daily": var_95,
        "var_95_pct": round(var_95 / total_value * 100, 2) if total_value else 0,
        "sharpe_ratio": sharpe,
        "portfolio_beta": portfolio_beta,
        "annualized_volatility_pct": round(volatility * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "risk_level": "high" if volatility > 0.25 else "moderate" if volatility > 0.15 else "low",
    }


@mcp.tool()
def get_sector_exposure(
    portfolio_id: Annotated[str, "Portfolio ID (e.g. 'PORT-0001')"],
) -> dict[str, Any]:
    """Get sector allocation breakdown for a portfolio.

    Returns each sector's weight, market value, and position count.
    Use this to check diversification. For overall risk metrics like
    VaR or Sharpe, use calculate_portfolio_risk instead.
    """
    portfolio = _portfolio_by_id(portfolio_id)
    if not portfolio:
        return {"error": f"Portfolio '{portfolio_id}' not found"}

    sector_data: dict[str, dict[str, float]] = defaultdict(
        lambda: {"value": 0.0, "positions": 0}
    )
    total_value = 0.0

    for pos in portfolio["positions"]:
        stock = _stock_by_ticker(pos["ticker"])
        if stock:
            val = pos["shares"] * stock["current_price"]
            sector_data[stock["sector"]]["value"] += val
            sector_data[stock["sector"]]["positions"] += 1
            total_value += val

    sectors = []
    for sector_name in sorted(sector_data.keys()):
        data = sector_data[sector_name]
        weight = round(data["value"] / total_value * 100, 2) if total_value else 0
        sectors.append(
            {
                "sector": sector_name,
                "market_value": round(data["value"], 2),
                "weight_pct": weight,
                "positions": int(data["positions"]),
            }
        )

    sectors.sort(key=lambda x: -x["weight_pct"])

    return {
        "portfolio_id": portfolio_id,
        "client_name": portfolio["client_name"],
        "total_securities_value": round(total_value, 2),
        "sector_count": len(sectors),
        "sectors": sectors,
        "largest_sector": sectors[0]["sector"] if sectors else None,
        "concentration_warning": any(s["weight_pct"] > 40 for s in sectors),
    }


@mcp.tool()
def run_stress_test(
    portfolio_id: Annotated[str, "Portfolio ID (e.g. 'PORT-0001')"],
    scenario: Annotated[
        str,
        "Scenario: market_crash_2008, rate_hike_300bps, tech_selloff_30pct, "
        "oil_shock, pandemic_lockdown",
    ],
) -> dict[str, Any]:
    """Run a hypothetical stress test on a portfolio.

    Simulates the impact of a named market scenario on portfolio value.
    Use this for 'what-if' analysis. For current risk metrics, use
    calculate_portfolio_risk. For sector diversification, use
    get_sector_exposure.
    """
    portfolio = _portfolio_by_id(portfolio_id)
    if not portfolio:
        return {"error": f"Portfolio '{portfolio_id}' not found"}

    valid_scenarios = {
        "market_crash_2008",
        "rate_hike_300bps",
        "tech_selloff_30pct",
        "oil_shock",
        "pandemic_lockdown",
    }
    if scenario not in valid_scenarios:
        return {"error": f"Invalid scenario. Choose from: {sorted(valid_scenarios)}"}

    scenario_impacts: dict[str, dict[str, float]] = {
        "market_crash_2008": {
            "Technology": -0.45, "Healthcare": -0.20, "Financial Services": -0.55,
            "Consumer Discretionary": -0.40, "Energy": -0.35, "Industrials": -0.38,
            "Real Estate": -0.42, "Utilities": -0.15,
        },
        "rate_hike_300bps": {
            "Technology": -0.20, "Healthcare": -0.05, "Financial Services": 0.10,
            "Consumer Discretionary": -0.15, "Energy": 0.05, "Industrials": -0.10,
            "Real Estate": -0.25, "Utilities": -0.18,
        },
        "tech_selloff_30pct": {
            "Technology": -0.30, "Healthcare": -0.05, "Financial Services": -0.08,
            "Consumer Discretionary": -0.15, "Energy": 0.02, "Industrials": -0.05,
            "Real Estate": -0.03, "Utilities": 0.02,
        },
        "oil_shock": {
            "Technology": -0.08, "Healthcare": -0.05, "Financial Services": -0.10,
            "Consumer Discretionary": -0.12, "Energy": 0.25, "Industrials": -0.15,
            "Real Estate": -0.05, "Utilities": -0.10,
        },
        "pandemic_lockdown": {
            "Technology": 0.15, "Healthcare": 0.10, "Financial Services": -0.20,
            "Consumer Discretionary": -0.30, "Energy": -0.40, "Industrials": -0.25,
            "Real Estate": -0.18, "Utilities": -0.08,
        },
    }

    impacts = scenario_impacts[scenario]
    position_impacts = []
    total_current = 0.0
    total_stressed = 0.0

    for pos in portfolio["positions"]:
        stock = _stock_by_ticker(pos["ticker"])
        if not stock:
            continue
        current_val = pos["shares"] * stock["current_price"]
        impact_pct = impacts.get(stock["sector"], -0.10)
        stressed_val = round(current_val * (1 + impact_pct), 2)
        position_impacts.append(
            {
                "ticker": pos["ticker"],
                "sector": stock["sector"],
                "current_value": round(current_val, 2),
                "stressed_value": stressed_val,
                "impact_pct": round(impact_pct * 100, 2),
                "impact_amount": round(stressed_val - current_val, 2),
            }
        )
        total_current += current_val
        total_stressed += stressed_val

    portfolio_impact_pct = round(
        (total_stressed - total_current) / total_current * 100, 2
    ) if total_current else 0

    return {
        "portfolio_id": portfolio_id,
        "client_name": portfolio["client_name"],
        "scenario": scenario,
        "current_portfolio_value": round(total_current, 2),
        "stressed_portfolio_value": round(total_stressed, 2),
        "portfolio_impact_pct": portfolio_impact_pct,
        "portfolio_impact_amount": round(total_stressed - total_current, 2),
        "position_impacts": position_impacts,
        "cash_unaffected": portfolio["cash_balance"],
    }


@mcp.tool()
def analyze_stock(
    ticker: Annotated[str, "Stock ticker symbol (e.g. 'AAPL')"],
) -> dict[str, Any]:
    """Run fundamental and technical analysis on a single stock.

    Returns valuation metrics, support/resistance levels, analyst-style
    summary, and sector comparison. Use this for in-depth stock research.
    For just the current price, use get_stock_quote. For screening many
    stocks, use screen_stocks.
    """
    stock = _stock_by_ticker(ticker)
    if not stock:
        return {"error": f"Ticker '{ticker}' not found"}

    rng = _rng.Random(hash(ticker) + 777)

    peers = [s for s in store.stocks if s["sector"] == stock["sector"] and s["ticker"] != stock["ticker"]]
    sector_avg_pe = round(
        statistics.mean([s["pe_ratio"] for s in peers]), 2
    ) if peers else stock["pe_ratio"]
    sector_avg_yield = round(
        statistics.mean([s["dividend_yield"] for s in peers]), 2
    ) if peers else stock["dividend_yield"]

    support_1 = round(stock["current_price"] * rng.uniform(0.92, 0.97), 2)
    support_2 = round(stock["current_price"] * rng.uniform(0.85, 0.92), 2)
    resistance_1 = round(stock["current_price"] * rng.uniform(1.03, 1.08), 2)
    resistance_2 = round(stock["current_price"] * rng.uniform(1.08, 1.15), 2)

    rsi = round(rng.uniform(25, 75), 1)
    ma_50 = round(stock["current_price"] * rng.uniform(0.95, 1.05), 2)
    ma_200 = round(stock["current_price"] * rng.uniform(0.88, 1.02), 2)

    if stock["pe_ratio"] < sector_avg_pe * 0.8:
        valuation = "undervalued"
    elif stock["pe_ratio"] > sector_avg_pe * 1.2:
        valuation = "overvalued"
    else:
        valuation = "fairly_valued"

    if rsi < 30:
        signal = "oversold"
    elif rsi > 70:
        signal = "overbought"
    elif stock["current_price"] > ma_50 > ma_200:
        signal = "bullish"
    elif stock["current_price"] < ma_50 < ma_200:
        signal = "bearish"
    else:
        signal = "neutral"

    return {
        "ticker": stock["ticker"],
        "name": stock["name"],
        "sector": stock["sector"],
        "current_price": stock["current_price"],
        "fundamentals": {
            "pe_ratio": stock["pe_ratio"],
            "eps": stock["eps"],
            "dividend_yield": stock["dividend_yield"],
            "beta": stock["beta"],
            "market_cap": stock["market_cap"],
            "market_cap_formatted": _format_market_cap(stock["market_cap"]),
        },
        "valuation": {
            "assessment": valuation,
            "pe_vs_sector": round(stock["pe_ratio"] - sector_avg_pe, 2),
            "sector_avg_pe": sector_avg_pe,
            "sector_avg_dividend_yield": sector_avg_yield,
        },
        "technicals": {
            "rsi_14": rsi,
            "ma_50": ma_50,
            "ma_200": ma_200,
            "signal": signal,
            "support_levels": [support_1, support_2],
            "resistance_levels": [resistance_1, resistance_2],
        },
        "year_range": {
            "high": stock["year_high"],
            "low": stock["year_low"],
            "pct_from_high": round(
                (stock["current_price"] - stock["year_high"]) / stock["year_high"] * 100, 2
            ),
        },
    }


# =========================================================================
# Cluster 4 — Trading & Compliance
# =========================================================================
@mcp.tool()
def submit_trade_order(
    portfolio_id: Annotated[str, "Portfolio ID (e.g. 'PORT-0001')"],
    ticker: Annotated[str, "Stock ticker symbol to trade"],
    action: Annotated[str, "Trade action: buy or sell"],
    shares: Annotated[int, "Number of shares to trade"],
    order_type: Annotated[str, "Order type: market, limit, stop"] = "market",
    limit_price: Annotated[
        Optional[float], "Limit price (required for limit/stop orders)"
    ] = None,
) -> dict[str, Any]:
    """Place a trade order with pre-trade compliance validation.

    Executes a buy or sell order after running compliance checks. Returns
    order confirmation or rejection details. Always runs check_compliance
    internally before execution. Use check_compliance separately for
    dry-run validation without placing an order.
    """
    portfolio = _portfolio_by_id(portfolio_id)
    if not portfolio:
        return {"error": f"Portfolio '{portfolio_id}' not found"}

    stock = _stock_by_ticker(ticker)
    if not stock:
        return {"error": f"Ticker '{ticker}' not found"}

    if action not in ("buy", "sell"):
        return {"error": "Action must be 'buy' or 'sell'"}

    if order_type not in ("market", "limit", "stop"):
        return {"error": "Order type must be 'market', 'limit', or 'stop'"}

    if order_type in ("limit", "stop") and limit_price is None:
        return {"error": f"limit_price is required for {order_type} orders"}

    compliance = check_compliance(portfolio_id, ticker, action, shares)
    if not compliance.get("compliant", False):
        return {
            "status": "rejected",
            "reason": "compliance_failure",
            "compliance_details": compliance,
        }

    exec_price = limit_price if limit_price else stock["current_price"]
    total = round(shares * exec_price, 2)

    if action == "buy" and total > portfolio["cash_balance"] + portfolio["margin_available"]:
        return {
            "status": "rejected",
            "reason": "insufficient_funds",
            "required": total,
            "available": round(
                portfolio["cash_balance"] + portfolio["margin_available"], 2
            ),
        }

    if action == "sell":
        held = 0
        for pos in portfolio["positions"]:
            if pos["ticker"].upper() == ticker.upper():
                held = pos["shares"]
                break
        if shares > held:
            return {
                "status": "rejected",
                "reason": "insufficient_shares",
                "requested": shares,
                "held": held,
            }

    import uuid

    order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"

    return {
        "status": "submitted",
        "order_id": order_id,
        "portfolio_id": portfolio_id,
        "ticker": ticker.upper(),
        "action": action,
        "shares": shares,
        "order_type": order_type,
        "execution_price": exec_price,
        "total_amount": total,
        "compliance_passed": True,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@mcp.tool()
def check_compliance(
    portfolio_id: Annotated[str, "Portfolio ID (e.g. 'PORT-0001')"],
    ticker: Annotated[str, "Stock ticker to check"],
    action: Annotated[str, "Proposed action: buy or sell"],
    shares: Annotated[int, "Number of shares in proposed trade"],
) -> dict[str, Any]:
    """Run pre-trade compliance validation without placing an order.

    Checks restricted list, position concentration, sector concentration,
    cash reserve requirements, and order type suitability. Use this for
    dry-run validation before submitting a trade. submit_trade_order
    calls this internally.
    """
    portfolio = _portfolio_by_id(portfolio_id)
    if not portfolio:
        return {"error": f"Portfolio '{portfolio_id}' not found"}

    stock = _stock_by_ticker(ticker)
    if not stock:
        return {"error": f"Ticker '{ticker}' not found"}

    rules = store.compliance_rules
    violations: list[str] = []
    warnings: list[str] = []

    if ticker.upper() in rules["restricted_tickers"]:
        violations.append(f"{ticker} is on the restricted securities list")

    total_value = sum(
        pos["shares"] * (_stock_by_ticker(pos["ticker"]) or {}).get("current_price", 0)
        for pos in portfolio["positions"]
    ) + portfolio["cash_balance"]

    if action == "buy" and total_value > 0:
        trade_value = shares * stock["current_price"]
        existing_value = 0.0
        for pos in portfolio["positions"]:
            if pos["ticker"].upper() == ticker.upper():
                existing_value = pos["shares"] * stock["current_price"]
                break
        new_position_value = existing_value + trade_value
        position_pct = new_position_value / total_value * 100
        if position_pct > rules["max_single_position_pct"]:
            violations.append(
                f"Position would be {position_pct:.1f}% of portfolio "
                f"(max {rules['max_single_position_pct']}%)"
            )

        sector_value = 0.0
        for pos in portfolio["positions"]:
            s = _stock_by_ticker(pos["ticker"])
            if s and s["sector"] == stock["sector"]:
                sector_value += pos["shares"] * s["current_price"]
        new_sector_value = sector_value + trade_value
        sector_pct = new_sector_value / total_value * 100
        if sector_pct > rules["max_sector_concentration_pct"]:
            violations.append(
                f"{stock['sector']} sector would be {sector_pct:.1f}% of portfolio "
                f"(max {rules['max_sector_concentration_pct']}%)"
            )

        remaining_cash = portfolio["cash_balance"] - trade_value
        cash_pct = remaining_cash / total_value * 100 if total_value else 0
        if cash_pct < rules["min_cash_reserve_pct"]:
            warnings.append(
                f"Cash reserve would drop to {cash_pct:.1f}% "
                f"(minimum {rules['min_cash_reserve_pct']}%)"
            )

    return {
        "portfolio_id": portfolio_id,
        "ticker": ticker.upper(),
        "action": action,
        "shares": shares,
        "compliant": len(violations) == 0,
        "violations": violations,
        "warnings": warnings,
        "risk_tolerance": portfolio["risk_tolerance"],
    }


@mcp.tool()
def get_regulatory_status(
    portfolio_id: Annotated[str, "Portfolio ID (e.g. 'PORT-0001')"],
) -> dict[str, Any]:
    """Get KYC/AML and regulatory status for a portfolio's client.

    Returns accreditation level, KYC status, AML status, risk tolerance,
    and any regulatory flags. Use this to verify a client's standing
    before high-value or complex trades. For trade-level compliance
    checks, use check_compliance.
    """
    portfolio = _portfolio_by_id(portfolio_id)
    if not portfolio:
        return {"error": f"Portfolio '{portfolio_id}' not found"}

    rng = _rng.Random(hash(portfolio_id) + 42)

    accreditation_levels = ["retail", "accredited", "qualified_purchaser"]
    weights = [0.5, 0.35, 0.15]
    accreditation = rng.choices(accreditation_levels, weights=weights, k=1)[0]

    return {
        "portfolio_id": portfolio_id,
        "client_id": portfolio["client_id"],
        "client_name": portfolio["client_name"],
        "kyc_status": "verified",
        "aml_status": "clear",
        "accreditation_level": accreditation,
        "risk_tolerance": portfolio["risk_tolerance"],
        "account_type": portfolio["account_type"],
        "investment_objective": portfolio["investment_objective"],
        "regulatory_flags": [],
        "last_review_date": _rng.Random(hash(portfolio_id)).choice(
            ["2025-01-15", "2025-03-22", "2025-06-10", "2025-09-05"]
        ),
        "next_review_due": "2026-01-15",
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8009"))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
