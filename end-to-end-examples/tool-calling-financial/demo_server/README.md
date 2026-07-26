# Demo MCP Server: FinanceInsights Advisory Platform

A FastMCP-based financial advisory server with **15 tools** organized into ambiguity clusters. Designed specifically for testing MCP distillation — the tools are deliberately similar within each cluster to challenge small models on tool selection.

## Tool Clusters

| Cluster | Tools | Challenge |
|---------|-------|-----------|
| Market Data | `get_stock_quote`, `get_market_summary`, `get_historical_prices`, `screen_stocks` | Single-ticker quote vs. broad indices vs. time series vs. filtered screening |
| Portfolio Management | `get_portfolio_positions`, `get_portfolio_performance`, `get_account_summary`, `get_transaction_history` | Holdings vs. returns vs. account overview vs. trade history |
| Risk & Analytics | `calculate_portfolio_risk`, `get_sector_exposure`, `run_stress_test`, `analyze_stock` | VaR/Sharpe vs. allocation vs. scenario analysis vs. individual stock research |
| Trading & Compliance | `submit_trade_order`, `check_compliance`, `get_regulatory_status` | Execute trade vs. dry-run validation vs. client-level regulatory status |

## Quick Start

### Install dependencies

```bash
pip install fastmcp
```

### Run the server

```bash
python server.py
```

The server starts on `http://0.0.0.0:8009` using Streamable-HTTP transport. Configure the port via the `PORT` environment variable:

```bash
PORT=9000 python server.py
```

### Connect from Langflow

1. In Langflow, add an MCP tool component
2. Set the server URL to `http://localhost:8009/mcp`
3. The 15 tools will auto-discover

## Data

The server uses deterministic seed data (`data.py`) with:
- 50 stocks across 8 sectors (Technology, Healthcare, Financial Services, etc.)
- 5 client portfolios with varying risk tolerances and account types
- 200 transactions spanning 2025
- 4 major market indices
- 3 themed watchlists
- Compliance rules (restricted list, concentration limits, cash reserves)

All data is generated with a fixed random seed (42) for reproducibility.

## Files

| File | Description |
|------|-------------|
| `server.py` | FastMCP server with 15 tool definitions |
| `data.py` | Deterministic data generation (stocks, portfolios, transactions, etc.) |
