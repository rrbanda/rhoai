# Demo MCP Server: ShopInsights Analytics Platform

A FastMCP-based e-commerce analytics server with **15 tools** organized into ambiguity clusters. Designed specifically for testing MCP distillation — the tools are deliberately similar within each cluster to challenge small models on tool selection.

## Tool Clusters

| Cluster | Tools | Challenge |
|---------|-------|-----------|
| Product Discovery | `search_products`, `browse_catalog`, `get_trending_products`, `get_product_details` | Which search/browse tool fits a given query? |
| Sales & Revenue | `get_sales_data`, `get_revenue_report`, `get_store_overview` | Per-product units vs. aggregate revenue vs. quick snapshot |
| Customer Analytics | `get_customer_segments`, `get_customer_profile`, `get_abandoned_carts` | Individual lookup vs. aggregate analysis vs. behavioral data |
| Multi-step | `analyze_product_performance`, `compare_products`, `forecast_demand`, `get_inventory_status`, `create_promotion` | Require chaining 2-4 tools in the correct order |

## Quick Start

### Install dependencies

```bash
pip install fastmcp
```

### Run the server

```bash
python server.py
```

The server starts on `http://0.0.0.0:8008` using Streamable-HTTP transport. Configure the port via the `PORT` environment variable:

```bash
PORT=9000 python server.py
```

### Connect from Langflow

1. In Langflow, add an MCP tool component
2. Set the server URL to `http://localhost:8008/mcp`
3. The 15 tools will auto-discover

## Data

The server uses deterministic seed data (`data.py`) with:
- ~50 products across 9 subcategories
- 30 customers with varied segments
- 200 orders spanning 2025
- Inventory across 3 warehouses
- 15 abandoned carts
- 5 promotional campaigns

All data is generated with a fixed random seed (42) for reproducibility.

## Files

| File | Description |
|------|-------------|
| `server.py` | FastMCP server with 15 tool definitions |
| `data.py` | Deterministic data generation (products, orders, customers, etc.) |
