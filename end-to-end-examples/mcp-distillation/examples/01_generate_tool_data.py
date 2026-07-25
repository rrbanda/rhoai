"""Generate tool-use training data via SDG Hub's MCP distillation flow.

This script runs the MCP Server Distillation pipeline end-to-end:
  1. Loads the built-in distillation flow from SDG Hub
  2. Configures teacher LLM and Langflow agent connections
  3. Creates an input dataset from MCP server tool schemas
  4. Runs the full pipeline (exploration -> question synthesis -> expert trajectories)
  5. Saves the output as a Parquet file for downstream formatting

The pipeline uses a frontier model (via Langflow + MCP server) to actively explore
your tools and generate expert-quality demonstrations, then a teacher LLM to
synthesize and quality-filter the training questions.
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
    """Build the input DataFrame with MCP server tool schemas.

    Replace this function with your own MCP server's tool definitions.
    Each row represents one MCP server to generate training data for.
    """
    tool_list = [
        {
            "name": "search_products",
            "description": "Search products by keyword, category, or price range.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "category": {"type": "string", "description": "Filter by category"},
                    "min_price": {"type": "number", "description": "Minimum price"},
                    "max_price": {"type": "number", "description": "Maximum price"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10},
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_product_details",
            "description": "Get detailed information about a specific product by ID.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "Product ID (e.g. PROD-0001)"},
                },
                "required": ["product_id"],
            },
        },
        {
            "name": "get_trending_products",
            "description": "Get currently trending products ranked by sales velocity or revenue.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["revenue", "units_sold", "views"],
                        "description": "Ranking metric",
                    },
                    "category": {"type": "string", "description": "Filter by category"},
                    "limit": {"type": "integer", "description": "Max results", "default": 5},
                },
            },
        },
        {
            "name": "browse_catalog",
            "description": "Browse the product catalog with filtering and sorting options.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Category filter"},
                    "sort_by": {
                        "type": "string",
                        "enum": ["price_asc", "price_desc", "rating", "newest"],
                    },
                    "page": {"type": "integer", "default": 1},
                    "page_size": {"type": "integer", "default": 20},
                },
            },
        },
        {
            "name": "get_sales_data",
            "description": "Get per-product sales data including units sold and revenue.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "Filter by product"},
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                },
            },
        },
        {
            "name": "get_revenue_report",
            "description": "Get aggregate revenue report with breakdowns by category and region.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["daily", "weekly", "monthly", "quarterly"],
                    },
                    "category": {"type": "string"},
                    "region": {"type": "string"},
                },
            },
        },
        {
            "name": "get_store_overview",
            "description": "Quick snapshot of store performance: total revenue, orders, AOV.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "get_customer_segments",
            "description": "Get customer segment breakdown with counts and average lifetime value.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "segment": {
                        "type": "string",
                        "enum": ["vip", "returning", "new", "at_risk"],
                    },
                },
            },
        },
        {
            "name": "get_customer_profile",
            "description": "Look up a specific customer by ID or email.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "email": {"type": "string"},
                },
            },
        },
        {
            "name": "get_abandoned_carts",
            "description": "List abandoned shopping carts with items and potential revenue.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "min_value": {"type": "number", "description": "Minimum cart value"},
                    "limit": {"type": "integer", "default": 10},
                },
            },
        },
        {
            "name": "get_inventory_status",
            "description": "Check inventory levels across warehouses for one or more products.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Product IDs to check",
                    },
                },
                "required": ["product_ids"],
            },
        },
        {
            "name": "forecast_demand",
            "description": "Forecast demand for a product over the next N days.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "days_ahead": {"type": "integer", "default": 30},
                },
                "required": ["product_id"],
            },
        },
        {
            "name": "compare_products",
            "description": "Compare two or more products on price, rating, and sales.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Product IDs to compare",
                    },
                },
                "required": ["product_ids"],
            },
        },
        {
            "name": "analyze_product_performance",
            "description": "Deep analysis of a product's performance across all metrics.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "period": {"type": "string", "enum": ["7d", "30d", "90d", "all"]},
                },
                "required": ["product_id"],
            },
        },
        {
            "name": "create_promotion",
            "description": "Create a promotional discount for specified products.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "product_ids": {"type": "array", "items": {"type": "string"}},
                    "discount_type": {
                        "type": "string",
                        "enum": ["percentage", "fixed_amount", "buy_one_get_one"],
                    },
                    "discount_value": {"type": "number"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["name", "product_ids", "discount_type", "discount_value"],
            },
        },
    ]

    return pd.DataFrame(
        {
            "tool_list": [tool_list],
            "mcp_server_name": ["ShopInsights Analytics Platform"],
            "mcp_server_description": [
                "E-commerce analytics platform for an online retailer. "
                "Provides product search, sales analytics, customer insights, "
                "demand forecasting, and promotional management. "
                "Features 15 tools organized across product discovery, sales & revenue, "
                "customer analytics, and multi-step analytical workflows."
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
        print("Ensure sdg_hub is installed: pip install sdg_hub[examples]")
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
