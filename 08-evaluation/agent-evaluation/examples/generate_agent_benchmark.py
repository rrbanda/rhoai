"""Generate synthetic MCP tool-use benchmark tasks using SDG Hub.

Given MCP servers and LangGraph agents connected to them, this script uses a
frontier model to explore tools, generate grounded questions, and produce
expert-quality gold-standard trajectories for evaluation.

Pipeline stages:
  1. Discover tools from each MCP server
  2. For each server, run the MCP Server Distillation flow at varying
     complexity levels (num_samples=2, 4, 8)
  3. Transform raw output into clean evaluation format with expert traces
  4. Save benchmark as JSONL

Usage:
    export OPENAI_API_KEY="your-api-key"
    export TEACHER_MODEL="openai/gpt-5.2"

    python generate_agent_benchmark.py \\
        --mcp-servers '{"Weather Data": "http://localhost:8001/mcp"}' \\
        --agent-urls '{"Weather Data": "http://localhost:2024"}' \\
        --output benchmark_tasks.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from sdg_hub import Flow, FlowRegistry


def normalize_tool_trace(raw_trace: list[dict] | str) -> list[dict]:
    """Normalize a tool trace to canonical format: [{name, input, output}, ...]."""
    if isinstance(raw_trace, str):
        raw_trace = json.loads(raw_trace)
    if not isinstance(raw_trace, list):
        return []

    cleaned: list[dict] = []
    pending: dict[str, dict] = {}

    for entry in raw_trace:
        if not isinstance(entry, dict):
            continue

        if entry.get("role") == "assistant" and entry.get("tool_calls"):
            for tc in entry["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                call: dict = {"name": func.get("name", ""), "input": args}
                tc_id = tc.get("id")
                if tc_id:
                    pending[tc_id] = call
                cleaned.append(call)

        elif entry.get("role") == "tool":
            tc_id = entry.get("tool_call_id")
            content = entry.get("content", "")
            if tc_id and tc_id in pending:
                pending[tc_id]["output"] = content
            elif cleaned and "output" not in cleaned[-1]:
                cleaned[-1]["output"] = content

        elif entry.get("type") == "tool_use":
            if "tool_calls" in entry:
                for tc in entry["tool_calls"]:
                    call = {"name": tc.get("name", ""), "input": tc.get("args", {})}
                    tc_id = tc.get("id")
                    if tc_id:
                        pending[tc_id] = call
                    cleaned.append(call)
            elif "name" in entry:
                step: dict = {
                    "name": entry["name"],
                    "input": entry.get("tool_input", {}),
                }
                if entry.get("output"):
                    step["output"] = entry["output"]
                cleaned.append(step)

        elif entry.get("type") == "tool_result":
            tc_id = entry.get("tool_call_id") or entry.get("id")
            if tc_id and tc_id in pending:
                pending[tc_id]["output"] = entry.get("content", "")
            elif cleaned and "output" not in cleaned[-1]:
                cleaned[-1]["output"] = entry.get("content", "")

    return cleaned


def extract_tool_names(trace: list[dict]) -> list[str]:
    """Extract ordered list of tool names from a canonical trace."""
    return [step["name"] for step in trace if "name" in step]


async def discover_tools(url: str) -> list[dict]:
    """Connect to an MCP server and list its tools."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            resp = await session.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema,
                }
                for t in resp.tools
            ]


def generate_tasks_for_server(
    server_name: str,
    tools: list[dict],
    agent_url: str,
    teacher_model: str,
    api_key: str,
    server_description: str = "",
    num_samples: int = 2,
    langgraph_api_key: str | None = None,
) -> pd.DataFrame:
    """Run the MCP Server Distillation flow for one server."""
    flow_path = FlowRegistry.get_flow_path("MCP Server Distillation")
    if flow_path is None:
        print("ERROR: Flow 'MCP Server Distillation' not found in registry.", file=sys.stderr)
        print("Ensure sdg_hub is installed: pip install sdg-hub[examples]", file=sys.stderr)
        sys.exit(1)
    flow_instance = Flow.from_yaml(flow_path)
    flow_instance.set_model_config(model=teacher_model, api_key=api_key)

    agent_kwargs: dict = {"agent_framework": "langgraph", "agent_url": agent_url}
    if langgraph_api_key:
        agent_kwargs["agent_api_key"] = langgraph_api_key
    flow_instance.set_agent_config(**agent_kwargs)
    flow_instance.set_agent_config(timeout=300, blocks=["explore_server"])
    flow_instance.set_agent_config(
        agent_framework="langgraph",
        blocks=["extract_exploration", "extract_agent_text"],
    )

    df = pd.DataFrame(
        {
            "tool_list": [tools],
            "mcp_server_name": [server_name],
            "mcp_server_description": [
                server_description or f"{server_name} MCP server"
            ],
        }
    )

    runtime_params = {}
    if num_samples != 2:
        runtime_params["sample_tools"] = {"num_samples": num_samples}

    result = flow_instance.generate(df, runtime_params=runtime_params)
    result_df = (
        result.to_pandas() if hasattr(result, "to_pandas") else pd.DataFrame(result)
    )

    export_cols = [
        c
        for c in [
            "question",
            "extract_agent_text_text",
            "extract_agent_text_tool_trace",
            "question_quality_rating",
            "completeness_rating",
        ]
        if c in result_df.columns
    ]
    return result_df[export_cols]


def transform_tasks(raw_df: pd.DataFrame, server_name: str) -> pd.DataFrame:
    """Transform raw distillation output into clean evaluation format."""
    df = raw_df.copy()
    df["expert_tool_trace"] = df["extract_agent_text_tool_trace"].apply(
        normalize_tool_trace
    )
    df["expert_tools"] = df["expert_tool_trace"].apply(extract_tool_names)
    df = df.rename(columns={"extract_agent_text_text": "expert_answer"})
    df["server"] = server_name
    return df[
        [
            "server",
            "question",
            "expert_answer",
            "expert_tools",
            "expert_tool_trace",
            "question_quality_rating",
            "completeness_rating",
        ]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic MCP benchmark tasks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mcp-servers",
        required=True,
        help='JSON dict of server_name → MCP URL, e.g. \'{"Weather": "http://localhost:8001/mcp"}\'.',
    )
    parser.add_argument(
        "--agent-urls",
        required=True,
        help='JSON dict of server_name → LangGraph agent URL, e.g. \'{"Weather": "http://localhost:2024"}\'.',
    )
    parser.add_argument(
        "--server-descriptions",
        default="{}",
        help="JSON dict of server_name → description (optional).",
    )
    parser.add_argument(
        "--complexity-levels",
        nargs="+",
        type=int,
        default=[2, 4, 8],
        help="Number of tools per question at each complexity level (default: 2 4 8).",
    )
    parser.add_argument(
        "--output",
        default="benchmark_tasks.jsonl",
        help="Output JSONL path (default: benchmark_tasks.jsonl).",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file for configuration (default: .env).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)

    mcp_servers: dict[str, str] = json.loads(args.mcp_servers)
    agent_urls: dict[str, str] = json.loads(args.agent_urls)
    server_descriptions: dict[str, str] = json.loads(args.server_descriptions)

    api_key = os.getenv("OPENAI_API_KEY", "")
    teacher_model = os.getenv("TEACHER_MODEL", "openai/gpt-5.2")
    langgraph_api_key = os.getenv("LANGGRAPH_API_KEY")

    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable must be set.", file=sys.stderr)
        sys.exit(1)

    FlowRegistry.discover_flows()
    output_path = Path(args.output)

    # Load existing benchmark for per-server caching
    if output_path.exists():
        benchmark_df = pd.read_json(output_path, orient="records", lines=True)
        cached_servers = set(benchmark_df["server"].unique())
        print(f"Loaded {len(benchmark_df)} existing tasks ({len(cached_servers)} servers cached)")
    else:
        benchmark_df = pd.DataFrame()
        cached_servers = set()

    # Step 1: Discover tools from each MCP server
    print(f"\n--- Step 1: Discovering tools from {len(mcp_servers)} servers ---")
    all_tools: dict[str, list[dict]] = {}
    for name, url in mcp_servers.items():
        try:
            tools = asyncio.run(discover_tools(url))
            all_tools[name] = tools
            print(f"  {name}: {len(tools)} tools")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    if not all_tools:
        print("ERROR: No MCP servers reachable.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Generate tasks for each server
    print(f"\n--- Step 2: Generating benchmark tasks ---")
    print(f"Teacher model: {teacher_model}")
    print(f"Complexity levels: {args.complexity_levels}")

    new_tasks = []
    for server_name, tools in all_tools.items():
        if server_name in cached_servers:
            n = len(benchmark_df[benchmark_df["server"] == server_name])
            print(f"  {server_name}: {n} tasks (cached)")
            continue

        agent_url = agent_urls.get(server_name, "")
        if not agent_url:
            print(f"  {server_name}: SKIP - no agent URL configured")
            continue

        server_tasks = []
        n_tools = len(tools)

        for ns in args.complexity_levels:
            if n_tools < ns:
                print(f"  {server_name} ns={ns}: skipped ({n_tools} tools < {ns})")
                continue

            print(f"  {server_name} - num_samples={ns} ({n_tools} tools)...")
            try:
                result_df = generate_tasks_for_server(
                    server_name,
                    tools,
                    agent_url,
                    teacher_model=teacher_model,
                    api_key=api_key,
                    server_description=server_descriptions.get(server_name, ""),
                    num_samples=ns,
                    langgraph_api_key=langgraph_api_key,
                )
                server_tasks.append(result_df)
                print(f"    Generated {len(result_df)} tasks")
            except Exception as e:
                print(f"    FAILED: {e}")

        if server_tasks:
            raw_combined = pd.concat(server_tasks, ignore_index=True)
            transformed = transform_tasks(raw_combined, server_name)
            new_tasks.append(transformed)
            print(f"  {server_name}: {len(transformed)} total tasks")

    if new_tasks:
        new_df = pd.concat(new_tasks, ignore_index=True)
        benchmark_df = pd.concat([benchmark_df, new_df], ignore_index=True)

    # Step 3: Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_json(output_path, orient="records", lines=True)

    print(f"\n--- Results ---")
    print(f"Benchmark: {len(benchmark_df)} tasks across {benchmark_df['server'].nunique()} servers")
    for server, group in benchmark_df.groupby("server"):
        print(f"  {server:<25s} {len(group)} tasks")
    print(f"\nSaved to {output_path}")

    # Print a sample task
    if len(benchmark_df) > 0:
        sample = benchmark_df.iloc[0]
        print(f"\n--- Sample Task ---")
        print(f"Server: {sample['server']}")
        question = str(sample["question"])
        print(f"Question: {question[:300]}{'...' if len(question) > 300 else ''}")
        print(f"Expert Tools: {sample['expert_tools']}")


if __name__ == "__main__":
    main()
