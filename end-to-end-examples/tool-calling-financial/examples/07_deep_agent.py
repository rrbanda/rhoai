"""Financial Insights Deep Agent — powered by a fine-tuned model on RHOAI.

Creates a LangChain Deep Agent that wraps a fine-tuned Qwen3-4B served via
vLLM on RHOAI with 15 financial tools from the FinanceInsights MCP server.

The agent uses the ``deepagents`` library (built on LangGraph) which provides:
  - Task planning and decomposition
  - Tool calling with automatic schema inference
  - Long-term memory via AGENTS.md
  - Subagent spawning for parallel tasks
  - Context offloading for long conversations

Architecture:
  ┌──────────────────────────────────────────────────────┐
  │  LangGraph Runtime (http://127.0.0.1:2024)           │
  │                                                      │
  │  ┌──────────────────────────────────────────────────┐ │
  │  │  Deep Agent (Financial Insights)                 │ │
  │  │                                                  │ │
  │  │  LLM: ChatOpenAI -> vLLM on RHOAI               │ │
  │  │       (fine-tuned Qwen3-4B with LoRA adapter)    │ │
  │  │                                                  │ │
  │  │  Tools: 15 financial @tool functions             │ │
  │  │         -> HTTP calls to MCP server              │ │
  │  │                                                  │ │
  │  │  Memory: /memory/AGENTS.md (persistent)          │ │
  │  └──────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────┘

Quick start:
  1. Start the MCP server: cd demo_server && python server.py
  2. Set MODEL_ENDPOINT to your vLLM route URL
  3. Interactive: langgraph dev
  4. Headless:   python 07_deep_agent.py

Environment variables:
  MODEL_ENDPOINT    vLLM endpoint URL (e.g. https://tool-calling-financial.apps.cluster.com/v1)
  MODEL_NAME        Model name at the endpoint (default: default)
  MCP_SERVER_URL    MCP server URL (default: http://localhost:8009/mcp)
  OPENAI_API_KEY    API key for the vLLM endpoint (default: not-needed)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


def _build_agent():
    """Construct the Deep Agent graph.

    Separated into a function so langgraph.json can import ``agent`` at
    module level without triggering side effects during import.
    """
    from deepagents import create_deep_agent
    from langchain_openai import ChatOpenAI

    from financial_prompts import FINANCIAL_AGENT_INSTRUCTIONS
    from financial_tools import ALL_TOOLS

    model_endpoint = os.environ.get(
        "MODEL_ENDPOINT", "http://localhost:8000/v1"
    )
    model_name = os.environ.get("MODEL_NAME", "default")
    api_key = os.environ.get("OPENAI_API_KEY", "not-needed")

    model = ChatOpenAI(
        base_url=model_endpoint,
        model=model_name,
        api_key=api_key,
        temperature=0.1,
        max_tokens=4096,
    )

    current_date = datetime.now().strftime("%Y-%m-%d")

    return create_deep_agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=FINANCIAL_AGENT_INSTRUCTIONS.format(date=current_date),
        memory=["/memory/AGENTS.md"],
    )


def _get_agent():
    """Lazy-build the agent so imports don't fail at parse time."""
    global _agent_instance
    try:
        return _agent_instance
    except NameError:
        _agent_instance = _build_agent()
        return _agent_instance


def __getattr__(name: str):
    """Module-level __getattr__ so ``langgraph dev`` can find ``agent``."""
    if name == "agent":
        return _get_agent()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    """Run the agent with a sample prompt for smoke-testing."""
    test_prompts = [
        "What is the current price of AAPL and how does it compare to the broader market?",
        "What is the risk-adjusted performance of portfolio PORT-0001?",
        "Check if buying 50 shares of JPM in PORT-0002 would be compliant, and if so, what would the sector exposure look like after the trade?",
    ]

    prompt = test_prompts[0]
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])

    print("=" * 60)
    print("Financial Insights Deep Agent")
    print("=" * 60)
    print(f"  Model endpoint: {os.environ.get('MODEL_ENDPOINT', 'http://localhost:8000/v1')}")
    print(f"  MCP server:     {os.environ.get('MCP_SERVER_URL', 'http://localhost:8009/mcp')}")
    print(f"  Prompt:         {prompt}")
    print("=" * 60)

    result = _get_agent().invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )

    print("\n--- Agent Response ---")
    for msg in result.get("messages", []):
        role = msg.get("type", msg.get("role", "unknown"))
        content = msg.get("content", "")
        if content:
            print(f"\n[{role}] {content[:500]}")
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                print(f"\n[tool_call] {tc['name']}({tc.get('args', {})})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
