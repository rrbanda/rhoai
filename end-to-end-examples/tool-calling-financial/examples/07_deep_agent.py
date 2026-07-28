#!/usr/bin/env python3
"""Financial Insights Deep Agent — powered by a fine-tuned model on RHOAI.

Adapts the content-builder-agent pattern from langchain-ai/deepagents to wrap
a fine-tuned Qwen3-4B served via vLLM on RHOAI with 15 financial tools from
the FinanceInsights MCP server.

The agent is configured through filesystem primitives:
  - AGENTS.md         — agent identity and tool usage guidelines (always loaded)
  - skills/           — on-demand workflows for portfolio, market, and trade tasks
  - financial_tools   — 15 @tool wrappers calling the MCP server

Quick start:
  1. Start the MCP server:  cd demo_server && python server.py
  2. Port-forward vLLM:     oc port-forward -n financial-agent deployment/financial-agent-lora-predictor 8000:8080
  3. Headless:              uv run python 07_deep_agent.py "What is the price of AAPL?"
  4. Interactive:           uv run langgraph dev --allow-blocking

Environment variables (set in .env):
  MODEL_ENDPOINT    vLLM endpoint URL        (default: http://localhost:8000/v1)
  MODEL_NAME        Model name at endpoint   (default: financial-agent)
  MCP_SERVER_URL    MCP server URL           (default: http://localhost:8009/mcp)
  OPENAI_API_KEY    API key for vLLM         (default: not-needed)
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from financial_tools import ALL_TOOLS

load_dotenv()

EXAMPLE_DIR = Path(__file__).parent
console = Console()


def create_financial_agent():
    """Create a Financial Insights Deep Agent with full middleware stack.

    Requires vLLM to be deployed with --max-model-len=16384 or higher, since
    the Deep Agents middleware (todo, filesystem, skills, memory, subagents,
    summarization) adds ~3500 tokens of system prompt.
    """
    model = ChatOpenAI(
        base_url=os.environ.get("MODEL_ENDPOINT", "http://localhost:8000/v1"),
        model=os.environ.get("MODEL_NAME", "financial-agent"),
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        temperature=0.1,
        max_tokens=4096,
    )

    return create_deep_agent(
        model=model,
        tools=ALL_TOOLS,
        memory=["./AGENTS.md"],
        skills=["./skills/"],
        subagents=[],
        backend=FilesystemBackend(root_dir=EXAMPLE_DIR),
    )


# Module-level export for langgraph.json ("./07_deep_agent.py:agent")
agent = create_financial_agent()


class AgentDisplay:
    """Rich-based display for agent progress, adapted from content_writer.py."""

    def __init__(self):
        self.printed_count = 0
        self.spinner = Spinner("dots", text="Thinking...")

    def update_status(self, status: str):
        self.spinner = Spinner("dots", text=status)

    def print_message(self, msg):
        if isinstance(msg, HumanMessage):
            console.print(Panel(str(msg.content), title="You", border_style="blue"))

        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = "\n".join(text_parts)

            if content and content.strip():
                console.print(
                    Panel(Markdown(content), title="Agent", border_style="green")
                )

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    if name == "write_file":
                        path = args.get("file_path", "file")
                        console.print(f"  [bold yellow]>> Writing:[/] {path}")
                    else:
                        args_summary = ", ".join(
                            f"{k}={v!r}" for k, v in list(args.items())[:3]
                        )
                        console.print(
                            f"  [bold cyan]>> {name}[/]({args_summary})"
                        )
                        self.update_status(f"Calling {name}...")

        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "")
            content = msg.content or ""
            if len(content) > 120:
                content = content[:120] + "..."
            console.print(f"  [green]✓ {name}:[/] {content}")


async def main():
    """Run the financial agent with streaming output."""
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = "What is the current price of AAPL and how does it compare to the broader market?"

    console.print()
    console.print("[bold blue]Financial Insights Deep Agent[/]")
    console.print(
        f"[dim]Model: {os.environ.get('MODEL_NAME', 'financial-agent')} "
        f"@ {os.environ.get('MODEL_ENDPOINT', 'http://localhost:8000/v1')}[/]"
    )
    console.print(f"[dim]MCP:   {os.environ.get('MCP_SERVER_URL', 'http://localhost:8009/mcp')}[/]")
    console.print(f"[dim]Task:  {task}[/]")
    console.print()

    display = AgentDisplay()

    with Live(display.spinner, console=console, refresh_per_second=10, transient=True) as live:
        async for chunk in agent.astream(
            {"messages": [("user", task)]},
            config={"configurable": {"thread_id": "financial-agent-demo"}},
            stream_mode="values",
        ):
            if "messages" in chunk:
                messages = chunk["messages"]
                if len(messages) > display.printed_count:
                    live.stop()
                    for msg in messages[display.printed_count :]:
                        display.print_message(msg)
                    display.printed_count = len(messages)
                    live.start()
                    live.update(display.spinner)

    console.print()
    console.print("[bold green]✓ Done![/]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")
