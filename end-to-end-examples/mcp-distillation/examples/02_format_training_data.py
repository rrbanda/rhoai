"""Format distillation output into tool-calling JSONL for training.

Converts the raw pipeline output (Parquet from step 01) into the structured
tool-calling conversation format used for supervised fine-tuning:

  [system]    Tool declarations (all MCP server tools)
  [user]      Natural language question
  [assistant] tool_calls: [{function: {name, arguments}}]
  [tool]      Tool response (with tool_call_id)
  ...         (repeated for multi-step tool use)
  [assistant] Final synthesized answer

Each line in the output JSONL file is a complete training example with a
"messages" field containing the full conversation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format distillation output into function-calling JSONL"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to distillation output Parquet (default: OUTPUT_DIR/distillation_output.parquet)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path for output JSONL (default: OUTPUT_DIR/training_data.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "./generated_data"),
        help="Base directory for input/output files (default: ./generated_data)",
    )
    return parser.parse_args()


def format_tool_trace(
    tool_trace: list[dict],
    tool_list: list[dict],
    question: str,
) -> dict | None:
    """Convert a single row's tool trace into a function-calling conversation.

    Parameters
    ----------
    tool_trace
        Structured tool trace from the pipeline's AgentResponseExtractorBlock.
        Each entry has a "role" field (assistant/tool) with tool call details.
    tool_list
        Full list of MCP server tool schemas for the system message.
    question
        The user's question that initiated this trajectory.

    Returns
    -------
    dict or None
        A {"messages": [...]} dict ready for JSONL, or None if the trace
        is empty or malformed.
    """
    if not tool_trace:
        return None

    # System message: declare all available tools
    tool_declarations = []
    for tool in tool_list:
        tool_declarations.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {}),
            },
        })

    system_content = (
        "You are a helpful assistant with access to the following tools:\n\n"
        + json.dumps(tool_declarations, indent=2)
    )

    messages = [{"role": "system", "content": system_content}]

    # User message
    messages.append({"role": "user", "content": question})

    # Convert each trace step into the appropriate message format
    call_counter = 0
    for step in tool_trace:
        role = step.get("role", "")

        if role == "assistant" and step.get("tool_calls"):
            tool_calls_list = []
            for tc in step["tool_calls"]:
                func = tc.get("function", {})
                arguments = func.get("arguments", {})
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)
                call_counter += 1
                tool_calls_list.append({
                    "id": f"call_{call_counter}",
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": arguments,
                    },
                })
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls_list,
            })

        elif role == "tool":
            content = step.get("content", "")
            if isinstance(content, dict):
                content = json.dumps(content)
            messages.append({
                "role": "tool",
                "content": content,
                "tool_call_id": f"call_{call_counter}",
            })

        elif role == "assistant":
            content = step.get("content", "")
            if content:
                messages.append({
                    "role": "assistant",
                    "content": content,
                })

    # Validate: must have at least system + user + one tool call
    has_tool_calls = any(m.get("tool_calls") for m in messages)
    if not has_tool_calls and sum(1 for m in messages if m["role"] == "assistant") < 2:
        return None

    return {"messages": messages}


def main() -> None:
    load_dotenv()
    args = parse_args()

    output_dir = Path(args.output_dir)

    input_file = Path(args.input_file) if args.input_file else output_dir / "distillation_output.parquet"
    output_file = Path(args.output_file) if args.output_file else output_dir / "training_data.jsonl"

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Run 01_generate_tool_data.py first to generate the distillation output.")
        sys.exit(1)

    # -- Load the distillation output -----------------------------------------
    print(f"Loading distillation output: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Verify required columns exist
    required_cols = ["extract_agent_text_tool_trace", "tool_list", "question"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print("The distillation output may be incomplete. Re-run step 01.")
        sys.exit(1)

    # -- Format each row into a training conversation -------------------------
    print("\nFormatting tool traces into function-calling conversations...")

    formatted = []
    skipped = 0
    total_tool_calls = 0

    for idx, row in df.iterrows():
        tool_trace = row["extract_agent_text_tool_trace"]
        tool_list = row["tool_list"]
        question = row["question"]

        result = format_tool_trace(tool_trace, tool_list, question)

        if result is None:
            skipped += 1
            continue

        # Count tool calls in this example
        n_calls = sum(1 for m in result["messages"] if m.get("tool_calls"))
        total_tool_calls += n_calls

        formatted.append(result)

    # -- Write JSONL ----------------------------------------------------------
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for example in formatted:
            f.write(json.dumps(example) + "\n")

    # -- Print statistics -----------------------------------------------------
    print("-" * 60)
    print("Formatting complete!")
    print(f"  Total input rows:    {len(df)}")
    print(f"  Formatted examples:  {len(formatted)}")
    print(f"  Skipped (malformed): {skipped}")
    print(f"  Total tool calls:    {total_tool_calls}")
    if formatted:
        avg_calls = total_tool_calls / len(formatted)
        avg_msgs = sum(len(e["messages"]) for e in formatted) / len(formatted)
        print(f"  Avg tool calls/example: {avg_calls:.1f}")
        print(f"  Avg messages/example:   {avg_msgs:.1f}")
    print(f"\n  Output: {output_file}")


if __name__ == "__main__":
    main()
