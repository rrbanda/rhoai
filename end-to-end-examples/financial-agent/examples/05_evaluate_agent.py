"""Evaluate a fine-tuned financial agent's tool-calling accuracy.

Generates a financial benchmark by connecting to the FinanceInsights MCP server
and running the MCP distillation flow at multiple complexity levels, then
evaluates the fine-tuned model's tool-use quality with both programmatic metrics
and an LLM-as-judge.

Pipeline stages:
  1. Generate (or load) a benchmark with tasks at 2, 4, and 8 tools per question
  2. Run the fine-tuned model against each benchmark task via the vLLM endpoint
  3. Compute programmatic metrics: tool_recall, tool_precision, order_match, param_match
  4. Run LLM-as-judge scoring on 6 dimensions:
     task_fulfillment, grounding, tool_appropriateness,
     parameter_accuracy, dependency_awareness, parallelism_and_efficiency
  5. Output a results table and save to JSON

Usage:
    export OPENAI_API_KEY="your-api-key"
    export JUDGE_MODEL="openai/gpt-4o"

    python 05_evaluate_agent.py \\
        --model-endpoint http://financial-agent-lora-predictor.financial-agent.svc.cluster.local:8080 \\
        --mcp-server-url http://localhost:8009 \\
        --output evaluation_results.json

    # Skip benchmark generation with a pre-generated file:
    python 05_evaluate_agent.py \\
        --model-endpoint http://localhost:8080 \\
        --benchmark-file benchmark_tasks.jsonl \\
        --output evaluation_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from sdg_hub import Flow, FlowRegistry


# ---------------------------------------------------------------------------
# Trace utilities
# ---------------------------------------------------------------------------

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
                step: dict = {"name": entry["name"], "input": entry.get("tool_input", {})}
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


def format_trace_for_judge(
    tool_trace: list[dict],
    max_args_len: int = 300,
    max_output_len: int = 200,
) -> str:
    """Format a canonical tool trace as a readable string for the judge prompt."""
    if not tool_trace:
        return "  No tool calls made."
    lines = []
    for i, step in enumerate(tool_trace, 1):
        args_str = json.dumps(step.get("input", {}), ensure_ascii=False)
        if len(args_str) > max_args_len:
            args_str = args_str[:max_args_len] + "..."
        line = f"  [{i}] {step['name']}({args_str})"
        output = step.get("output")
        if output:
            out_str = str(output)
            if len(out_str) > max_output_len:
                out_str = out_str[:max_output_len] + "..."
            line += f"\n      -> {out_str}"
        lines.append(line)
    return "\n".join(lines)


def compute_tool_metrics(
    model_tools: list[str],
    expert_tools: list[str],
    model_trace: list[dict] | None = None,
    expert_trace: list[dict] | None = None,
) -> dict[str, float]:
    """Compute tool recall, precision, order match, and parameter similarity."""
    model_set, expert_set = set(model_tools), set(expert_tools)
    if not expert_set:
        return {"tool_recall": 1.0, "tool_precision": 1.0, "order_match": 1.0, "param_match": 1.0}

    intersection = model_set & expert_set
    recall = len(intersection) / len(expert_set)
    precision = len(intersection) / len(model_set) if model_set else 0.0

    m, n = len(model_tools), len(expert_tools)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (
                dp[i - 1][j - 1] + 1
                if model_tools[i - 1] == expert_tools[j - 1]
                else max(dp[i - 1][j], dp[i][j - 1])
            )
    order = dp[m][n] / len(expert_tools)

    param_match = 0.0
    if model_trace and expert_trace:
        matched, total = 0, 0
        available = list(range(len(model_trace)))
        for et in expert_trace:
            for idx in available:
                mt = model_trace[idx]
                if mt.get("name") == et.get("name"):
                    total += 1
                    available.remove(idx)
                    e_in = et.get("input", {})
                    m_in = mt.get("input", {})
                    if not e_in and not m_in:
                        matched += 1
                    elif isinstance(e_in, dict) and isinstance(m_in, dict):
                        e_keys = set(e_in.keys())
                        m_keys = set(m_in.keys())
                        if e_keys:
                            key_ov = len(e_keys & m_keys) / len(e_keys)
                            val_m = sum(
                                1
                                for k in e_keys & m_keys
                                if str(e_in[k]).lower() == str(m_in[k]).lower()
                            )
                            val_r = val_m / len(e_keys & m_keys) if (e_keys & m_keys) else 0
                            matched += (key_ov + val_r) / 2
                    break
        param_match = matched / total if total > 0 else 0.0

    return {
        "tool_recall": round(recall, 3),
        "tool_precision": round(precision, 3),
        "order_match": round(order, 3),
        "param_match": round(param_match, 3),
    }


JUDGE_DIMENSIONS = [
    "task_fulfillment", "grounding", "tool_appropriateness",
    "parameter_accuracy", "dependency_awareness", "parallelism_and_efficiency",
]
ZERO_JUDGE = {d: 0 for d in JUDGE_DIMENSIONS}
ZERO_METRICS = {"tool_recall": 0.0, "tool_precision": 0.0, "order_match": 0.0, "param_match": 0.0}
ZERO_RESULT = {**ZERO_METRICS, **ZERO_JUDGE}


# ---------------------------------------------------------------------------
# Benchmark generation
# ---------------------------------------------------------------------------

def generate_benchmark(
    mcp_server_url: str,
    langflow_url: str,
    langflow_api_key: str | None,
    teacher_model: str,
    teacher_api_key: str,
    output_path: Path,
) -> pd.DataFrame:
    """Generate a multi-complexity benchmark from the FinanceInsights MCP server."""
    import nest_asyncio
    nest_asyncio.apply()

    from sdg_hub import Flow, FlowRegistry

    FlowRegistry.discover_flows()
    flow_yaml = FlowRegistry.get_flow_path("MCP Server Distillation")
    if flow_yaml is None:
        print("ERROR: MCP Server Distillation flow not found in registry.")
        print("Ensure sdg-hub is installed: pip install sdg-hub[examples]")
        sys.exit(1)

    flow = Flow.from_yaml(flow_yaml)
    flow.set_model_config(model=teacher_model, api_key=teacher_api_key)

    agent_kwargs = {"agent_framework": "langflow", "agent_url": langflow_url}
    if langflow_api_key:
        agent_kwargs["agent_api_key"] = langflow_api_key
    flow.set_agent_config(**agent_kwargs)
    flow.set_agent_config(timeout=300, blocks=["explore_server"])

    # Build input dataset from the co-located data generation script
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "gen", Path(__file__).parent / "01_generate_tool_data.py"
    )
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    dataset = mod.build_input_dataset()

    all_tasks = []
    for tools_per_question in [2, 4, 8]:
        print(f"  Generating benchmark tasks (tools_per_question={tools_per_question})...")
        result = flow.generate(
            dataset,
            runtime_params={
                "multiply_tool_rows": {"num_samples": 5},
                "sample_tools": {"num_samples": tools_per_question},
            },
        )
        result_df = result.to_pandas() if hasattr(result, "to_pandas") else result
        result_df["complexity"] = tools_per_question
        result_df["server"] = "FinanceInsights"
        all_tasks.append(result_df)

    benchmark_df = pd.concat(all_tasks, ignore_index=True)
    benchmark_df.to_json(output_path, orient="records", lines=True)
    print(f"  Benchmark saved: {output_path} ({len(benchmark_df)} tasks)")
    return benchmark_df


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def run_model_inference(
    endpoint: str,
    question: str,
    tools: list[dict],
) -> dict:
    """Send a tool-calling request to the vLLM endpoint and parse the response."""
    import requests

    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]

    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": question}],
        "tools": openai_tools,
        "tool_choice": "auto",
        "max_tokens": 1024,
        "temperature": 0.1,
    }

    resp = requests.post(f"{endpoint}/v1/chat/completions", json=payload, timeout=120)
    if resp.status_code != 200:
        return {"text": "", "tool_trace": [], "error": resp.text}

    data = resp.json()
    message = data["choices"][0]["message"]
    text = message.get("content", "") or ""
    trace = []

    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            trace.append({"name": func.get("name", ""), "input": args})

    return {"text": text, "tool_trace": trace}


# ---------------------------------------------------------------------------
# EvalHub integration (RHOAI 3.5 TP)
# ---------------------------------------------------------------------------

def submit_to_evalhub(results_df: pd.DataFrame, model_name: str) -> bool:
    """Submit evaluation results to EvalHub if the SDK is available."""
    try:
        from evalhub import EvalHubClient

        evalhub_url = os.environ.get("EVALHUB_URL")
        evalhub_api_key = os.environ.get("EVALHUB_API_KEY")
        if not evalhub_url:
            print("  EVALHUB_URL not set, skipping EvalHub submission.")
            return False

        client = EvalHubClient(url=evalhub_url, api_key=evalhub_api_key)
        client.submit_evaluation(
            model=model_name,
            task="financial-agent-tool-calling",
            results=results_df.to_dict("records"),
        )
        print(f"  Submitted {len(results_df)} results to EvalHub.")
        return True
    except ImportError:
        print("  EvalHub SDK not installed (pip install evalhub). Skipping submission.")
        return False
    except Exception as e:
        print(f"  EvalHub submission failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned financial agent's tool-calling accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-endpoint",
        required=True,
        help="vLLM endpoint URL (e.g. http://financial-agent-lora-predictor.financial-agent.svc.cluster.local:8080)",
    )
    parser.add_argument(
        "--mcp-server-url",
        default=os.environ.get("MCP_SERVER_URL", "http://localhost:8009"),
        help="FinanceInsights MCP server URL for benchmark generation (default: $MCP_SERVER_URL)",
    )
    parser.add_argument(
        "--benchmark-file",
        default=None,
        help="Path to pre-generated benchmark JSONL (skip generation if provided)",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Output JSON path (default: evaluation_results.json)",
    )
    parser.add_argument(
        "--use-evalhub",
        action="store_true",
        help="Submit results to EvalHub (RHOAI 3.5 TP) if SDK is available",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file for configuration (default: .env)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)

    judge_model = os.getenv("JUDGE_MODEL", "openai/gpt-4o")
    teacher_model = os.getenv("TEACHER_MODEL", "openai/gpt-5.2")
    teacher_api_key = os.getenv("TEACHER_API_KEY", "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    langflow_url = os.getenv("LANGFLOW_URL", "")
    langflow_api_key = os.getenv("LANGFLOW_API_KEY")

    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable must be set.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Financial Agent — Tool-Calling Evaluation")
    print("=" * 60)
    print(f"  Model endpoint:   {args.model_endpoint}")
    print(f"  Judge model:      {judge_model}")
    print(f"  Output:           {args.output}")
    print("=" * 60)

    # -- Load or generate benchmark -------------------------------------------
    if args.benchmark_file:
        benchmark_path = Path(args.benchmark_file)
        if not benchmark_path.exists():
            print(f"ERROR: Benchmark file not found: {benchmark_path}", file=sys.stderr)
            sys.exit(1)
        benchmark_df = pd.read_json(benchmark_path, orient="records", lines=True)
        print(f"\nLoaded benchmark: {len(benchmark_df)} tasks from {benchmark_path}")
    else:
        if not langflow_url:
            print("ERROR: LANGFLOW_URL required for benchmark generation.", file=sys.stderr)
            print("Set it in .env or provide --benchmark-file to skip generation.")
            sys.exit(1)
        if not teacher_api_key:
            print("ERROR: TEACHER_API_KEY required for benchmark generation.", file=sys.stderr)
            sys.exit(1)

        print("\nGenerating benchmark from FinanceInsights MCP server...")
        benchmark_path = Path(args.output).parent / "benchmark_tasks.jsonl"
        benchmark_df = generate_benchmark(
            mcp_server_url=args.mcp_server_url,
            langflow_url=langflow_url,
            langflow_api_key=langflow_api_key,
            teacher_model=teacher_model,
            teacher_api_key=teacher_api_key,
            output_path=benchmark_path,
        )

    # -- Load the judge flow --------------------------------------------------
    FlowRegistry.discover_flows()
    eval_flow = Flow.from_yaml(FlowRegistry.get_flow_path("Agent Tool-Use Evaluation"))
    eval_flow.set_model_config(model=judge_model, api_key=api_key)
    print(f"  Judge model: {judge_model}")

    # -- Resolve tool list for inference --------------------------------------
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "gen", Path(__file__).parent / "01_generate_tool_data.py"
    )
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    tool_dataset = mod.build_input_dataset()
    all_tools = tool_dataset["tool_list"].iloc[0]

    # -- Run model against benchmark ------------------------------------------
    all_results = []
    complexities = sorted(benchmark_df["complexity"].unique()) if "complexity" in benchmark_df.columns else [0]

    for complexity in complexities:
        if "complexity" in benchmark_df.columns:
            tasks = benchmark_df[benchmark_df["complexity"] == complexity]
        else:
            tasks = benchmark_df
        print(f"\n{'=' * 60}")
        print(f"Complexity: {complexity} tools/question ({len(tasks)} tasks)")
        print(f"{'=' * 60}")

        judge_rows = []
        task_meta = []

        for idx in range(len(tasks)):
            task = tasks.iloc[idx]
            question = task["question"]

            response = run_model_inference(args.model_endpoint, question, all_tools)
            m_trace = response["tool_trace"]
            m_text = response["text"]
            m_tools = extract_tool_names(m_trace)

            e_tools = task.get("expert_tools", [])
            e_trace = task.get("expert_tool_trace", [])
            if isinstance(e_tools, str):
                e_tools = json.loads(e_tools)
            if isinstance(e_trace, str):
                e_trace = json.loads(e_trace)

            metrics = compute_tool_metrics(m_tools, e_tools, m_trace, e_trace)

            judge_rows.append(
                {
                    "question": question,
                    "expert_answer_truncated": str(task.get("expert_answer", ""))[:2000],
                    "expert_trace_formatted": format_trace_for_judge(e_trace),
                    "model_answer": m_text[:2000],
                    "model_trace_formatted": format_trace_for_judge(m_trace),
                }
            )
            task_meta.append(
                {"complexity": complexity, "task_idx": idx, **metrics}
            )

            if (idx + 1) % 5 == 0:
                print(f"  Processed {idx + 1}/{len(tasks)} tasks", end="\r", flush=True)

        print(f"  Processed {len(tasks)}/{len(tasks)} tasks")

        # LLM-as-judge scoring
        if judge_rows:
            try:
                judge_ds = Dataset.from_list(judge_rows)
                judge_result = eval_flow.generate(judge_ds)
                judge_df = (
                    judge_result.to_pandas()
                    if hasattr(judge_result, "to_pandas")
                    else pd.DataFrame(judge_result)
                )
                for i, meta in enumerate(task_meta):
                    scores = {}
                    for col in JUDGE_DIMENSIONS:
                        val = judge_df[col].iloc[i] if col in judge_df.columns else 0
                        try:
                            scores[col] = int(val)
                        except (ValueError, TypeError):
                            scores[col] = 0
                    all_results.append({**meta, **scores})
                print(f"  Judge scoring complete")
            except Exception as e:
                print(f"  Judge failed: {e}")
                for meta in task_meta:
                    all_results.append({**meta, **ZERO_JUDGE})

    # -- Save results ---------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_json(output_path, orient="records", indent=2)
    print(f"\nSaved {len(results_df)} results to {output_path}")

    # -- Print summary table --------------------------------------------------
    tool_cols = ["tool_recall", "tool_precision", "order_match", "param_match"]
    all_cols = tool_cols + JUDGE_DIMENSIONS

    print(f"\n{'=' * 80}")
    print("Results by Complexity")
    print(f"{'=' * 80}")

    if "complexity" in results_df.columns:
        summary = results_df.groupby("complexity")[all_cols].mean()
        for col in JUDGE_DIMENSIONS:
            summary[col] = summary[col] / 10.0
        summary["overall"] = summary.mean(axis=1)
        print(summary.round(3).to_string())
    else:
        means = results_df[all_cols].mean()
        for col in JUDGE_DIMENSIONS:
            means[col] = means[col] / 10.0
        means["overall"] = means.mean()
        print(means.round(3).to_string())

    # Overall averages
    overall_metrics = results_df[tool_cols].mean()
    overall_judge = results_df[JUDGE_DIMENSIONS].mean() / 10.0
    combined = pd.concat([overall_metrics, overall_judge])
    combined["overall"] = combined.mean()

    print(f"\n{'=' * 80}")
    print("Aggregate Scores")
    print(f"{'=' * 80}")
    for metric, val in combined.items():
        print(f"  {metric:30s} {val:.3f}")

    # -- EvalHub submission (optional) ----------------------------------------
    if args.use_evalhub:
        model_name = args.model_endpoint.split("/")[-1] or "financial-agent"
        submit_to_evalhub(results_df, model_name)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
