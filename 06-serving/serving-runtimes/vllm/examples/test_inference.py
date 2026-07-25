"""Test inference against a deployed vLLM endpoint on RHOAI 3.4.

Sends requests using the OpenAI-compatible chat completions API exposed
by vLLM.  Supports both streaming and non-streaming modes, and reports
token counts and latency measurements.

Requirements:
    pip install requests

Usage:
    python test_inference.py --endpoint https://my-model-ns.apps.cluster.example.com
    python test_inference.py --endpoint http://localhost:8080 --stream
    python test_inference.py --endpoint $INFERENCE_URL --prompt "Summarize quantum computing"

Environment variables:
    INFERENCE_URL       Model endpoint URL (overridden by --endpoint)
    INFERENCE_TOKEN     Bearer token for authenticated endpoints
    MODEL_NAME          Served model name (overridden by --model)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests as http_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test inference against a vLLM endpoint (OpenAI-compatible API).",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Model endpoint URL (default: $INFERENCE_URL)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Served model name (default: $MODEL_NAME or 'default')",
    )
    parser.add_argument(
        "--prompt",
        default="What are the key benefits of fine-tuning a language model?",
        help="User message to send",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system message prepended to the conversation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling (default: 0.95)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming mode (tokens printed as they arrive)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of requests to send for benchmarking (default: 1)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer token for authentication (default: $INFERENCE_TOKEN)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )
    return parser.parse_args()


def build_headers(token: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def build_messages(prompt: str, system_prompt: str | None) -> list[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def request_non_streaming(
    url: str,
    headers: dict,
    payload: dict,
    timeout: int,
) -> tuple[str, dict, float]:
    """Send a non-streaming request and return content, usage, and latency."""
    start = time.time()
    resp = http_client.post(url, headers=headers, json=payload, timeout=timeout)
    latency = time.time() - start

    if resp.status_code != 200:
        print(f"ERROR: HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, usage, latency


def request_streaming(
    url: str,
    headers: dict,
    payload: dict,
    timeout: int,
) -> tuple[str, float, float]:
    """Send a streaming request.  Returns content, time-to-first-token, and total latency."""
    payload["stream"] = True
    start = time.time()
    resp = http_client.post(
        url, headers=headers, json=payload, timeout=timeout, stream=True
    )

    if resp.status_code != 200:
        print(f"ERROR: HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    content_parts: list[str] = []
    ttft: float | None = None

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[len("data: "):]
        if data_str.strip() == "[DONE]":
            break

        chunk = json.loads(data_str)
        delta = chunk["choices"][0].get("delta", {})
        token = delta.get("content", "")
        if token:
            if ttft is None:
                ttft = time.time() - start
            content_parts.append(token)
            print(token, end="", flush=True)

    total_latency = time.time() - start
    print()
    return "".join(content_parts), ttft or total_latency, total_latency


def main() -> None:
    args = parse_args()

    endpoint = args.endpoint or os.getenv("INFERENCE_URL")
    if not endpoint:
        print("ERROR: Provide --endpoint or set $INFERENCE_URL", file=sys.stderr)
        sys.exit(1)

    model = args.model or os.getenv("MODEL_NAME", "default")
    token = args.token or os.getenv("INFERENCE_TOKEN")
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"

    print("=" * 60)
    print("vLLM Inference Test")
    print("=" * 60)
    print(f"  Endpoint:           {url}")
    print(f"  Model:              {model}")
    print(f"  Streaming:          {args.stream}")
    print(f"  Max tokens:         {args.max_tokens}")
    print(f"  Temperature:        {args.temperature}")
    print(f"  Repeat count:       {args.repeat}")
    print("=" * 60)

    headers = build_headers(token)
    messages = build_messages(args.prompt, args.system_prompt)

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    latencies: list[float] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"\n--- Request {i + 1}/{args.repeat} ---")

        if args.stream:
            content, ttft, latency = request_streaming(
                url, headers, payload, args.timeout
            )
            print(f"  TTFT:               {ttft * 1000:.0f}ms")
            print(f"  Total latency:      {latency * 1000:.0f}ms")
            approx_tokens = len(content.split())
            print(f"  Approx tokens:      ~{approx_tokens}")
        else:
            content, usage, latency = request_non_streaming(
                url, headers, payload, args.timeout
            )
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

            print(f"\n  Response:\n    {content[:500]}")
            print(f"\n  Prompt tokens:      {prompt_tokens}")
            print(f"  Completion tokens:  {completion_tokens}")
            print(f"  Latency:            {latency * 1000:.0f}ms")
            if completion_tokens > 0:
                tps = completion_tokens / latency
                print(f"  Tokens/sec:         {tps:.1f}")

        latencies.append(latency)

    if args.repeat > 1:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        avg_ms = sum(latencies) * 1000 / len(latencies)
        min_ms = min(latencies) * 1000
        max_ms = max(latencies) * 1000
        print(f"  Requests:           {args.repeat}")
        print(f"  Avg latency:        {avg_ms:.0f}ms")
        print(f"  Min latency:        {min_ms:.0f}ms")
        print(f"  Max latency:        {max_ms:.0f}ms")
        if total_completion_tokens > 0:
            avg_tps = total_completion_tokens / sum(latencies)
            print(f"  Avg tokens/sec:     {avg_tps:.1f}")
        print(f"  Total tokens:       {total_prompt_tokens + total_completion_tokens}")


if __name__ == "__main__":
    main()
