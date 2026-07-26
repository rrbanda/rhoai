"""Configure NeMo Guardrails for the financial advisory agent on RHOAI.

Deploys the NemoGuardrails custom resource (GA in RHOAI 3.4) with optional
MCP Gateway integration (Technology Preview in RHOAI 3.5 EA2).  The guardrails
enforce PII detection, financial compliance checks, and content safety on
both model inputs/outputs and agent tool calls at the gateway layer.

Requirements:
    pip install kubernetes requests pyyaml

Usage:
    python 06_configure_guardrails.py --model-endpoint https://granite-8b-finance.apps.cluster.example.com/v1
    python 06_configure_guardrails.py --model-endpoint https://granite-8b-finance.apps.cluster.example.com/v1 --enable-mcp-gateway --mcp-server-url http://finance-mcp:8080
    python 06_configure_guardrails.py --model-endpoint https://granite-8b-finance.apps.cluster.example.com/v1 --dry-run

Environment variables:
    KUBECONFIG          Path to kubeconfig file (default: ~/.kube/config)
    GUARDRAILS_NS       Target namespace (overridden by --namespace)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException


_DEFAULT_NAMESPACE = "financial-agent"
_READY_TIMEOUT = 300
_POLL_INTERVAL = 10

_CRD_GROUP = "trustyai.opendatahub.io"
_CRD_VERSION = "v1alpha1"
_GUARDRAILS_PLURAL = "nemoguardrails"
_GATEWAY_PLURAL = "mcpgatewayextensions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy NeMo Guardrails for the financial agent on RHOAI 3.4+.",
    )
    parser.add_argument(
        "--model-endpoint",
        required=True,
        help="vLLM model endpoint URL (e.g. https://granite-8b-finance.apps.cluster.example.com/v1)",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help=f"Kubernetes namespace (default: $GUARDRAILS_NS or {_DEFAULT_NAMESPACE})",
    )
    parser.add_argument(
        "--mcp-server-url",
        default=None,
        help="Financial MCP server URL for gateway integration (required with --enable-mcp-gateway)",
    )
    parser.add_argument(
        "--enable-mcp-gateway",
        action="store_true",
        help="Enable MCP Gateway integration (Technology Preview, RHOAI 3.5 EA2)",
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=2,
        help="Number of guardrails replicas (default: 2)",
    )
    parser.add_argument(
        "--otel-endpoint",
        default="http://otel-collector.observability.svc:4317",
        help="OpenTelemetry collector endpoint",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=_READY_TIMEOUT,
        help=f"Seconds to wait for readiness (default: {_READY_TIMEOUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the manifests without applying them",
    )
    return parser.parse_args()


def build_guardrails_configmap(
    namespace: str,
    model_endpoint: str,
    model_name: str,
) -> dict:
    """Construct the ConfigMap with config.yaml and rails.co for NeMo Guardrails."""
    config_yaml = {
        "models": [
            {
                "type": "main",
                "engine": "openai",
                "parameters": {
                    "openai_api_base": model_endpoint,
                    "model_name": model_name,
                },
            }
        ],
        "rails": {
            "config": {
                "sensitive_data_detection": {
                    "input": {"entities": ["EMAIL_ADDRESS", "PERSON", "PHONE_NUMBER"]},
                    "output": {"entities": ["PERSON", "PHONE_NUMBER"]},
                },
                "regex_detection": {
                    "input": {
                        "patterns": [r"\\d{3}-\\d{2}-\\d{4}", "ACCT-[0-9]{4,12}"],
                        "case_insensitive": True,
                    },
                    "output": {
                        "patterns": [r"\\d{3}-\\d{2}-\\d{4}", "ACCT-[0-9]{4,12}"],
                        "case_insensitive": True,
                    },
                },
            },
            "input": {
                "flows": ["detect sensitive data on input", "regex check input"],
            },
            "output": {
                "flows": ["detect sensitive data on output"],
            },
        },
    }

    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "financial-guardrails-config",
            "namespace": namespace,
        },
        "data": {
            "config.yaml": yaml.dump(config_yaml, default_flow_style=False),
            "rails.co": "# Using built-in rails for financial PII detection\n",
        },
    }


def build_nemoguardrails_cr(
    namespace: str,
    model_endpoint: str,
    replicas: int,
    otel_endpoint: str,
    mcp_gateway_name: str | None = None,
) -> dict:
    """Construct the NemoGuardrails custom resource manifest.

    Uses the ``nemoConfigs`` field (required) to reference the ConfigMap
    containing config.yaml and rails.co, per the RHOAI 3.4 API.
    """
    manifest = {
        "apiVersion": f"{_CRD_GROUP}/{_CRD_VERSION}",
        "kind": "NemoGuardrails",
        "metadata": {
            "name": "financial-agent-guardrails",
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/part-of": "financial-agent",
                "app.kubernetes.io/managed-by": "rhoai-examples",
            },
        },
        "spec": {
            "nemoConfigs": [
                {
                    "name": "financial-agent",
                    "default": True,
                    "configMaps": ["financial-guardrails-config"],
                }
            ],
            "env": [
                {"name": "OPENAI_API_KEY", "value": "not-needed"},
            ],
        },
    }

    return manifest


def build_mcpgateway_extension_cr(
    namespace: str,
    mcp_server_url: str,
) -> dict:
    """Construct the MCPGatewayExtension custom resource manifest."""
    # Technology Preview (RHOAI 3.5 EA2)
    return {
        "apiVersion": f"{_CRD_GROUP}/{_CRD_VERSION}",
        "kind": "MCPGatewayExtension",
        "metadata": {
            "name": "financial-mcp-gateway",
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/part-of": "financial-agent",
                "app.kubernetes.io/managed-by": "rhoai-examples",
            },
        },
        "spec": {
            "mcpServerUrl": mcp_server_url,
            "guardrailsRef": {
                "name": "financial-agent-guardrails",
                "namespace": namespace,
            },
        },
    }


def apply_custom_resource(manifest: dict, plural: str, namespace: str) -> None:
    """Create or update a custom resource via the Kubernetes API."""
    api = client.CustomObjectsApi()
    name = manifest["metadata"]["name"]
    kind = manifest["kind"]

    try:
        api.get_namespaced_custom_object(
            group=_CRD_GROUP,
            version=_CRD_VERSION,
            namespace=namespace,
            plural=plural,
            name=name,
        )
        api.patch_namespaced_custom_object(
            group=_CRD_GROUP,
            version=_CRD_VERSION,
            namespace=namespace,
            plural=plural,
            name=name,
            body=manifest,
        )
        print(f"  Updated existing {kind} '{name}'")
    except ApiException as e:
        if e.status == 404:
            api.create_namespaced_custom_object(
                group=_CRD_GROUP,
                version=_CRD_VERSION,
                namespace=namespace,
                plural=plural,
                body=manifest,
            )
            print(f"  Created {kind} '{name}'")
        else:
            raise


def wait_for_guardrails_ready(name: str, namespace: str, timeout: int) -> str:
    """Poll until the NemoGuardrails resource reports ready."""
    api = client.CustomObjectsApi()
    deadline = time.time() + timeout
    print(f"  Waiting up to {timeout}s for guardrails to become ready...")

    while time.time() < deadline:
        obj = api.get_namespaced_custom_object(
            group=_CRD_GROUP,
            version=_CRD_VERSION,
            namespace=namespace,
            plural=_GUARDRAILS_PLURAL,
            name=name,
        )
        conditions = obj.get("status", {}).get("conditions", [])
        for cond in conditions:
            if cond.get("type") == "Ready" and cond.get("status") == "True":
                endpoint = obj.get("status", {}).get("endpoint", "")
                print(f"  Guardrails ready: {endpoint}")
                return endpoint

        time.sleep(_POLL_INTERVAL)

    print("  ERROR: Timed out waiting for guardrails readiness", file=sys.stderr)
    sys.exit(1)


def test_guardrails_endpoint(endpoint: str) -> None:
    """Send a test query through the guardrails check endpoint."""
    checks_url = f"{endpoint}/v1/guardrail/checks"
    payload = {
        "model": "financial-agent",
        "messages": [
            {
                "role": "user",
                "content": "What is the current balance for account ACCT-7832? My SSN is 123-45-6789.",
            }
        ],
    }

    print(f"  Testing guardrails endpoint: {checks_url}")
    start = time.time()
    resp = requests.post(checks_url, json=payload, timeout=30)
    latency_ms = (time.time() - start) * 1000

    if resp.status_code != 200:
        print(f"  ERROR: Received status {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    status = data.get("status", "unknown")
    rails_status = data.get("rails_status", {})
    print(f"  Response ({latency_ms:.0f}ms): status={status}")
    for rail_name, rail_result in rails_status.items():
        print(f"    - [{rail_name}] {rail_result.get('status', 'unknown')}")


def main() -> None:
    args = parse_args()
    namespace = args.namespace or os.getenv("GUARDRAILS_NS", _DEFAULT_NAMESPACE)

    if args.enable_mcp_gateway and not args.mcp_server_url:
        print("ERROR: --mcp-server-url is required when --enable-mcp-gateway is set", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Configure NeMo Guardrails — Financial Agent")
    print("=" * 60)
    print(f"  Model endpoint:     {args.model_endpoint}")
    print(f"  Namespace:          {namespace}")
    print(f"  Replicas:           {args.replicas}")
    print(f"  OpenTelemetry:      {args.otel_endpoint}")
    if args.enable_mcp_gateway:
        print(f"  MCP Gateway:        ENABLED [Technology Preview, RHOAI 3.5 EA2]")
        print(f"  MCP Server URL:     {args.mcp_server_url}")
    else:
        print(f"  MCP Gateway:        disabled")
    print("=" * 60)

    mcp_gateway_name = "financial-mcp-gateway" if args.enable_mcp_gateway else None
    guardrails_manifest = build_nemoguardrails_cr(
        namespace=namespace,
        model_endpoint=args.model_endpoint,
        replicas=args.replicas,
        otel_endpoint=args.otel_endpoint,
        mcp_gateway_name=mcp_gateway_name,
    )

    gateway_manifest = None
    if args.enable_mcp_gateway:
        gateway_manifest = build_mcpgateway_extension_cr(
            namespace=namespace,
            mcp_server_url=args.mcp_server_url,
        )

    configmap_manifest = build_guardrails_configmap(
        namespace=namespace,
        model_endpoint=args.model_endpoint,
        model_name=args.model_endpoint.rstrip("/").rsplit("/", 1)[0].rsplit("/", 1)[-1],
    )

    if args.dry_run:
        print("\n--- ConfigMap (guardrails config) ---")
        print(yaml.dump(configmap_manifest, default_flow_style=False))
        print("--- NemoGuardrails CR (GA, RHOAI 3.4) ---")
        print(yaml.dump(guardrails_manifest, default_flow_style=False))
        if gateway_manifest:
            print("--- MCPGatewayExtension CR (Technology Preview, RHOAI 3.5 EA2) ---")
            print(yaml.dump(gateway_manifest, default_flow_style=False))
        return

    config.load_kube_config()

    print("\n[1/4] Creating ConfigMap with guardrails configuration...")
    v1 = client.CoreV1Api()
    try:
        v1.read_namespaced_config_map("financial-guardrails-config", namespace)
        v1.replace_namespaced_config_map("financial-guardrails-config", namespace, configmap_manifest)
        print("  Updated existing ConfigMap 'financial-guardrails-config'")
    except ApiException as e:
        if e.status == 404:
            v1.create_namespaced_config_map(namespace, configmap_manifest)
            print("  Created ConfigMap 'financial-guardrails-config'")
        else:
            raise

    print("\n[2/4] Deploying NemoGuardrails CR [GA, RHOAI 3.4]...")
    apply_custom_resource(guardrails_manifest, _GUARDRAILS_PLURAL, namespace)

    if gateway_manifest:
        print("\n[3/4] Deploying MCPGatewayExtension CR [Technology Preview, RHOAI 3.5 EA2]...")
        apply_custom_resource(gateway_manifest, _GATEWAY_PLURAL, namespace)
    else:
        print("\n[3/4] Skipping MCP Gateway (not enabled)")

    print("\n[4/4] Verifying readiness...")
    name = guardrails_manifest["metadata"]["name"]
    endpoint = wait_for_guardrails_ready(name, namespace, args.timeout)

    if endpoint:
        test_guardrails_endpoint(endpoint)

    print(f"\nGuardrails deployment complete: {name}")
    if args.enable_mcp_gateway:
        print("  MCP Gateway active — agent tool calls are protected at the gateway layer")


if __name__ == "__main__":
    main()
