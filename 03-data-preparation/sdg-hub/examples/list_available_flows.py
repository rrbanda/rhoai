"""List all built-in SDG Hub flows and their metadata.

This utility queries the FlowRegistry to discover every flow shipped with
SDG Hub, then prints a summary of each flow including its ID and name.

Usage:
    python list_available_flows.py
"""

from sdg_hub import FlowRegistry


def main() -> None:
    FlowRegistry.discover_flows()
    flows = FlowRegistry.list_flows()

    print(f"Found {len(flows)} registered flows:\n")

    for flow_info in flows:
        print(f"  {flow_info['id']}: {flow_info['name']}")

    print(f"\nTotal: {len(flows)} flows.")


if __name__ == "__main__":
    main()
