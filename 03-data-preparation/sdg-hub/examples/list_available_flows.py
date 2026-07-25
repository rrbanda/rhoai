"""List all built-in SDG Hub flows and their metadata.

This utility queries the FlowRegistry to discover every flow shipped with
SDG Hub, then prints a summary of each flow including its name, description,
and the blocks it contains.

Usage:
    python list_available_flows.py
    python list_available_flows.py --verbose
"""

import argparse

from sdg_hub import FlowRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List available SDG Hub flows."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show block-level details for each flow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    registry = FlowRegistry()
    flows = registry.list_flows()

    print(f"Found {len(flows)} registered flows:\n")

    for flow_info in flows:
        print(f"  {flow_info.name}: {flow_info.description}")

        if args.verbose and hasattr(flow_info, "blocks"):
            for block in flow_info.blocks:
                print(f"    - {block}")
            print()

    if not args.verbose:
        print("\nRe-run with --verbose to see block details.")


if __name__ == "__main__":
    main()
