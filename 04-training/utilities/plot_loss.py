"""Visualize training loss curves from Training Hub checkpoints.

Uses ``training_hub.plot_loss()`` to read metrics from checkpoint
directories and create loss-curve plots.  Supports SFT, OSFT, and LoRA
training runs, with optional EMA smoothing and multi-experiment
comparison.

Adapted from the Training Hub plot_loss_example notebook.

Requirements:
    pip install training-hub matplotlib

Usage:
    python plot_loss.py /path/to/checkpoints
    python plot_loss.py /path/to/run1 /path/to/run2 --labels "lr=1e-5" "lr=5e-6"
    python plot_loss.py /path/to/checkpoints --ema --ema-span 50
    python plot_loss.py /path/to/checkpoints -o ./reports/loss.png
"""

from __future__ import annotations

import argparse
import sys

from training_hub import plot_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training loss curves from Training Hub checkpoint directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ckpt_dirs",
        nargs="+",
        help="One or more checkpoint directories to plot",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Legend labels (one per checkpoint dir; auto-generated if omitted)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path (default: loss_plot.png in the first ckpt dir)",
    )
    parser.add_argument(
        "--metrics-file",
        default=None,
        help="Metrics filename to look for (auto-detected if omitted)",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Overlay an EMA-smoothed curve",
    )
    parser.add_argument(
        "--ema-span",
        type=int,
        default=30,
        help="EMA smoothing window (higher = smoother)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dirs = args.ckpt_dirs if len(args.ckpt_dirs) > 1 else args.ckpt_dirs[0]

    print("=" * 60)
    print("Plot Training Loss")
    print("=" * 60)
    if isinstance(dirs, list):
        for i, d in enumerate(dirs):
            label = args.labels[i] if args.labels and i < len(args.labels) else d
            print(f"  [{i + 1}] {label}: {d}")
    else:
        print(f"  Checkpoint dir: {dirs}")
    print("=" * 60)

    try:
        plot_path = plot_loss(
            dirs,
            metrics_file=args.metrics_file,
            output_path=args.output,
            labels=args.labels,
            ema=args.ema,
            ema_span=args.ema_span,
            show=args.show,
        )
    except Exception as exc:
        print(f"\nPlotting failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nPlot saved to: {plot_path}")


if __name__ == "__main__":
    main()
