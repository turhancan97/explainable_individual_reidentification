#!/usr/bin/env python
"""Plot per-animal accuracy versus candidate budget from experiment artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reid.reporting.plot_figures import (  # noqa: E402
    DEFAULT_PLOT_BUDGETS,
    DEFAULT_PLOT_METRICS,
    PLOT_METRICS,
    plot_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--animal", action="append", help="Animal to plot; repeat for multiple animals")
    parser.add_argument(
        "--split-protocol",
        action="append",
        help="Split protocol to plot; repeat for multiple splits (default: all discovered splits)",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=[*PLOT_METRICS, "all"],
        help="Metric to plot; repeat for multiple metrics (default: top_1, top_5, top_10)",
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_PLOT_BUDGETS))
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = plot_metrics(
        args.root,
        args.output_dir,
        animals=args.animal,
        split_protocols=args.split_protocol,
        metrics=tuple(args.metric or DEFAULT_PLOT_METRICS),
        budgets=args.budgets,
        formats=args.formats,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

# python scripts/plot_paper_figures.py --metric balanced_top_1
