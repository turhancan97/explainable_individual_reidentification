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
    PLOT_METHODS,
    PLOT_STYLES,
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
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_PLOT_BUDGETS),
        help="Candidate budgets; unseen_eval_split defaults to 10, 50, 100, 160.",
    )
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf"])
    parser.add_argument(
        "--exclude-method",
        dest="exclude_methods",
        action="append",
        choices=PLOT_METHODS,
        default=[],
        help="Exclude a method family and both checkpoint variants; repeat for multiple methods.",
    )
    parser.add_argument(
        "--style",
        choices=PLOT_STYLES,
        default="diagnostic",
        help="Visual style: paper (default), presentation, or diagnostic.",
    )
    parser.add_argument(
        "--x-scale",
        choices=["log", "categorical"],
        default=None,
        help="Override the style's x-axis scale. Paper defaults to a log-scaled candidate budget.",
    )
    y_group = parser.add_mutually_exclusive_group()
    y_group.add_argument(
        "--shared-y",
        dest="shared_y",
        action="store_true",
        help="Use one y-axis range across all panels.",
    )
    y_group.add_argument(
        "--independent-y",
        dest="shared_y",
        action="store_false",
        help="Use an independently scaled y-axis for each panel.",
    )
    parser.set_defaults(shared_y=None)
    parser.add_argument(
        "--label-endpoints",
        action="store_true",
        help="Annotate the last available point of each series with its label.",
    )
    parser.add_argument(
        "--descriptor-family",
        action="append",
        choices=["rdd", "loma"],
        help="Generate separate descriptor fine-tuning figures for this matcher family; repeat or omit for all available families.",
    )
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
        descriptor_families=args.descriptor_family,
        style=args.style,
        x_scale=args.x_scale,
        shared_y=args.shared_y,
        label_endpoints=args.label_endpoints,
        exclude_methods=args.exclude_methods,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

# python scripts/plot_paper_figures.py --metric balanced_top_1

# python scripts/plot_paper_figures.py \
#   --animal CzechLynx \
#   --animal NyalaData \
#   --animal BelugaID \
#   --animal HyenaID2022 \
#   --animal LeopardID2022 \
#   --animal SeaStarReID2023 \
#   --animal WhaleSharkID \
#   --animal ZindiTurtleRecall \
#   --metric all \
#   --formats png pdf \
#   --output-dir reports/figures/cvpr_scope
