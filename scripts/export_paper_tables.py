#!/usr/bin/env python
"""Export per-animal experiment results as LaTeX and CSV paper tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reid.reporting.paper_tables import DEFAULT_ABLATION_BUDGETS, export_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments"), help="Experiment artifact root")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/paper_tables"))
    parser.add_argument("--animal", action="append", help="Animal to export; repeat for multiple animals")
    parser.add_argument(
        "--split-protocol",
        action="append",
        dest="split_protocols",
        help="Split protocol to export; repeat for multiple protocols",
    )
    parser.add_argument("--main-candidate-k", type=int, default=50)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_ABLATION_BUDGETS))
    parser.add_argument(
        "--detailed-comments",
        action="store_true",
        help="Include generation timestamp, run IDs, and manifest paths in LaTeX comments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = export_tables(
        args.root,
        args.output_dir,
        animals=args.animal,
        split_protocols=args.split_protocols,
        main_candidate_k=args.main_candidate_k,
        budgets=args.budgets,
        detailed_comments=args.detailed_comments,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

# python scripts/export_paper_tables.py \
#   --animal CzechLynx \
#   --animal NyalaData \
#   --animal BelugaID \
#   --animal HyenaID2022 \
#   --animal LeopardID2022 \
#   --animal SeaStarReID2023 \
#   --animal WhaleSharkID \
#   --animal ZindiTurtleRecall \
#   --output-dir reports/paper_tables/cvpr_scope