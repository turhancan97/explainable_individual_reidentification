#!/usr/bin/env python
"""Collect few-shot results (probe runs + few-shot views) into a CSV, a markdown summary
and accuracy-vs-training-images figures.

    python scripts/fewshot_results.py --animal CowDataset
    python scripts/fewshot_results.py --animal NyalaData --metric top_1 --metric mAP_at_k --candidate-k 50

``--k-sweep`` adds the accuracy-vs-candidate-budget figures of the full-gallery setting: one
per metric (top-1/5/10 and their balanced counterparts) and matcher, so RDD-LightGlue and LoMa
each get six. They need probe runs at several budgets (PROBE_CANDIDATE_K="10 50 100 250"), and
the balanced top-5/top-10 of older runs come from scripts/fewshot_backfill_balanced.py:

    python scripts/fewshot_results.py --animal CzechLynx --k-sweep
    python scripts/fewshot_results.py --animal CzechLynx --k-sweep \
        --fallback-checkpoint $FEWSHOT_ROOT/checkpoints/CzechLynx/legacy/frac1.0-seed0/loma-finetuned-gpu1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reid.reporting.fewshot import (  # noqa: E402
    DEFAULT_K_VALUES,
    DEFAULT_PLOT_METRICS,
    K_SWEEP_METRICS,
    PLOT_METRICS,
    collect,
)


def parse_args() -> argparse.Namespace:
    default_root = os.environ.get("FEWSHOT_ROOT") or (
        Path(os.environ.get("CZECHLYNX_DATA_ROOT", "/shared/sets/datasets/vision/czechlynx")) / "fewshot"
    )
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--animal", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protocol", default="legacy")
    parser.add_argument("--experiments", type=Path, default=Path("experiments"))
    parser.add_argument("--fewshot-root", type=Path, default=Path(default_root))
    parser.add_argument("--output-dir", type=Path, default=None, help="default: reports/fewshot/<animal>")
    parser.add_argument("--metric", action="append", choices=[*PLOT_METRICS, "all"],
                        help="metric to plot; repeat for several (default: top_1, top_5, balanced_top_1)")
    parser.add_argument("--candidate-k", type=int, default=50, help="candidate budget of the plotted probe runs")
    parser.add_argument("--k-sweep", action="store_true",
                        help="also write the accuracy-vs-k figures (full gallery, one per metric and matcher)")
    parser.add_argument("--k", type=int, action="append",
                        help=f"candidate budgets on the x axis of --k-sweep (default: {', '.join(map(str, DEFAULT_K_VALUES))}); "
                             "repeat, or pass none to plot every budget that has runs")
    parser.add_argument("--k-sweep-metric", action="append", choices=[*K_SWEEP_METRICS, "all"],
                        help=f"metric of the --k-sweep figures (default: all of {', '.join(K_SWEEP_METRICS)})")
    parser.add_argument("--fallback-checkpoint", action="append", default=[], metavar="DIR",
                        help="checkpoint directory of a stand-in training run (e.g. the single-GPU "
                             "loma-finetuned-gpu1) that fills in for the regular fine-tuned series "
                             "wherever the regular run has no probe yet; repeat for several. Drop it "
                             "once the regular checkpoint is probed and the curve uses that one again")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = tuple(args.metric or DEFAULT_PLOT_METRICS)
    if "all" in metrics:
        metrics = PLOT_METRICS
    sweep_metrics = tuple(args.k_sweep_metric or K_SWEEP_METRICS)
    if "all" in sweep_metrics:
        sweep_metrics = K_SWEEP_METRICS
    outputs = collect(
        experiment_root=args.experiments,
        fewshot_root=args.fewshot_root,
        animal=args.animal,
        protocol=args.protocol,
        seed=args.seed,
        output_dir=args.output_dir or Path("reports") / "fewshot" / args.animal,
        metrics=metrics,
        candidate_k=args.candidate_k,
        formats=args.formats,
        k_sweep=args.k_sweep,
        k_sweep_metrics=sweep_metrics,
        k_values=tuple(args.k) if args.k else DEFAULT_K_VALUES,
        fallback_checkpoints=tuple(args.fallback_checkpoint),
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
