#!/usr/bin/env python
"""Summarize completed experiment runs from reports/runs.csv."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reid.reporting.summary import filter_run_rows, format_csv, format_markdown, load_run_rows, sort_run_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize readable experiment artifacts")
    parser.add_argument("--root", type=Path, default=Path("experiments"))
    parser.add_argument("--index", type=Path, default=None, help="Central run index; defaults to reports/runs.csv")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--workflow", choices=["probe", "finetune"], default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--matcher", default=None)
    parser.add_argument("--sort-by", default="run_utc")
    parser.add_argument("--ascending", action="store_true")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = args.index or (args.root.parent / "reports" / "runs.csv")
    rows = load_run_rows(index_path, args.root)
    rows = filter_run_rows(
        rows,
        dataset=args.dataset,
        workflow=args.workflow,
        method=args.method,
        matcher=args.matcher,
    )
    rows = sort_run_rows(rows, args.sort_by, descending=not args.ascending)
    rendered = format_csv(rows) if args.format == "csv" else format_markdown(rows)
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
