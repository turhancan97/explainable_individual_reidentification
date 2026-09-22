#!/usr/bin/env python
"""Print the launcher variants (method|matcher|label|path, ';'-separated) that have no
completed probe run yet for an animal / split column / candidate budget.

    python scripts/fewshot_probe_missing.py --animal CowDataset --split-col split_frac0.5_seed0 \
        --candidate-k 50 --variants "cosine|-|default|-;vismatch|rdd-lightglue|custom|/path/model.safetensors"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reid.reporting.fewshot import missing_variants  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--animal", required=True)
    parser.add_argument("--split-col", required=True)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--experiments", type=Path, default=Path("experiments"))
    args = parser.parse_args()
    variants = [v for v in args.variants.split(";") if v]
    print(";".join(missing_variants(args.experiments, args.animal, args.split_col, args.candidate_k, variants)))


if __name__ == "__main__":
    main()
