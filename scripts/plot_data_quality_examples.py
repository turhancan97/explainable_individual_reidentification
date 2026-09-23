#!/usr/bin/env python
"""Plot confirmed low-quality dataset examples: raw photo above model input.

Every example below was flagged by ``scripts/audit_image_quality.py`` and then
confirmed by eye against the raw source file. The bottom row is the exact probe
input (pre-masked WildlifeReID-10k file, or the CzechLynx RLE mask applied through
``BenchmarkDatasetView``). Edit ``EXAMPLES`` only after confirming a new image.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
for entry in (ROOT_DIR, ROOT_DIR / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from audit_image_quality import _FrameAdapter, load_model_input  # noqa: E402
from reid.data.dataset_view import BenchmarkDatasetView  # noqa: E402
from reid.reporting.paper_datasets import PAPER_PROFILES  # noqa: E402


class Example(NamedTuple):
    group: str
    dataset: str
    row: int
    note: str
    # Ring the foreground centre; only for confirmed near-invisible specks.
    ring: bool = False


# Confirmed by eye on 2026-09-23 (raw file and model input both inspected).
EXAMPLES = [
    Example("(a) Washed-out frame", "lynx_closed", 38308, "no animal visible"),
    Example("(a) Washed-out frame", "leopard", 2824, "IR flash glare"),
    Example("(b) Mask on wrong object", "lynx_closed", 3914, "branches, not lynx"),
    Example("(b) Mask on wrong object", "lynx_closed", 36426, "stick, no lynx"),
    Example("(c) Corrupted frame", "lynx_closed", 2307, "colour banding"),
    Example("(d) Blurred frame", "hyena", 550, "no fine detail"),
]
DATASET_NAMES = {
    "lynx_closed": "CzechLynx",
    "leopard": "LeopardID2022",
    "hyena": "HyenaID2022",
    "sea_star": "SeaStarReID2023",
    "whale_shark": "WhaleSharkID",
}
THUMBNAIL = 480


def load_pair(profiles, frames, example: Example):
    profile = profiles[example.dataset]
    frame = frames.setdefault(example.dataset, pd.read_csv(profile.metadata, low_memory=False))
    view = None
    if profile.mask_col:
        view = BenchmarkDatasetView(_FrameAdapter(frame, profile.identity_col), label_col=profile.identity_col, no_background=True, mask_col=profile.mask_col)
    model_input, foreground = load_model_input(profile, frame, view, example.row)
    raw_path = str(frame.iloc[example.row]["path"])
    if profile.mask_col is None:
        # Pre-masked WildlifeReID-10k files mirror the raw tree under images/.
        raw_path = raw_path.replace("masked_images/", "images/", 1)
    raw = Image.open(profile.root / raw_path).convert("RGB")
    masked = Image.fromarray(model_input)
    # Opt-in: a dark animal in a pre-masked file falls below the foreground
    # threshold, so an automatic small-foreground rule would ring it wrongly.
    ys, xs = np.nonzero(foreground)
    centre = None
    if example.ring and len(xs):
        centre = _square_coords(masked.size, float(xs.mean()), float(ys.mean()))
    return _square(raw, (255, 255, 255)), _square(masked, (0, 0, 0)), centre


def _square_coords(size, x: float, y: float):
    width, height = size
    scale = THUMBNAIL / max(width, height)
    return (x * scale + (THUMBNAIL - width * scale) / 2, y * scale + (THUMBNAIL - height * scale) / 2)


def _square(image: Image.Image, fill) -> Image.Image:
    """Letterbox onto a square canvas so every panel has identical geometry."""
    image = image.copy()
    image.thumbnail((THUMBNAIL, THUMBNAIL))
    canvas = Image.new("RGB", (THUMBNAIL, THUMBNAIL), fill)
    canvas.paste(image, ((THUMBNAIL - image.width) // 2, (THUMBNAIL - image.height) // 2))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--width", type=float, default=6.875, help="Figure width in inches (CVPR full text width)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    profiles = {profile.key: profile for profile in PAPER_PROFILES}
    frames: dict = {}
    columns = len(EXAMPLES)
    cell = args.width / columns
    figure, axes = plt.subplots(2, columns, figsize=(args.width, 2 * cell + 0.5), squeeze=False)
    for column, example in enumerate(EXAMPLES):
        raw, masked, centre = load_pair(profiles, frames, example)
        for row, image in enumerate((raw, masked)):
            axis = axes[row][column]
            axis.imshow(image)
            if row == 1 and centre is not None:
                axis.add_patch(plt.Circle(centre, THUMBNAIL * 0.07, fill=False, edgecolor="#D55E00", linewidth=0.9))
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor("black")
            for spine in axis.spines.values():
                spine.set_linewidth(0.4)
        axes[0][column].set_title(DATASET_NAMES[example.dataset], fontsize=6.5, pad=2)
        axes[1][column].set_xlabel(example.note, fontsize=6.5, labelpad=2)
    axes[0][0].set_ylabel("Raw image", fontsize=7)
    axes[1][0].set_ylabel("Model input", fontsize=7)
    figure.subplots_adjust(left=0.035, right=0.995, top=0.86, bottom=0.08, wspace=0.06, hspace=0.06)

    # Group labels centred over each run of same-group columns.
    start = 0
    for index in range(1, columns + 1):
        if index == columns or EXAMPLES[index].group != EXAMPLES[start].group:
            left = axes[0][start].get_position().x0
            right = axes[0][index - 1].get_position().x1
            figure.text((left + right) / 2, 0.965, EXAMPLES[start].group, ha="center", va="center", fontsize=7, fontweight="bold")
            figure.add_artist(plt.Line2D([left + 0.004, right - 0.004], [0.935, 0.935], color="0.35", linewidth=0.6))
            start = index

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        output = args.output_dir / f"data_quality_examples.{fmt}"
        figure.savefig(output, dpi=600 if fmt == "png" else 300)
        print(f"[data-quality] {output}")


if __name__ == "__main__":
    main()
