"""Dataset/split profiles reported in the paper.

Mirrors the DATASET_PROFILES rows in probe-parallel-wildlife.sh and
probe-parallel-czechlynx.sh. Keep both in sync when a paper split changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional

WILDLIFE_ROOT = Path("/shared/sets/datasets/vision/czechlynx/WildlifeReID-10k")
CZECHLYNX_ROOT = Path("/shared/sets/datasets/vision/czechlynx/CzechLynx_v2")
CZECHLYNX_METADATA = CZECHLYNX_ROOT / "CzechLynxDataset-Metadata-Real.csv"


class PaperProfile(NamedTuple):
    key: str
    label: str
    source: str
    metadata: Path
    identity_col: str
    split_col: str
    database_value: str
    query_value: str
    dataset_name: str
    animal: str
    root: Path
    # RLE mask column applied at load time (dataset.no_background=true); None
    # when the metadata already points at pre-masked image files.
    mask_col: Optional[str]


def _wildlife(key: str, label: str, animal: str, metadata_dir: str) -> PaperProfile:
    return PaperProfile(
        key, label, animal, WILDLIFE_ROOT / metadata_dir / f"metadata_{animal}.csv",
        "identity", "split", "train", "test", "WildlifeReID-10k", animal, WILDLIFE_ROOT, None,
    )


def _czechlynx(key: str, label: str, split_col: str) -> PaperProfile:
    return PaperProfile(
        key, label, "CzechLynx v2", CZECHLYNX_METADATA, "unique_name", split_col, "train", "test",
        "CzechLynx_v2", "CzechLynx", CZECHLYNX_ROOT, "mask",
    )


PAPER_PROFILES = [
    _wildlife("nyala", "Nyala", "NyalaData", "metadata_no_background"),
    _wildlife("beluga", "Beluga", "BelugaID", "metadata_no_background"),
    _wildlife("hyena", "Hyena", "HyenaID2022", "metadata_mdsplit_no_background"),
    _wildlife("leopard", "Leopard", "LeopardID2022", "metadata_mdsplit_no_background"),
    _wildlife("sea_star", "Sea Star", "SeaStarReID2023", "metadata_mdsplit_no_background"),
    _wildlife("whale_shark", "Whale Shark", "WhaleSharkID", "metadata_no_background"),
    _wildlife("turtle", "Turtle", "ZindiTurtleRecall", "metadata_no_background"),
    _czechlynx("lynx_closed", "Lynx (closed)", "split-time_closed"),
    _czechlynx("lynx_open", "Lynx (open)", "split-time_open"),
]
