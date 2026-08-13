"""Dataset identity fields shared by feature-cache implementations."""

from pathlib import Path
from typing import Any, Dict


SUPPORTED_IMAGE_VARIANTS = ("background", "no_background")


def validate_image_variant(value: Any) -> str:
    """Return a supported image variant or raise a clear configuration error."""

    variant = str(value).strip().lower()
    if variant not in SUPPORTED_IMAGE_VARIANTS:
        supported = ", ".join(SUPPORTED_IMAGE_VARIANTS)
        raise ValueError(f"Unsupported dataset.image_variant '{variant}'. Supported variants: {supported}")
    return variant


def build_dataset_cache_identity(dataset_cfg: Any) -> Dict[str, str]:
    """Build stable provenance fields that must distinguish extracted image content."""

    root = Path(str(dataset_cfg.root)).expanduser().resolve()
    metadata_file = Path(str(dataset_cfg.metadata_file)).as_posix()
    return {
        "dataset_root": root.as_posix(),
        "metadata_file": metadata_file,
        "image_variant": validate_image_variant(dataset_cfg.image_variant),
    }
