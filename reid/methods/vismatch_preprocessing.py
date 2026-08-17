"""Canonical image preprocessing for Vismatch feature-level evaluation."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def to_rgb_float_tensor(image: Any) -> torch.Tensor:
    """Convert an RGB image to a CHW float32 tensor in ``[0, 1]``.

    The conversion follows the value semantics of ``torchvision.transforms.ToTensor``
    without resizing.  Resizing is kept separate so every Vismatch matcher uses the
    same tensor interpolation path.
    """

    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    elif torch.is_tensor(image):
        array = image.detach().cpu().numpy()
        if array.ndim == 4:
            if array.shape[0] != 1:
                raise ValueError(f"Expected one image, got tensor shape {array.shape}")
            array = array[0]
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
    else:
        array = np.asarray(image)

    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"Expected an image with 2 or 3 dimensions, got shape {array.shape}")

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] >= 4:
        array = array[..., :3]
    elif array.shape[-1] != 3:
        raise ValueError(f"Expected one, three, or four channels, got shape {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32, copy=False)
        if array.size and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
            array = array / 255.0
    else:
        array = array.astype(np.float32) / 255.0

    array = np.clip(array, 0.0, 1.0)
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).contiguous()


def resize_long_side_divisible(
    image: torch.Tensor,
    resize_max: int,
    divisible_by: int = 32,
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """Resize to a target long side and floor dimensions to a divisor.

    Returns the processed CHW tensor, the source ``(H, W)``, and processed ``(H, W)``.
    The target long side is used even for smaller source images, and each dimension
    is floored to a multiple of ``divisible_by`` before interpolation. LoMa uses
    ``divisible_by=14`` because its DINOv2-L/14 descriptor requires 14-pixel patch
    divisibility; the other Vismatch profiles use 32.
    """

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected a CHW RGB tensor, got shape {tuple(image.shape)}")
    resize_max = int(resize_max)
    if resize_max <= 0:
        raise ValueError("Vismatch resize_max must be a positive target resolution")
    divisible_by = int(divisible_by)
    if divisible_by <= 0:
        raise ValueError("Vismatch spatial divisor must be positive")

    height, width = (int(value) for value in image.shape[-2:])
    scale = float(resize_max) / float(max(height, width))
    processed_height = max(divisible_by, int(height * scale) // divisible_by * divisible_by)
    processed_width = max(divisible_by, int(width * scale) // divisible_by * divisible_by)
    if processed_height <= 0 or processed_width <= 0:
        raise ValueError(
            f"Vismatch resize produced a zero-sized /{divisible_by} image; source dimensions must be positive"
        )

    processed = F.interpolate(
        image.float().unsqueeze(0),
        size=(processed_height, processed_width),
        mode="bilinear",
        align_corners=False,
    )[0]
    return processed, (height, width), (processed_height, processed_width)


def resize_long_side_div32(
    image: torch.Tensor,
    resize_max: int,
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """Apply the standard RDD/LightGlue Vismatch `/32` resize protocol."""

    return resize_long_side_divisible(image, resize_max, divisible_by=32)


def preprocess_vismatch_image(
    image: Any,
    resize_max: int,
    divisible_by: int = 32,
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """Convert and resize one image for a Vismatch matcher profile."""

    return resize_long_side_divisible(to_rgb_float_tensor(image), resize_max, divisible_by=divisible_by)


def to_uint8_vismatch_image(image: Any, resize_max: int, divisible_by: int = 32) -> np.ndarray:
    """Return a visualization image using the same Vismatch preprocessing."""

    processed, _source_size, _processed_size = preprocess_vismatch_image(image, resize_max, divisible_by=divisible_by)
    return (
        processed.clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .byte()
        .cpu()
        .numpy()
    )
