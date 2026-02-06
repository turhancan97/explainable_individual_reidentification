import json
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset


class BenchmarkDatasetView(Dataset):
    def __init__(
        self,
        base_dataset: Any,
        label_col: str,
        transform: Optional[Any] = None,
        no_background: bool = False,
        mask_col: str = "mask",
    ):
        self.base_dataset = base_dataset
        self.transform = transform
        self.no_background = no_background
        self.mask_col = mask_col
        self.col_label = getattr(base_dataset, "col_label", label_col)
        self.df = getattr(base_dataset, "df", getattr(base_dataset, "metadata", None))
        self.metadata = getattr(base_dataset, "metadata", self.df)
        if self.df is None:
            raise ValueError("Dataset view requires base dataset to expose df or metadata")

        if self.no_background and self.mask_col not in self.df.columns:
            raise KeyError(f"Mask column '{self.mask_col}' is missing while dataset.no_background=true")

        if self.col_label in self.df.columns:
            self.labels_string = self.df[self.col_label].astype(str).to_numpy()
        else:
            self.labels_string = np.array([], dtype=str)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _decode_mask(self, row: pd.Series, idx: int) -> np.ndarray:
        raw_mask = row.get(self.mask_col)
        if pd.isna(raw_mask):
            raise ValueError(f"Missing mask at row index {idx}")

        if isinstance(raw_mask, str):
            try:
                mask_data = json.loads(raw_mask)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON mask at row index {idx}") from exc
        elif isinstance(raw_mask, dict):
            mask_data = raw_mask
        else:
            raise ValueError(
                f"Unsupported mask type at row index {idx}: {type(raw_mask)}. "
                "Expected JSON string or COCO-RLE dict."
            )

        try:
            mask = mask_utils.decode(mask_data).astype(np.uint8)
        except Exception as exc:
            raise ValueError(f"Failed to decode mask at row index {idx}") from exc

        if mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = mask[..., 0]
            else:
                mask = mask.max(axis=-1)
        if mask.ndim != 2:
            raise ValueError(f"Decoded mask must be 2D at row index {idx}, got shape={mask.shape}")
        return mask

    def _to_hwc_uint8(self, image: Any, idx: int) -> np.ndarray:
        if isinstance(image, Image.Image):
            arr = np.array(image)
        elif torch.is_tensor(image):
            arr = image.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
        else:
            arr = np.asarray(image)

        if arr.ndim not in (2, 3):
            raise ValueError(f"Unsupported image shape at row index {idx}: {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _apply_no_background(self, image: Any, idx: int) -> Image.Image:
        row = self.df.iloc[idx]
        mask = self._decode_mask(row, idx)
        image_np = self._to_hwc_uint8(image, idx)

        if image_np.shape[0] != mask.shape[0] or image_np.shape[1] != mask.shape[1]:
            raise ValueError(
                f"Mask/Image size mismatch at row index {idx}: mask={mask.shape}, image={image_np.shape[:2]}"
            )

        if image_np.ndim == 2:
            image_np = image_np * mask
        else:
            image_np = image_np * np.expand_dims(np.asfortranarray(mask), axis=-1)
        return Image.fromarray(image_np)

    def __getitem__(self, idx: int):
        item = self.base_dataset[idx]
        if isinstance(item, tuple):
            image = item[0]
            tail = item[1:]
        elif isinstance(item, list):
            image = item[0]
            tail = tuple(item[1:])
        else:
            image = item
            tail = tuple()

        if self.no_background:
            image = self._apply_no_background(image, idx)
        if self.transform is not None:
            image = self.transform(image)

        if isinstance(item, tuple):
            return (image, *tail)
        if isinstance(item, list):
            return [image, *tail]
        return image
