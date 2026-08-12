"""Vismatch-backed local matcher benchmark."""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

from reid.methods.vismatch_profiles import (
    FEATURE_SCHEMA_VERSION,
    SUPPORTED_VISMATCH_MATCHERS,
    VISMATCH_COMMIT,
    FrameFeatures,
    MatcherProfile,
    build_matcher_profile,
    default_matcher_threshold,
    normalize_match_confidences,
    validate_matcher_name,
)
from reid.utils.io import ensure_file


# Compatibility name for internal callers and saved feature semantics.
FrameFeat = FrameFeatures


@dataclass
class MatchResult:
    score: float
    match_count: int
    confidence_sum: float
    confidence_mean: float
    matched_kpts0: np.ndarray | None = None
    matched_kpts1: np.ndarray | None = None
    diagnostics: Dict[str, Any] | None = None
    confidences: np.ndarray | None = None


def _choose_vismatch_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Vismatch device is set to cuda but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _as_numpy(value[0])
        return np.asarray([_as_numpy(item) for item in value])
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _drop_batch(value: Any) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim >= 3 and array.shape[0] == 1:
        return array[0]
    return array


class VismatchMatcherBackend:
    """Feature-level Vismatch adapter with explicit pairwise diagnostics."""

    def __init__(self, matcher: str, device: torch.device, top_k: int, threshold: float, feature_matching_mode: str = "feature_level") -> None:
        matcher = validate_matcher_name(matcher)
        try:
            from vismatch import get_matcher
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Vismatch is required for the vismatch method. Install the pinned dependency "
                f"at commit {VISMATCH_COMMIT}."
            ) from exc

        self.matcher_name = matcher
        self.device = device
        self.top_k = int(top_k)
        self.threshold = float(threshold)
        self.feature_matching_mode = str(feature_matching_mode)
        if self.feature_matching_mode == "pairwise":
            raise ValueError("pairwise Vismatch mode is diagnostics-only; use match_images() explicitly")
        self.profile = build_matcher_profile(matcher, top_k, threshold, self.feature_matching_mode)
        self.last_diagnostics: Dict[str, Any] = {}
        self.model = get_matcher(matcher, device=str(device), max_num_keypoints=self.top_k)

        if matcher == "rdd-lightglue":
            required = ("matcher", "lightglue")
            missing = [name for name in required if not hasattr(self.model, name)]
            if missing or not hasattr(self.model.matcher, "RDD"):
                raise RuntimeError(
                    "Pinned Vismatch RDD-LightGlue matcher does not expose the required native "
                    f"components: {', '.join(missing or ['matcher.RDD'])}."
                )
            self.extractor = self.model.matcher.RDD
            self.pair_matcher = self.model.lightglue
            self.extractor.top_k = self.top_k
            self.extractor.set_softdetect(top_k=self.top_k)
        elif matcher == "loma":
            if not hasattr(self.model, "matcher") or not hasattr(self.model, "preprocess"):
                raise RuntimeError(
                    "Pinned Vismatch LoMa matcher does not expose matcher/preprocess "
                    "components required for feature-level matching."
                )
            try:
                from vismatch.im_models.loma import filter_matches
            except (ImportError, AttributeError) as exc:
                raise RuntimeError(
                    "Pinned Vismatch LoMa adapter does not expose its mutual-match filter."
                ) from exc
            self.extractor = self.model.matcher
            self.pair_matcher = self.model.matcher
            self._loma_filter_matches = filter_matches
            self._loma_threshold = self.threshold
        else:
            if not hasattr(self.model, "extractor") or not hasattr(self.model, "matcher"):
                raise RuntimeError(
                    f"Pinned Vismatch matcher '{matcher}' does not expose feature-level "
                    "extractor and matcher components."
                )
            self.extractor = self.model.extractor
            self.pair_matcher = self.model.matcher

    @staticmethod
    def _prepare_rdd(image: torch.Tensor) -> tuple[torch.Tensor, int, int, float, float]:
        tensor = image[0] if image.ndim == 4 else image
        height, width = (int(value) for value in tensor.shape[-2:])
        processed_height = max(32, (height // 32) * 32)
        processed_width = max(32, (width // 32) * 32)
        prepared = F.interpolate(
            tensor.unsqueeze(0),
            size=(processed_height, processed_width),
            mode="bilinear",
            align_corners=False,
        )
        return prepared, height, width, height / processed_height, width / processed_width

    @staticmethod
    def _prepare_loma(image: torch.Tensor) -> tuple[torch.Tensor, int, int, int, int]:
        tensor = image[0] if image.ndim == 4 else image
        height, width = (int(value) for value in tensor.shape[-2:])
        pad_height = (-height) % 14
        pad_width = (-width) % 14
        prepared = F.pad(tensor, (0, pad_width, 0, pad_height)).unsqueeze(0)
        return prepared, height, width, height + pad_height, width + pad_width

    def extract_frame(self, image: torch.Tensor) -> FrameFeatures:
        image = image.to(self.device)
        if self.matcher_name == "rdd-lightglue":
            prepared, height, width, height_scale, width_scale = self._prepare_rdd(image)
            output = self.extractor.extract(prepared)[0]
            keypoints = _drop_batch(output["keypoints"]).astype(np.float32, copy=False)
            keypoints *= np.asarray([width_scale, height_scale], dtype=np.float32)
            descriptors = _drop_batch(output["descriptors"]).astype(np.float32, copy=False)
            raw_scores = output.get("scores", np.ones(keypoints.shape[0], dtype=np.float32))
            scores = _drop_batch(raw_scores).astype(np.float32, copy=False)
            image_size = np.asarray([height, width], dtype=np.int32)
            coordinate_convention = "pixel"
            original_image_size = image_size.copy()
        elif self.matcher_name == "loma":
            prepared, height, width, padded_height, padded_width = self._prepare_loma(image)
            output = self.extractor.detect_and_describe(prepared, self.top_k)
            keypoints = _drop_batch(output[0]).astype(np.float32, copy=False)
            descriptors = _drop_batch(output[1]).astype(np.float32, copy=False)
            scores = np.ones(keypoints.shape[0], dtype=np.float32)
            image_size = np.asarray([padded_height, padded_width], dtype=np.int32)
            coordinate_convention = "normalized[-1,1]"
            original_image_size = np.asarray([height, width], dtype=np.int32)
        else:
            tensor = image[0] if image.ndim == 4 else image
            output = self.extractor.extract(tensor.unsqueeze(0))
            keypoints = _drop_batch(output["keypoints"]).astype(np.float32, copy=False)
            descriptors = _drop_batch(output["descriptors"]).astype(np.float32, copy=False)
            raw_scores = output.get("scores", output.get("keypoint_scores", np.ones(keypoints.shape[0])))
            scores = _drop_batch(raw_scores).astype(np.float32, copy=False)
            image_size = np.asarray(tensor.shape[-2:], dtype=np.int32)
            coordinate_convention = "pixel"
            original_image_size = image_size.copy()
        return FrameFeatures(
            keypoints=keypoints,
            descriptors=descriptors,
            scores=scores,
            image_size=image_size,
            schema_version=FEATURE_SCHEMA_VERSION,
            coordinate_convention=coordinate_convention,
            image_size_convention="hw",
            original_image_size=original_image_size,
        )

    def _matcher_inputs(self, left: FrameFeatures, right: FrameFeatures) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "image0": {
                "keypoints": torch.from_numpy(left.keypoints).to(self.device).unsqueeze(0),
                "descriptors": torch.from_numpy(left.descriptors).to(self.device).unsqueeze(0),
                "image_size": torch.tensor(left.image_size[::-1].copy(), device=self.device).unsqueeze(0),
            },
            "image1": {
                "keypoints": torch.from_numpy(right.keypoints).to(self.device).unsqueeze(0),
                "descriptors": torch.from_numpy(right.descriptors).to(self.device).unsqueeze(0),
                "image_size": torch.tensor(right.image_size[::-1].copy(), device=self.device).unsqueeze(0),
            },
        }

    @torch.no_grad()
    def _match_loma_features(self, left: FrameFeatures, right: FrameFeatures) -> MatchResult:
        k0 = torch.from_numpy(left.keypoints).to(self.device).unsqueeze(0)
        k1 = torch.from_numpy(right.keypoints).to(self.device).unsqueeze(0)
        d0 = torch.from_numpy(left.descriptors).to(self.device).unsqueeze(0)
        d1 = torch.from_numpy(right.descriptors).to(self.device).unsqueeze(0)
        prediction = self.pair_matcher(k0, k1, d0, d1)
        m0, _, match_scores0, _ = self._loma_filter_matches(
            prediction["scores"], self._loma_threshold
        )
        valid = m0[0] >= 0
        matched_indices0 = torch.where(valid)[0]
        matched_indices1 = m0[0][valid]
        confidences = match_scores0[0][valid].float()
        confidence_sum = float(confidences.sum().item())
        denominator = min(max(1, len(left.keypoints)), max(1, len(right.keypoints)))
        matched0 = left.keypoints[matched_indices0.detach().cpu().numpy()] if valid.any() else np.empty((0, 2), dtype=np.float32)
        matched1 = right.keypoints[matched_indices1.detach().cpu().numpy()] if valid.any() else np.empty((0, 2), dtype=np.float32)
        self.last_diagnostics = {
            "match_count": int(confidences.numel()),
            "confidence_sum": confidence_sum,
            "confidence_mean": float(confidences.mean().item()) if confidences.numel() else 0.0,
            "inlier_count": None,
            "homography_available": None,
        }
        return MatchResult(
            score=confidence_sum / float(denominator),
            match_count=int(confidences.numel()),
            confidence_sum=confidence_sum,
            confidence_mean=self.last_diagnostics["confidence_mean"],
            matched_kpts0=matched0,
            matched_kpts1=matched1,
            diagnostics=dict(self.last_diagnostics),
            confidences=confidences.detach().cpu().numpy().astype(np.float32, copy=False),
        )

    @torch.no_grad()
    def match_features(self, left: FrameFeatures, right: FrameFeatures) -> MatchResult:
        if left.schema_version != FEATURE_SCHEMA_VERSION or right.schema_version != FEATURE_SCHEMA_VERSION:
            raise RuntimeError("Cached Vismatch features use an unsupported schema version")
        if self.matcher_name == "loma":
            return self._match_loma_features(left, right)
        prediction = self.pair_matcher(self._matcher_inputs(left, right))
        confidences = _as_numpy(prediction.get("scores", np.empty((1, 0)))).reshape(-1).astype(np.float64)
        matches = prediction.get("matches")
        if matches is not None:
            matches_np = _drop_batch(matches).astype(np.int64, copy=False)
            if matches_np.ndim == 2 and matches_np.shape[1] == 2 and len(matches_np) == len(confidences):
                valid = matches_np[:, 0] >= 0
                matches_np = matches_np[valid]
                confidences = confidences[valid]
            else:
                matches_np = np.empty((0, 2), dtype=np.int64)
        else:
            matches_np = np.empty((0, 2), dtype=np.int64)
        score, match_count, filtered = normalize_match_confidences(
            confidences, len(left.keypoints), len(right.keypoints), self.threshold
        )
        filtered_confidences = np.asarray(filtered, dtype=np.float64)
        if len(matches_np) == len(confidences):
            keep = confidences >= self.threshold
            matches_np = matches_np[keep]
        confidences = filtered_confidences
        confidence_sum = float(confidences.sum())
        matched0 = left.keypoints[matches_np[:, 0]] if len(matches_np) else np.empty((0, 2), dtype=np.float32)
        matched1 = right.keypoints[matches_np[:, 1]] if len(matches_np) else np.empty((0, 2), dtype=np.float32)
        self.last_diagnostics = {
            "match_count": int(len(confidences)),
            "confidence_sum": confidence_sum,
            "confidence_mean": float(confidences.mean()) if len(confidences) else 0.0,
            "inlier_count": None,
            "homography_available": None,
        }
        return MatchResult(score, int(match_count), confidence_sum, self.last_diagnostics["confidence_mean"], matched0, matched1, dict(self.last_diagnostics), confidences.astype(np.float32, copy=False))

    @torch.no_grad()
    def match_images(self, image0: Any, image1: Any) -> MatchResult:
        result = self.model(image0, image1)
        confidences = _as_numpy(result.get("matched_confidences", np.empty(0))).reshape(-1).astype(np.float64)
        if self.threshold > 0:
            confidences = confidences[confidences >= self.threshold]
        norm = min(max(1, len(_as_numpy(result.get("all_kpts0", [])))), max(1, len(_as_numpy(result.get("all_kpts1", [])))))
        confidence_sum = float(confidences.sum())
        self.last_diagnostics = {
            "match_count": int(len(confidences)),
            "confidence_sum": confidence_sum,
            "confidence_mean": float(confidences.mean()) if len(confidences) else 0.0,
            "inlier_count": int(result.get("num_inliers", 0)),
            "homography_available": result.get("H") is not None,
        }
        return MatchResult(confidence_sum / float(norm), int(len(confidences)), confidence_sum, self.last_diagnostics["confidence_mean"], diagnostics=dict(self.last_diagnostics), confidences=confidences.astype(np.float32, copy=False))

    def score_features(self, left: FrameFeatures, right: FrameFeatures) -> Tuple[float, int]:
        result = self.match_features(left, right)
        return result.score, result.match_count

def _to_image_tensor(image: Any, resize_max: int) -> torch.Tensor:
    if isinstance(image, Image.Image):
        img = image.convert("RGB")
    elif torch.is_tensor(image):
        arr = image.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
    else:
        arr = np.asarray(image)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")

    if resize_max and resize_max > 0:
        w, h = img.size
        scale = float(resize_max) / float(max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _to_uint8_rgb(
    image: Any,
    resize_max: int,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
) -> np.ndarray:
    if isinstance(image, Image.Image):
        img = image.convert("RGB")
    elif torch.is_tensor(image):
        arr = image.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if np.issubdtype(arr.dtype, np.floating):
            if arr.min() >= 0.0 and arr.max() <= 1.0:
                arr = arr * 255.0
            elif mean is not None and std is not None and arr.ndim == 3 and arr.shape[2] >= 3:
                mean_np = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
                std_np = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
                arr = arr[..., :3] * std_np + mean_np
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            elif arr.max() <= 5.0 and arr.min() >= -5.0:
                # Fallback for normalized tensors when explicit stats are unavailable.
                arr_min = float(arr.min())
                arr_max = float(arr.max())
                if arr_max > arr_min:
                    arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
                else:
                    arr = np.zeros_like(arr)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
    else:
        arr = np.asarray(image)
        if np.issubdtype(arr.dtype, np.floating) and arr.min() >= 0.0 and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")

    if resize_max and resize_max > 0:
        w, h = img.size
        scale = float(resize_max) / float(max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return np.asarray(img)


def _cache_key(
    image_path: str,
    split_name: str,
    resize_max: int,
    top_k: int,
    cfg_tag: str,
) -> str:
    payload = f"{split_name}|{image_path}|resize_max={resize_max}|top_k={top_k}|{cfg_tag}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.npz"


def _save_cached_feat(path: Path, feat: FrameFeat) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keypoints=feat.keypoints,
        descriptors=feat.descriptors,
        scores=feat.scores,
        image_size=feat.image_size,
        original_image_size=(feat.original_image_size if feat.original_image_size is not None else feat.image_size),
        schema_version=np.asarray(feat.schema_version, dtype=np.int64),
        coordinate_convention=np.asarray(feat.coordinate_convention, dtype="U16"),
        image_size_convention=np.asarray(feat.image_size_convention, dtype="U8"),
    )


def _load_cached_feat(path: Path) -> FrameFeat:
    data = np.load(path)
    return FrameFeat(
        keypoints=data["keypoints"],
        descriptors=data["descriptors"],
        scores=data["scores"],
        image_size=data["image_size"],
        schema_version=int(np.asarray(data["schema_version"]).item()) if "schema_version" in data else 0,
        coordinate_convention=str(np.asarray(data["coordinate_convention"]).item()) if "coordinate_convention" in data else "pixel",
        image_size_convention=str(np.asarray(data["image_size_convention"]).item()) if "image_size_convention" in data else "hw",
        original_image_size=(data["original_image_size"] if "original_image_size" in data else data["image_size"]),
    )


def _resolve_image_path(image_path: str, dataset_root: Path) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return dataset_root / path


def _decode_mask_from_row(row: Any, mask_col: str, idx: int) -> np.ndarray:
    raw_mask = row.get(mask_col)
    if raw_mask is None:
        raise ValueError(f"Missing mask at row index {idx}")
    if isinstance(raw_mask, float) and np.isnan(raw_mask):
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


def _load_raw_rgb_image(
    row: Any,
    idx: int,
    dataset_root: Path,
    path_col: str,
    no_background: bool,
    mask_col: str,
) -> Image.Image:
    if path_col not in row.index:
        raise KeyError(f"Vismatch method requires path_col='{path_col}' in dataset dataframe")
    image_path = _resolve_image_path(str(row[path_col]), dataset_root=dataset_root)
    ensure_file(image_path, "Vismatch image file")
    image = Image.open(image_path).convert("RGB")
    if not no_background:
        return image

    image_np = np.asarray(image, dtype=np.uint8)
    mask = _decode_mask_from_row(row=row, mask_col=mask_col, idx=idx)
    if image_np.shape[0] != mask.shape[0] or image_np.shape[1] != mask.shape[1]:
        raise ValueError(
            f"Mask/Image size mismatch at row index {idx}: mask={mask.shape}, image={image_np.shape[:2]}"
        )
    image_np = image_np * np.expand_dims(np.asfortranarray(mask), axis=-1)
    return Image.fromarray(image_np)


@torch.no_grad()
def _extract_frame(backend: VismatchMatcherBackend, image_tensor: torch.Tensor) -> FrameFeatures:
    return backend.extract_frame(image_tensor)


def _score_pair(backend: VismatchMatcherBackend, fa: FrameFeatures, fb: FrameFeatures) -> Tuple[float, int]:
    return backend.score_features(fa, fb)


@torch.no_grad()
def _loma_points_to_pixel(points: np.ndarray, feat: FrameFeatures) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    processed_height, processed_width = (float(value) for value in feat.image_size)
    original_size = feat.original_image_size if feat.original_image_size is not None else feat.image_size
    original_height, original_width = (float(value) for value in original_size)
    pixel = points.astype(np.float32, copy=True)
    pixel[:, 0] = processed_width * (pixel[:, 0] + 1.0) / 2.0
    pixel[:, 1] = processed_height * (pixel[:, 1] + 1.0) / 2.0
    pixel[:, 0] *= original_width / processed_width
    pixel[:, 1] *= original_height / processed_height
    return pixel


def _match_frames(backend: VismatchMatcherBackend, fa: FrameFeatures, fb: FrameFeatures) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = backend.match_features(fa, fb)
    mkpts0 = result.matched_kpts0 if result.matched_kpts0 is not None else np.empty((0, 2), dtype=np.float32)
    mkpts1 = result.matched_kpts1 if result.matched_kpts1 is not None else np.empty((0, 2), dtype=np.float32)
    if backend.matcher_name == "loma":
        mkpts0 = _loma_points_to_pixel(mkpts0, fa)
        mkpts1 = _loma_points_to_pixel(mkpts1, fb)
    confidence = result.confidences if result.confidences is not None else np.asarray([], dtype=np.float32)
    return mkpts0, mkpts1, confidence


def _draw_matches_save(
    out_path: Path,
    img0: np.ndarray,
    img1: np.ndarray,
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    conf: np.ndarray,
    max_matches: int,
    title: str,
) -> str:
    if len(conf) > int(max_matches):
        order = np.argsort(-conf)[: int(max_matches)]
        mkpts0 = mkpts0[order]
        mkpts1 = mkpts1[order]
        conf = conf[order]

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    target_h = max(h0, h1)
    target_w = max(w0, w1)

    img0, mkpts0 = _letterbox_image_and_keypoints(img0, mkpts0, target_h=target_h, target_w=target_w)
    img1, mkpts1 = _letterbox_image_and_keypoints(img1, mkpts1, target_h=target_h, target_w=target_w)

    out = np.zeros((target_h, target_w * 2, 3), dtype=np.uint8)
    out[:, :target_w] = img0
    out[:, target_w:] = img1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(out)
    ax.axis("off")
    if title:
        ax.set_title(title)

    if len(conf) > 0:
        conf_norm = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
        colors = cm.viridis(conf_norm)
        for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, colors):
            ax.plot([x0, x1 + target_w], [y0, y1], color=c, linewidth=1)
        ax.scatter(mkpts0[:, 0], mkpts0[:, 1], s=6, c=colors, marker="o")
        ax.scatter(mkpts1[:, 0] + target_w, mkpts1[:, 1], s=6, c=colors, marker="o")
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=float(conf.min()), vmax=float(conf.max())))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("match confidence")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _letterbox_image_and_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: int = 114,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image, keypoints

    scale = min(float(target_w) / float(w), float(target_h) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))

    canvas = np.full((target_h, target_w, 3), fill_value=pad_value, dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    if keypoints.size == 0:
        return canvas, keypoints
    remapped = keypoints.astype(np.float32, copy=True)
    remapped[:, 0] = remapped[:, 0] * scale + float(pad_x)
    remapped[:, 1] = remapped[:, 1] * scale + float(pad_y)
    return canvas, remapped


def _extract_split_features(
    dataset: Any,
    split_name: str,
    backend: VismatchMatcherBackend,
    device: torch.device,
    top_k: int,
    resize_max: int,
    cache_dir: Path,
    dataset_root: Path,
    no_background: bool,
    mask_col: str,
    path_col: str,
    cfg_tag: str,
) -> List[FrameFeat]:
    if path_col not in dataset.df.columns:
        raise KeyError(f"Vismatch method requires path_col='{path_col}' in dataset dataframe")

    feats: List[FrameFeat] = []
    iterator = tqdm(range(len(dataset)), desc=f"[vismatch][extract:{split_name}]", mininterval=1, ncols=120)
    for idx in iterator:
        image_path = str(dataset.df.iloc[idx][path_col])
        key = _cache_key(image_path=image_path, split_name=split_name, resize_max=resize_max, top_k=top_k, cfg_tag=cfg_tag)
        cp = _cache_path(cache_dir, key)
        if cp.is_file():
            try:
                cached = _load_cached_feat(cp)
                if cached.schema_version == FEATURE_SCHEMA_VERSION:
                    feats.append(cached)
                    continue
            except (OSError, ValueError, KeyError):
                pass
        row = dataset.df.iloc[idx]
        image = _load_raw_rgb_image(
            row=row,
            idx=idx,
            dataset_root=dataset_root,
            path_col=path_col,
            no_background=no_background,
            mask_col=mask_col,
        )
        image_tensor = _to_image_tensor(image=image, resize_max=resize_max)
        feat = _extract_frame(backend=backend, image_tensor=image_tensor)
        _save_cached_feat(cp, feat)
        feats.append(feat)
    return feats


def run_vismatch_benchmark(
    cfg: Any,
    dataset_query: Any,
    dataset_database: Any,
    run_dir: Path,
    checkpoint_path: Optional[Path] = None,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    candidate_indices: Optional[np.ndarray] = None,
    method_artifacts: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    settings = cfg.benchmark.methods.vismatch
    matcher_name = validate_matcher_name(str(settings.matcher))
    device = _choose_vismatch_device(str(settings.device))
    top_k = int(settings.top_k)
    resize_max = int(settings.resize_max)
    configured_threshold = getattr(settings, "matcher_threshold", None)
    threshold = (
        default_matcher_threshold(matcher_name)
        if configured_threshold is None
        else float(configured_threshold)
    )
    path_col = str(settings.path_col)
    mask_col = str(cfg.dataset.mask_col)
    dataset_root = Path(str(cfg.dataset.root))
    no_background = bool(cfg.dataset.no_background)
    cache_dir = Path(str(settings.cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_tag = "none"
    if checkpoint_path is not None:
        checkpoint_tag = str(checkpoint_path)
        if checkpoint_path.is_file():
            stat = checkpoint_path.stat()
            checkpoint_tag = f"{checkpoint_path}|{stat.st_size}|{int(stat.st_mtime)}"
    stage_a_method = str(settings.stage_a_method) if "stage_a_method" in settings else "none"
    feature_matching_mode = str(getattr(settings, "feature_matching_mode", "feature_level"))
    profile = build_matcher_profile(matcher_name, top_k, threshold, feature_matching_mode)
    cfg_tag = hashlib.sha256(
        (
            f"matcher={matcher_name}|vismatch_commit={VISMATCH_COMMIT}|stage_a={stage_a_method}|"
            f"model_type={cfg.model.type}|model_mode={cfg.model.mode}|checkpoint={checkpoint_tag}|"
            f"top_k={top_k}|resize_max={resize_max}|threshold={threshold}|no_bg={no_background}|"
            f"path_col={path_col}|mask_col={mask_col}|feature_matching_mode={feature_matching_mode}|schema={FEATURE_SCHEMA_VERSION}"
        ).encode("utf-8")
    ).hexdigest()
    print(
        "[vismatch] cache tag components: "
        f"matcher={matcher_name} stage_a={stage_a_method} model_type={cfg.model.type} "
        f"model_mode={cfg.model.mode} checkpoint={checkpoint_tag} no_background={no_background} "
        f"top_k={top_k} resize_max={resize_max} threshold={threshold}"
    )

    t_build = time.perf_counter()
    backend = VismatchMatcherBackend(matcher_name, device, top_k, threshold, feature_matching_mode)
    model_build_sec = time.perf_counter() - t_build

    t_extract = time.perf_counter()
    query_feats = _extract_split_features(dataset_query, "query", backend, device, top_k, resize_max, cache_dir, dataset_root, no_background, mask_col, path_col, cfg_tag)
    db_feats = _extract_split_features(dataset_database, "database", backend, device, top_k, resize_max, cache_dir, dataset_root, no_background, mask_col, path_col, cfg_tag)
    extract_sec = time.perf_counter() - t_extract

    t_sim = time.perf_counter()
    similarity = np.full((len(query_feats), len(db_feats)), fill_value=-1e9, dtype=np.float32)
    match_counts: List[int] = []
    for qi in tqdm(range(len(query_feats)), desc="[vismatch][match]", mininterval=1, ncols=120):
        db_candidates = range(len(db_feats)) if candidate_indices is None else candidate_indices[qi].tolist()
        for di in db_candidates:
            score, nm = _score_pair(backend, query_feats[qi], db_feats[di])
            similarity[qi, di] = float(score)
            match_counts.append(int(nm))
    rerank_sec = time.perf_counter() - t_sim

    if bool(cfg.visualization.enabled):
        rng = np.random.default_rng(int(cfg.benchmark.seed))
        num_examples = min(int(cfg.visualization.num_examples), len(dataset_query))
        sampled_indices = rng.choice(np.arange(len(dataset_query)), size=num_examples, replace=False) if num_examples > 0 else []
        max_matches = int(getattr(cfg.visualization, "vismatch_max_matches", 200))
        out_dir = Path(cfg.visualization.dir) / run_dir.name
        match_paths: List[str] = []
        for q_idx in sampled_indices:
            q_idx_int = int(q_idx)
            db_idx = int(np.argmax(similarity[q_idx_int]))
            q_row = dataset_query.df.iloc[q_idx_int]
            db_row = dataset_database.df.iloc[db_idx]
            q_img = _load_raw_rgb_image(q_row, q_idx_int, dataset_root, path_col, no_background, mask_col)
            db_img = _load_raw_rgb_image(db_row, db_idx, dataset_root, path_col, no_background, mask_col)
            q_vis = _to_uint8_rgb(q_img, resize_max=resize_max, mean=mean, std=std)
            db_vis = _to_uint8_rgb(db_img, resize_max=resize_max, mean=mean, std=std)
            mkpts0, mkpts1, conf = _match_frames(backend, query_feats[q_idx_int], db_feats[db_idx])
            q_label = str(dataset_query.df.iloc[q_idx_int][cfg.dataset.label_col])
            db_label = str(dataset_database.df.iloc[db_idx][cfg.dataset.label_col])
            out_path = out_dir / f"vismatch_matches_q{q_idx_int}_db{db_idx}.png"
            match_paths.append(_draw_matches_save(out_path, q_vis, db_vis, mkpts0, mkpts1, conf, max_matches, f"q{q_idx_int}:{q_label} -> db{db_idx}:{db_label} | matcher={matcher_name}"))
        if method_artifacts is not None:
            method_artifacts["vismatch_match_paths"] = match_paths

    timings = {
        "vismatch_model_build_sec": float(model_build_sec),
        "vismatch_feature_extraction_sec": float(extract_sec),
        "vismatch_rerank_sec": float(rerank_sec),
        "total_method_sec": float(model_build_sec + extract_sec + rerank_sec),
    }
    method_metrics = {
        "vismatch_avg_matches": float(np.mean(match_counts)) if match_counts else 0.0,
    }
    return similarity, timings, method_metrics
