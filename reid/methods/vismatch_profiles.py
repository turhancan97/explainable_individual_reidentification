"""Lightweight Vismatch matcher registry and reproducibility profiles."""

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from typing import Iterable

VISMATCH_COMMIT = "4a743b75749a3770af59d275483ed341dea51ff0"
VISMATCH_PREPROCESSING_VERSION = "lynx_finetuning_v1"
LOMA_PREPROCESSING_VERSION = "lynx_loma_finetuning_v1"
FEATURE_SCHEMA_VERSION = 3
SUPPORTED_VISMATCH_MATCHERS = (
    "rdd-lightglue",
    "aliked-lightglue",
    "superpoint-lightglue",
    "loma",
)


@dataclass
class FrameFeatures:
    """Dependency-light, versioned feature container shared by cache and matcher code."""

    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray
    image_size: np.ndarray
    schema_version: int = FEATURE_SCHEMA_VERSION
    coordinate_convention: str = "pixel"
    image_size_convention: str = "hw"
    original_image_size: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.keypoints.ndim != 2 or self.keypoints.shape[1] != 2:
            raise ValueError(f"Vismatch keypoints must have shape (N, 2), got {self.keypoints.shape}")
        if self.descriptors.ndim != 2 or self.descriptors.shape[0] != self.keypoints.shape[0]:
            raise ValueError(
                "Vismatch descriptors must have shape (N, D) with the same N as keypoints, "
                f"got descriptors={self.descriptors.shape}, keypoints={self.keypoints.shape}"
            )
        if self.scores.ndim != 1 or self.scores.shape[0] != self.keypoints.shape[0]:
            raise ValueError(
                "Vismatch scores must have shape (N,) with the same N as keypoints, "
                f"got scores={self.scores.shape}, keypoints={self.keypoints.shape}"
            )
        if self.image_size.shape != (2,) or np.any(self.image_size <= 0):
            raise ValueError(f"Vismatch image_size must be positive (H, W), got {self.image_size}")
        if self.coordinate_convention not in {"pixel", "normalized[-1,1]"} or self.image_size_convention != "hw":
            raise ValueError(
                "Vismatch features must use pixel or normalized[-1,1] coordinates "
                "and image_size convention 'hw'"
            )
        if self.original_image_size is not None and (
            self.original_image_size.shape != (2,) or np.any(self.original_image_size <= 0)
        ):
            raise ValueError(
                f"Vismatch original_image_size must be positive (H, W), got {self.original_image_size}"
            )


@dataclass(frozen=True)
class MatcherProfile:
    matcher: str
    framework: str
    framework_version: str
    preprocessing: str
    keypoint_budget: int
    threshold: float
    score_mode: str
    feature_matching_mode: str = "feature_level"
    feature_schema_version: int = FEATURE_SCHEMA_VERSION
    preprocessing_version: str = VISMATCH_PREPROCESSING_VERSION


def validate_matcher_name(matcher: str) -> str:
    matcher = str(matcher).strip().lower()
    if matcher not in SUPPORTED_VISMATCH_MATCHERS:
        supported = ", ".join(SUPPORTED_VISMATCH_MATCHERS)
        raise ValueError(f"Unsupported Vismatch matcher '{matcher}'. Supported matchers: {supported}")
    return matcher


def default_matcher_threshold(matcher: str) -> float:
    matcher = validate_matcher_name(matcher)
    return 0.1 if matcher == "loma" else 0.01


def build_matcher_profile(matcher: str, top_k: int, threshold: float, feature_matching_mode: str = "feature_level") -> MatcherProfile:
    matcher = validate_matcher_name(matcher)
    if feature_matching_mode not in {"feature_level", "pairwise"}:
        raise ValueError("feature_matching_mode must be 'feature_level' or 'pairwise'")
    preprocessing = (
        "RGB float32 [0,1]; Lynx fine-tuning tensor resize; target long side; "
        "floor each spatial dimension to a multiple of 32; bilinear align_corners=False"
    )
    preprocessing_version = VISMATCH_PREPROCESSING_VERSION
    score_mode = "matched_confidence_sum_over_min_keypoints"
    if matcher == "loma":
        preprocessing = (
            "RGB float32 [0,1]; LoMa fine-tuning tensor resize; target long side; "
            "floor each spatial dimension to a multiple of 14 for DINOv2-L/14; "
            "bilinear align_corners=False"
        )
        preprocessing_version = LOMA_PREPROCESSING_VERSION
        score_mode = "mutual_confidence_sum_over_min_keypoints"
    return MatcherProfile(
        matcher=matcher,
        framework="Vismatch",
        framework_version=VISMATCH_COMMIT,
        preprocessing=preprocessing,
        keypoint_budget=int(top_k),
        threshold=float(threshold),
        score_mode=score_mode,
        feature_matching_mode=feature_matching_mode,
        preprocessing_version=preprocessing_version,
    )


def profile_fingerprint(profile: MatcherProfile) -> str:
    payload = json.dumps(asdict(profile), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_match_confidences(
    confidences: Iterable[float],
    query_keypoints: int,
    gallery_keypoints: int,
    threshold: float = 0.0,
) -> tuple[float, int, list[float]]:
    values = [float(value) for value in confidences if float(value) >= float(threshold)]
    denominator = min(max(1, int(query_keypoints)), max(1, int(gallery_keypoints)))
    return sum(values) / float(denominator), len(values), values
