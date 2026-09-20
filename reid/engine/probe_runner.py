import hashlib
import json
import math
import random
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from wildlife_datasets.datasets import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.features.local import AlikedExtractor
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.calibration import IsotonicCalibration
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion

from models.model import get_model
from models.objective import SoftmaxLoss, SoftmaxLossEP
from reid.data.safety_checks import run_split_safety_checks
from reid.data.dataset_view import BenchmarkDatasetView
from reid.evaluation.candidate_scoring import candidate_recall_metrics, save_score_matrix
from reid.evaluation.metrics import compute_identity_metrics, compute_metrics
from reid.evaluation.ranking import stable_rank_indices
from reid.features.containers import FeatureContainer, get_labels_string
from reid.methods.vismatch import run_vismatch_benchmark
from reid.methods.wildfusion_calibration import fit_pipeline_calibration, fit_wildfusion_calibration
from reid.reporting.artifacts import build_run_context, file_identity, run_index_row, upsert_run_index
from reid.reporting.timing import set_primary_compute_runtime
from reid.reporting.visualizations import finalize_visualizations
from reid.reporting.wandb_naming import probe_wandb_name
from reid.training.accumulation import accumulation_group_size, should_step_accumulated_gradients
from reid.training.checkpointing import resolve_configured_model_checkpoint, resolve_model_checkpoint
from reid.training.class_weights import compute_identity_class_weights
from reid.utils.cache_identity import build_dataset_cache_identity
from reid.utils.fingerprints import file_digest_cache, model_fingerprint, sha256_file
from reid.utils.io import append_csv_row, ensure_dir, ensure_file
from reid.utils.repro import set_reproducible


def _format_metric_value(value: Any) -> str:
    """Format console metrics without assuming every diagnostic is numeric."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _instrument_local_pipeline(pipeline: Any, timings: Dict[str, float]) -> None:
    """Time local feature extraction and pairwise matching separately.

    ``wildlife-tools`` pipelines expose both operations as callables inside
    ``SimilarityPipeline``. Wrapping those callables preserves the package's
    scoring path while making the Stage-B matcher boundary observable.
    """

    original_get_features = pipeline.get_feature_dataset

    def timed_get_features(dataset: Any) -> Any:
        started = time.perf_counter()
        result = original_get_features(dataset)
        timings["feature_extraction_compute_sec"] = timings.get("feature_extraction_compute_sec", 0.0) + (
            time.perf_counter() - started
        )
        return result

    pipeline.get_feature_dataset = timed_get_features
    original_matcher = pipeline.matcher

    def timed_matcher(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = original_matcher(*args, **kwargs)
        timings["matcher_runtime_sec"] = timings.get("matcher_runtime_sec", 0.0) + (
            time.perf_counter() - started
        )
        return result

    pipeline.matcher = timed_matcher


PROBE_CSV_METADATA_COLUMNS = [
    "run_id",
    "run_utc",
    "method",
    "device",
    "model_type",
    "model_mode",
    "no_background",
    "image_variant",
    "checkpoint_path",
    "dataset_root",
    "metadata_file",
    "num_query",
    "num_database",
]

PROBE_CSV_METRIC_COLUMNS = [
    "top_1",
    "top_5",
    "top_10",
    "mAP",
    "balanced_top_1",
    "classification_top_1",
    "classification_top_5",
    "classification_top_10",
    "classification_balanced_top_1",
    "classification_seen_top_1",
    "classification_seen_top_5",
    "classification_seen_top_10",
    "classification_seen_balanced_top_1",
    "classification_open_top_1",
    "classification_open_top_5",
    "classification_open_top_10",
    "classification_open_balanced_top_1",
    "classification_num_query_images",
    "classification_num_seen_query_images",
    "classification_num_unseen_query_images",
    "classification_num_query_identities",
    "classification_num_seen_query_identities",
    "classification_num_unseen_query_identities",
    "classification_query_seen_coverage",
    "classifier_open_set_policy",
    "classification_embedding_retrieval_enabled",
    "vismatch_avg_matches",
    "num_candidate_pairs",
    "num_unscored_pairs",
    "candidate_fraction",
    "score_matrix_policy",
    "linear_probe_class_weighting",
    "linear_probe_class_weight_formula",
    "linear_probe_class_weight_normalize",
    "linear_probe_class_weight_cap",
    "linear_probe_class_weight_min",
    "linear_probe_class_weight_max",
    "linear_probe_class_weight_mean",
    "linear_probe_class_weight_median",
    "linear_probe_class_weight_num_identities",
    "linear_probe_class_weight_stats",
    "efficient_probe_class_weighting",
    "efficient_probe_class_weight_formula",
    "efficient_probe_class_weight_normalize",
    "efficient_probe_class_weight_cap",
    "efficient_probe_class_weight_min",
    "efficient_probe_class_weight_max",
    "efficient_probe_class_weight_mean",
    "efficient_probe_class_weight_median",
    "efficient_probe_class_weight_num_identities",
    "efficient_probe_class_weight_stats",
]

PROBE_CSV_TIMING_COLUMNS = [
    "primary_compute_runtime_sec",
    "matcher_runtime_sec",
    "method_compute_runtime_sec",
    "feature_extraction_compute_sec",
    "feature_cache_lookup_sec",
    "feature_cache_hits",
    "feature_cache_misses",
    "model_setup_sec",
    "calibration_sec",
    "feature_extraction_sec",
    "similarity_sec",
    "vismatch_stage_a_sec",
    "benchmark_candidate_k",
    "vismatch_candidate_k",
    "vismatch_model_build_sec",
    "vismatch_feature_extraction_sec",
    "vismatch_rerank_sec",
    "linear_probe_train_sec",
    "linear_probe_eval_sec",
    "efficient_probe_train_sec",
    "efficient_probe_eval_sec",
    "total_method_sec",
    "total_run_sec",
    "total_run_min",
]


def _as_csv_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, int, np.floating, np.integer, bool, np.bool_)):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _build_probe_csv_row(
    base: Dict[str, Any],
    metrics: Dict[str, Any],
    timings: Dict[str, Any],
) -> Dict[str, Any]:
    known = set(PROBE_CSV_METADATA_COLUMNS + PROBE_CSV_METRIC_COLUMNS + PROBE_CSV_TIMING_COLUMNS)
    extra_metric_columns = sorted([k for k in metrics.keys() if k not in known])
    extra_timing_columns = sorted([k for k in timings.keys() if k not in known])
    columns = (
        PROBE_CSV_METADATA_COLUMNS
        + PROBE_CSV_METRIC_COLUMNS
        + extra_metric_columns
        + PROBE_CSV_TIMING_COLUMNS
        + extra_timing_columns
    )

    row: Dict[str, Any] = {col: "" for col in columns}
    for k, v in base.items():
        row[k] = _as_csv_scalar(v)
    for k, v in metrics.items():
        row[k] = _as_csv_scalar(v)
    for k, v in timings.items():
        row[k] = _as_csv_scalar(v)
    return row


def choose_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_latest_checkpoint(results_dir: Path, filename: str) -> Path:
    return resolve_model_checkpoint(results_dir=results_dir, filename=filename)


def load_backbone(
    cfg: DictConfig,
) -> Tuple[Any, int, Tuple[float, ...], Tuple[float, ...], int, str, Optional[int], Optional[Path]]:
    model, embedding_size, mean, std, img_size, arch, patch_size, number_of_patches = get_model(cfg.model.type)
    checkpoint_path: Optional[Path] = None

    if cfg.model.mode == "finetuned":
        if cfg.model.checkpoint.path:
            checkpoint_path = resolve_configured_model_checkpoint(
                explicit_path=Path(cfg.model.checkpoint.path),
                results_dir=Path(cfg.model.checkpoint.results_dir),
                filename=cfg.model.checkpoint.filename,
                fallback_dirs=[Path(cfg.model.checkpoint.legacy_results_dir)]
                if getattr(cfg.model.checkpoint, "legacy_results_dir", None)
                else [],
            )
        elif cfg.model.checkpoint.from_latest_results:
            checkpoint_path = resolve_configured_model_checkpoint(
                explicit_path=None,
                results_dir=Path(cfg.model.checkpoint.results_dir),
                filename=cfg.model.checkpoint.filename,
                fallback_dirs=[Path(cfg.model.checkpoint.legacy_results_dir)]
                if getattr(cfg.model.checkpoint, "legacy_results_dir", None)
                else [],
            )
        else:
            raise ValueError("Finetuned mode requires checkpoint.path or checkpoint.from_latest_results=true")

        ensure_file(checkpoint_path, "Checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint)

    return model, embedding_size, mean, std, img_size, arch, number_of_patches, checkpoint_path


def build_transforms(mean: Tuple[float, ...], std: Tuple[float, ...], img_size: int) -> Tuple[T.Compose, T.Compose, T.Compose]:
    transform_display = T.Compose([T.Resize([img_size, img_size])])
    transform_model = T.Compose([*transform_display.transforms, T.ToTensor(), T.Normalize(mean=mean, std=std)])
    transform_aliked = T.Compose([T.Resize([512, 512]), T.ToTensor()])
    return transform_display, transform_model, transform_aliked


def load_dataset_splits(cfg: DictConfig) -> Tuple[WildlifeDataset, WildlifeDataset, WildlifeDataset]:
    root = Path(cfg.dataset.root)
    build_dataset_cache_identity(cfg.dataset)
    ensure_dir(root, "Dataset root")
    metadata_path = root / cfg.dataset.metadata_file
    ensure_file(metadata_path, "Metadata CSV")

    metadata = pd.read_csv(metadata_path)
    if cfg.dataset.split_col not in metadata.columns:
        raise KeyError(f"split_col '{cfg.dataset.split_col}' not found in metadata columns")
    if cfg.dataset.label_col not in metadata.columns:
        raise KeyError(f"label_col '{cfg.dataset.label_col}' not found in metadata columns")
    if bool(cfg.dataset.no_background) and cfg.dataset.mask_col not in metadata.columns:
        raise KeyError(
            f"mask_col '{cfg.dataset.mask_col}' not found in metadata while dataset.no_background=true"
        )

    dataset = WildlifeDataset(
        str(root),
        metadata,
        transform=None,
        load_label=True,
        col_label=cfg.dataset.label_col,
    )
    dataset_database = dataset.get_subset(dataset.metadata[cfg.dataset.split_col] == cfg.dataset.database_split_value)
    dataset_query = dataset.get_subset(dataset.metadata[cfg.dataset.split_col] == cfg.dataset.query_split_value)
    return dataset, dataset_database, dataset_query


def get_calibration_dataset(
    dataset_database: WildlifeDataset,
    size: int,
    root: str,
    label_col: str,
    source_dataset: Optional[WildlifeDataset] = None,
    split_col: Optional[str] = None,
    split_value: Optional[str] = None,
) -> WildlifeDataset:
    if size <= 0:
        raise ValueError("dataset.calibration_size must be > 0")
    source = source_dataset
    if source is not None and split_col and split_value is not None and split_col in source.metadata.columns:
        selected = source.metadata[source.metadata[split_col] == split_value]
        if len(selected) > 0:
            return WildlifeDataset(root, df=selected.iloc[:size], load_label=True, col_label=label_col)
    if len(dataset_database.metadata) < size:
        size = len(dataset_database.metadata)
    return WildlifeDataset(root, df=dataset_database.metadata.iloc[:size], load_label=True, col_label=label_col)

def dataset_digest(dataset: WildlifeDataset, label_col: str, root: Optional[Path] = None) -> str:
    df = dataset.df if hasattr(dataset, "df") else dataset.metadata
    candidate_cols = ["path", "filepath", "file", "image_path", label_col]
    cols = [c for c in candidate_cols if c in df.columns]
    if not cols:
        cols = [label_col] if label_col in df.columns else []
    if cols:
        records = []
        path_col = next((c for c in ("path", "filepath", "file", "image_path") if c in df.columns), None)
        for _, row in df[cols].iterrows():
            values = [str(row[col]) for col in cols]
            if path_col is not None and root is not None:
                path = Path(values[cols.index(path_col)])
                if not path.is_absolute():
                    path = root / path
                try:
                    values.append(sha256_file(path))
                except (OSError, FileNotFoundError):
                    values.append("missing")
            records.append("|".join(values))
        payload = "\n".join(records)
    else:
        payload = f"n={len(df)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FeatureCache:
    def __init__(self, enabled: bool, cache_dir: Path, fmt: str):
        self.enabled = enabled
        self.cache_dir = cache_dir
        self.fmt = fmt
        self.used_keys: List[str] = []
        self.lookup_sec = 0.0
        self.compute_sec = 0.0
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.fmt not in {"pt", "npz"}:
            raise ValueError("benchmark.cache.format must be 'pt' or 'npz'")

    def _path_for(self, key: str) -> Path:
        ext = "pt" if self.fmt == "pt" else "npz"
        return self.cache_dir / f"{key}.{ext}"

    def get_or_compute(self, key: str, compute_fn) -> np.ndarray:
        self.used_keys.append(str(key))
        path = self._path_for(key)
        t_lookup = time.perf_counter()
        if self.enabled and path.is_file():
            data = self._load(path)
            self.lookup_sec += time.perf_counter() - t_lookup
            self.hits += 1
            return data
        else:
            self.lookup_sec += time.perf_counter() - t_lookup
        self.misses += 1
        t_compute = time.perf_counter()
        data = self._normalize_features(compute_fn())
        self.compute_sec += time.perf_counter() - t_compute
        if self.enabled:
            self._save(path, data)
        return data

    def _load(self, path: Path) -> np.ndarray:
        if self.fmt == "pt":
            loaded = torch.load(path, map_location="cpu")
        else:
            loaded = np.load(path)["features"]
        return self._normalize_features(loaded)

    def _save(self, path: Path, data: np.ndarray) -> None:
        if self.fmt == "pt":
            torch.save(torch.from_numpy(data), path)
        else:
            np.savez_compressed(path, features=data)

    def _to_numpy(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _maybe_extract_feature_container(self, value: Any) -> Any:
        if isinstance(value, dict):
            for key in ("features", "embeddings", "vectors", "x"):
                if key in value:
                    return value[key]
            return value

        if isinstance(value, pd.DataFrame):
            for key in ("features", "embeddings", "vectors"):
                if key in value.columns:
                    return value[key].tolist()
            if value.shape[1] >= 1:
                return value.iloc[:, 0].tolist()
            return value

        for attr in ("features", "embeddings", "vectors"):
            if hasattr(value, attr):
                return getattr(value, attr)

        return value

    def _extract_from_sequence_rows(self, seq: Any) -> Optional[np.ndarray]:
        if not isinstance(seq, (list, tuple)) or len(seq) == 0:
            return None

        first = seq[0]
        if isinstance(first, (list, tuple)) and len(first) >= 1:
            first_item = first[0]
            if isinstance(first_item, (np.ndarray, list, tuple)) or torch.is_tensor(first_item):
                rows = [self._to_numpy(row[0]) for row in seq]
                return np.stack(rows, axis=0)
        return None

    def _normalize_features(self, raw: Any) -> np.ndarray:
        value = self._maybe_extract_feature_container(raw)

        if isinstance(value, tuple) and len(value) >= 1:
            first = value[0]
            if isinstance(first, (np.ndarray, list, tuple)) or torch.is_tensor(first):
                value = first

        maybe_rows = self._extract_from_sequence_rows(value)
        if maybe_rows is not None:
            arr = maybe_rows
        elif isinstance(value, (list, tuple)):
            rows = [self._to_numpy(v) for v in value]
            try:
                arr = np.stack(rows, axis=0)
            except ValueError:
                arr = np.asarray(rows, dtype=np.float32)
        else:
            try:
                arr = self._to_numpy(value)
            except ValueError:
                try:
                    seq = list(value)
                except Exception as exc:
                    raise ValueError(
                        f"Could not convert features to numpy. type={type(raw)}"
                    ) from exc
                maybe_rows = self._extract_from_sequence_rows(seq)
                if maybe_rows is None:
                    raise ValueError(
                        f"Could not parse structured feature output. type={type(raw)} first_item={type(seq[0]) if seq else None}"
                    )
                arr = maybe_rows

        if arr.dtype == object:
            try:
                arr = np.stack([np.asarray(x) for x in arr], axis=0)
            except Exception as exc:
                raise ValueError(
                    f"Could not normalize extracted features to a numeric array. "
                    f"Received type={type(raw)} with object dtype."
                ) from exc

        if arr.ndim == 1:
            raise ValueError(
                f"Extracted features must be at least 2D (N, D). Got shape={arr.shape}, type={type(raw)}."
            )
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        return arr.astype(np.float32, copy=False)


class EncodedLabelDataset(Dataset):
    def __init__(self, dataset: Any, label_to_index: Dict[str, int], allow_unknown: bool = False):
        self.dataset = dataset
        self.label_to_index = label_to_index
        self.allow_unknown = bool(allow_unknown)
        self.df = dataset.df
        self.metadata = dataset.metadata
        self.col_label = dataset.col_label

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError("EncodedLabelDataset expects dataset items as (image, label)")
        image = item[0]
        raw_label = item[1]
        label_str = str(raw_label)
        if label_str not in self.label_to_index:
            if self.allow_unknown:
                return image, -1
            raise ValueError(f"Label '{label_str}' not found in label mapping")
        return image, self.label_to_index[label_str]


def _build_label_mapping(dataset_database: Any, label_col: str) -> Dict[str, int]:
    labels = dataset_database.df[label_col].astype(str).tolist()
    unique_labels = sorted(set(labels))
    return {label: idx for idx, label in enumerate(unique_labels)}


CLASSIFIER_OPEN_SET_POLICIES = {"open", "warn", "closed"}


def resolve_classifier_evaluation(cfg: DictConfig) -> Tuple[str, bool]:
    """Resolve the shared classifier-probe open-set policy and diagnostics flag."""
    settings = getattr(cfg.benchmark, "classifier_evaluation", {})
    if isinstance(settings, dict):
        policy = str(settings.get("open_set_policy", "open")).lower()
        embedding_retrieval = bool(settings.get("embedding_retrieval", False))
    else:
        policy = str(getattr(settings, "open_set_policy", "open")).lower()
        embedding_retrieval = bool(getattr(settings, "embedding_retrieval", False))
    if policy not in CLASSIFIER_OPEN_SET_POLICIES:
        allowed = ", ".join(sorted(CLASSIFIER_OPEN_SET_POLICIES))
        raise ValueError(f"benchmark.classifier_evaluation.open_set_policy must be one of: {allowed}")
    return policy, embedding_retrieval


def _label_indices(labels: Any, label_to_index: Dict[str, int]) -> np.ndarray:
    """Map labels to classifier indices, using -1 for identities unseen in training."""
    return np.asarray(
        [label_to_index.get(str(label), -1) for label in list(labels)],
        dtype=np.int64,
    )


def classifier_open_set_coverage(
    query_labels: Any,
    query_labels_idx: np.ndarray,
    label_to_index: Dict[str, int],
    policy: str,
) -> Dict[str, Any]:
    """Return auditable seen/unseen query counts for classifier probes."""
    labels = [str(label) for label in list(query_labels)]
    unique_labels = set(labels)
    seen_unique = {label for label in unique_labels if label in label_to_index}
    num_query = len(labels)
    num_seen = int(np.count_nonzero(np.asarray(query_labels_idx) >= 0))
    num_unseen = num_query - num_seen
    return {
        "classifier_open_set_policy": policy,
        "classification_num_query_images": float(num_query),
        "classification_num_seen_query_images": float(num_seen),
        "classification_num_unseen_query_images": float(num_unseen),
        "classification_num_query_identities": float(len(unique_labels)),
        "classification_num_seen_query_identities": float(len(seen_unique)),
        "classification_num_unseen_query_identities": float(len(unique_labels - seen_unique)),
        "classification_query_seen_coverage": float(num_seen / num_query) if num_query else float("nan"),
    }


def validate_classifier_open_set_labels(
    dataset_database: Any,
    dataset_query: Any,
    label_col: str,
    policy: str,
    emit_warning: bool = True,
) -> None:
    """Fail early only for strict classifier mode and warn for open modes."""
    database_labels = set(dataset_database.df[label_col].astype(str).tolist())
    query_labels = set(dataset_query.df[label_col].astype(str).tolist())
    unseen = sorted(query_labels - database_labels)
    if not unseen:
        return
    detail = (
        f"{len(unseen)} query identities are absent from the training/database identity set"
    )
    if policy == "closed":
        raise ValueError(
            f"Closed-set classifier evaluation failed: {detail}. "
            "Use open_set_policy=open to score unseen queries as incorrect."
        )
    if emit_warning:
        print(
            f"[classifier][warning] {detail}; unseen query samples will not contribute to "
            "validation loss and will receive zero classification credit."
        )


def _set_trainable_params(model: Any, cfg: DictConfig, method_key: str) -> None:
    method_cfg = cfg.benchmark.methods[method_key]
    mode = str(method_cfg.train_mode)
    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        return
    if mode == "classifier":
        for p in model.parameters():
            p.requires_grad = False
        return
    if mode == "partial":
        rules = method_cfg.partial_rules
        patterns = rules.get(cfg.model.type, rules.get("default", []))
        if not patterns:
            raise ValueError(
                f"No partial unfreeze patterns configured for model.type={cfg.model.type} and no default fallback"
            )
        for name, p in model.named_parameters():
            p.requires_grad = any(pattern in name for pattern in patterns)
        return
    raise ValueError(f"{method_key}.train_mode must be one of: all, partial, classifier")


def _set_probe_training_mode(model: Any, objective: Any, train_mode: str) -> None:
    """Set module modes without accidentally training a frozen probe backbone.

    In classifier mode the backbone is a fixed feature extractor. Calling
    ``model.train()`` would still switch dropout/stochastic-depth layers to
    training behavior and would update running statistics in stateful
    normalization layers, even though all backbone parameters are frozen.
    The classifier objective is a separate module and remains in training mode.
    """

    objective.train()
    if str(train_mode) == "classifier":
        model.eval()
    else:
        model.train()


def _build_optimizer(params, method_cfg: DictConfig, method_key: str):
    opt_name = str(method_cfg.optimizer).lower()
    lr = float(method_cfg.lr)
    weight_decay = float(method_cfg.weight_decay)
    if opt_name == "sgd":
        return SGD(params=params, lr=lr, momentum=float(method_cfg.momentum), weight_decay=weight_decay)
    if opt_name == "adam":
        return Adam(params=params, lr=lr, weight_decay=weight_decay)
    if opt_name == "adamw":
        return AdamW(params=params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"{method_key}.optimizer must be one of: sgd, adam, adamw")


def _classification_topk_accuracy(
    probs: np.ndarray,
    query_labels_idx: np.ndarray,
    topk_values: List[int],
    prefix: str = "classification",
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    ranked = stable_rank_indices(probs)
    num_classes = probs.shape[1]
    for k in topk_values:
        kk = min(int(k), num_classes)
        hits = [int(query_labels_idx[i]) in ranked[i, :kk] for i in range(len(query_labels_idx))]
        metrics[f"{prefix}_top_{k}"] = float(np.mean(hits)) if hits else float("nan")
    metrics[f"{prefix}_balanced_top_1"] = _balanced_accuracy_top1_idx(
        query_labels_idx=query_labels_idx,
        predicted_top1_labels=ranked[:, 0],
    )
    return metrics


def _classifier_metrics(
    probs: np.ndarray,
    query_labels_idx: np.ndarray,
    query_labels: Any,
    label_to_index: Dict[str, int],
    policy: str,
) -> Dict[str, Any]:
    """Build open-world, seen-only, and policy-selected classifier metrics."""
    probs = np.asarray(probs)
    query_labels_idx = np.asarray(query_labels_idx, dtype=np.int64)
    open_metrics = _classification_topk_accuracy(
        probs, query_labels_idx, [1, 5, 10], prefix="classification_open"
    )
    seen_mask = np.asarray(query_labels_idx) >= 0
    seen_metrics = _classification_topk_accuracy(
        probs[seen_mask], query_labels_idx[seen_mask], [1, 5, 10], prefix="classification_seen"
    )
    metrics: Dict[str, Any] = {
        **open_metrics,
        **seen_metrics,
        **classifier_open_set_coverage(query_labels, query_labels_idx, label_to_index, policy),
        "classification_embedding_retrieval_enabled": False,
    }
    selected_prefix = "classification_open" if policy == "open" else "classification_seen"
    for suffix in ("top_1", "top_5", "top_10", "balanced_top_1"):
        metrics[f"classification_{suffix}"] = metrics[f"{selected_prefix}_{suffix}"]
    return metrics


def _similarity_from_class_probs(probs_query: np.ndarray, db_labels_idx: np.ndarray) -> np.ndarray:
    return probs_query[:, db_labels_idx]


def _probe_retrieval_metrics(
    cfg: DictConfig,
    dataset_query: Any,
    dataset_database: Any,
    probs_query: np.ndarray,
    db_labels_idx: np.ndarray,
    query_labels_idx: np.ndarray,
) -> Dict[str, float]:
    """Return identity-level metrics plus image-level diagnostic metrics."""
    # `probs_query` has one column per identity, so an identity's score is already a
    # single value: every database image of that identity maps to the same column.
    # Selecting the identity columns directly is what the previous per-image maximum
    # was reaching for, and it avoids indexing a 319-wide class axis with an
    # image-length mask.
    identity_labels = np.unique(db_labels_idx)
    if probs_query.ndim != 2:
        raise ValueError(f"probs_query must be a two-dimensional array, got shape {probs_query.shape}")
    if identity_labels.size and int(identity_labels.max()) >= probs_query.shape[1]:
        raise ValueError(
            f"Database identity index {int(identity_labels.max())} is outside the classifier's "
            f"{probs_query.shape[1]} output classes; the label mapping and the head disagree."
        )
    identity_similarity = probs_query[:, identity_labels]
    primary = compute_identity_metrics(
        query_labels=query_labels_idx,
        identity_labels=identity_labels,
        similarity=identity_similarity,
        top_k_values=[int(k) for k in cfg.benchmark.top_k],
        compute_map=bool(cfg.benchmark.compute_map),
        map_at_k=resolve_map_at_k(cfg, len(db_labels_idx)),
    )
    diagnostic = compute_metrics(
        dataset_query=dataset_query,
        dataset_database=dataset_database,
        similarity=_similarity_from_class_probs(probs_query, db_labels_idx),
        top_k_values=[int(k) for k in cfg.benchmark.top_k],
        compute_map=bool(cfg.benchmark.compute_map),
        map_at_k=resolve_map_at_k(cfg, len(db_labels_idx)),
    )
    return {**primary, **{f"image_{key}": value for key, value in diagnostic.items()}}

def _balanced_accuracy_top1_idx(query_labels_idx: np.ndarray, predicted_top1_labels: np.ndarray) -> float:
    classes = np.unique(query_labels_idx)
    if classes.size == 0:
        return float("nan")

    recalls: List[float] = []
    for cls in classes:
        mask = query_labels_idx == cls
        denom = int(mask.sum())
        if denom == 0:
            continue
        tp = int((predicted_top1_labels[mask] == cls).sum())
        recalls.append(tp / denom)
    return float(np.mean(recalls)) if recalls else float("nan")


def _predict_class_probabilities(objective: Any, embeddings: torch.Tensor) -> torch.Tensor:
    if hasattr(objective, "predict_probabilities"):
        return objective.predict_probabilities(embeddings)
    if hasattr(objective, "linear"):
        logits = objective.linear(embeddings)
        return torch.softmax(logits, dim=1)
    raise AttributeError("Softmax objective must provide either predict_probabilities() or linear layer")


def _extract_hidden_state(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, dict) and "last_hidden_state" in outputs:
        return outputs["last_hidden_state"]
    if torch.is_tensor(outputs):
        return outputs
    raise ValueError(
        f"Unsupported ViT output type: {type(outputs)}. "
        "Expected tensor or object/dict with `last_hidden_state`."
    )


def _forward_patch_tokens(model: Any, x: torch.Tensor, number_of_patches: int) -> torch.Tensor:
    if number_of_patches <= 0:
        raise ValueError(f"number_of_patches must be > 0, got {number_of_patches}")

    raw_model = model.backbone if hasattr(model, "backbone") else model
    outputs = raw_model(x)
    hidden = _extract_hidden_state(outputs)
    if hidden.ndim != 3:
        raise ValueError(f"Expected hidden state shape (B, N, D). Got shape {tuple(hidden.shape)}")
    if hidden.shape[1] < number_of_patches:
        raise ValueError(
            f"number_of_patches ({number_of_patches}) exceeds token count ({hidden.shape[1]})"
        )
    return hidden[:, -number_of_patches:, :]


def _collect_probe_embeddings(
    model: Any,
    dataset: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    number_of_patches: Optional[int] = None,
) -> np.ndarray:
    """Collect final backbone representations for the optional open-world diagnostic."""
    loader = DataLoader(dataset, batch_size=int(batch_size), num_workers=int(num_workers), shuffle=False)
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            if number_of_patches is None:
                representation = model(x)
            else:
                representation = _forward_patch_tokens(model, x, number_of_patches).mean(dim=1)
            chunks.append(representation.detach().float().cpu().numpy())
    if not chunks:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _cosine_similarity_matrix(query_features: np.ndarray, database_features: np.ndarray) -> np.ndarray:
    query_features = np.asarray(query_features, dtype=np.float32)
    database_features = np.asarray(database_features, dtype=np.float32)
    if query_features.ndim != 2 or database_features.ndim != 2:
        raise ValueError("Embedding arrays must have shape (N, D)")
    query_norm = np.linalg.norm(query_features, axis=1, keepdims=True)
    database_norm = np.linalg.norm(database_features, axis=1, keepdims=True)
    query_normalized = query_features / np.maximum(query_norm, 1e-12)
    database_normalized = database_features / np.maximum(database_norm, 1e-12)
    return query_normalized @ database_normalized.T


def _embedding_retrieval_diagnostic(
    cfg: DictConfig,
    model: Any,
    dataset_query: Any,
    dataset_database: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    number_of_patches: Optional[int] = None,
) -> Tuple[Dict[str, Any], float]:
    started = time.perf_counter()
    query_features = _collect_probe_embeddings(
        model, dataset_query, device, batch_size, num_workers, number_of_patches
    )
    database_features = _collect_probe_embeddings(
        model, dataset_database, device, batch_size, num_workers, number_of_patches
    )
    similarity = _cosine_similarity_matrix(query_features, database_features)
    metrics = compute_metrics(
        dataset_query=dataset_query,
        dataset_database=dataset_database,
        similarity=similarity,
        top_k_values=[int(k) for k in cfg.benchmark.top_k],
        compute_map=bool(cfg.benchmark.compute_map),
        map_at_k=resolve_map_at_k(cfg, len(dataset_database)),
    )
    return {f"embedding_{key}": value for key, value in metrics.items()}, time.perf_counter() - started


def _sample_query_indices(total: int, num_examples: int, seed: int) -> Set[int]:
    if total <= 0 or num_examples <= 0:
        return set()
    count = min(total, num_examples)
    rng = random.Random(seed)
    return set(rng.sample(range(total), count))


def _save_attention_overlay_grid(
    out_path: Path,
    images: List[torch.Tensor],
    attention_maps: List[torch.Tensor],
    gt_indices: List[int],
    pred_indices: List[int],
    index_to_label: Dict[int, str],
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    average_queries: bool = True,
    alpha: float = 0.25,
) -> Optional[str]:
    if not images:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mean_np = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_np = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

    grid_size = int(math.ceil(math.sqrt(len(images))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3.0, grid_size * 3.5))
    axes = np.array(axes).reshape(-1)

    for i, (img_t, attn_t, gt_idx, pred_idx) in enumerate(zip(images, attention_maps, gt_indices, pred_indices)):
        if i >= len(axes):
            break
        ax = axes[i]
        img_np = img_t.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * std_np + mean_np, 0.0, 1.0)
        img_h, img_w = img_np.shape[:2]

        if attn_t.ndim != 2:
            ax.imshow(img_np)
            ax.axis("off")
            continue
        if average_queries:
            attn_1d = attn_t.mean(dim=0)
        else:
            attn_1d = attn_t[0]
        num_patches = int(attn_1d.numel())
        side = int(math.sqrt(num_patches))
        if side * side != num_patches:
            ax.imshow(img_np)
            ax.axis("off")
            continue

        attn_2d = attn_1d.reshape(side, side).unsqueeze(0).unsqueeze(0)
        attn_up = torch.nn.functional.interpolate(
            attn_2d,
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        attn_min = float(attn_up.min())
        attn_max = float(attn_up.max())
        if attn_max > attn_min:
            attn_up = (attn_up - attn_min) / (attn_max - attn_min)
        else:
            attn_up = np.zeros_like(attn_up)

        heatmap_rgb = plt.get_cmap("jet")(attn_up)[..., :3]
        blended = (1.0 - alpha) * img_np + alpha * heatmap_rgb
        ax.imshow(blended)
        ax.axis("off")

        gt_label = index_to_label.get(int(gt_idx), str(gt_idx))
        pred_label = index_to_label.get(int(pred_idx), str(pred_idx))
        title_color = "green" if int(gt_idx) == int(pred_idx) else "red"
        ax.set_title(f"GT: {gt_label}\nPred: {pred_label}", fontsize=9, color=title_color, pad=5)

    for j in range(len(images), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def resolve_configured_candidate_k(cfg: DictConfig) -> int:
    """Return the single public probe comparison budget."""
    raw_value = getattr(cfg.benchmark, "candidate_k", None)
    if raw_value is None:
        raise ValueError("benchmark.candidate_k is required; it controls Vismatch candidates, WildFusion refinement, and mAP@k evaluation.")
    try:
        candidate_k = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("benchmark.candidate_k must be an integer") from exc
    if candidate_k <= 0:
        raise ValueError("benchmark.candidate_k must be > 0")
    return candidate_k


def resolve_candidate_k(cfg: DictConfig, database_size: Optional[int] = None) -> int:
    """Return the configured budget, capped to the available database size."""
    candidate_k = resolve_configured_candidate_k(cfg)
    if database_size is None:
        return candidate_k
    database_size = int(database_size)
    if database_size < 0:
        raise ValueError("database_size must be >= 0")
    return min(candidate_k, database_size)


def resolve_map_at_k(cfg: DictConfig, database_size: Optional[int] = None) -> int:
    """Use the shared candidate budget as the retrieval evaluation cutoff."""
    return resolve_candidate_k(cfg, database_size=database_size)


def validate_evaluation_cutoffs(cfg: DictConfig, method: str) -> None:
    """Fail fast when a metric cutoff would read unscored score-matrix positions.

    Shortlist methods only score ``candidate_k`` database entries per query; every
    other position stays ``-inf`` and is ordered by original database index. Any
    cutoff beyond the shortlist would silently grade that tail, so the requested
    ``top_k`` and the derived mAP cutoff must both fit inside it.
    """
    candidate_k = resolve_configured_candidate_k(cfg)
    if method != "vismatch":
        return
    cutoffs = {f"benchmark.top_k={int(k)}": int(k) for k in cfg.benchmark.top_k}
    cutoffs[f"mAP_at_k (benchmark.candidate_k)={resolve_map_at_k(cfg)}"] = resolve_map_at_k(cfg)
    offending = sorted(name for name, value in cutoffs.items() if value > candidate_k)
    if offending:
        raise ValueError(
            f"Evaluation cutoffs exceed the Vismatch shortlist (candidate_k={candidate_k}): "
            f"{', '.join(offending)}. Metrics beyond the shortlist would rank unscored "
            "database entries by original index. Raise candidate_k or lower the cutoffs."
        )


def make_cache_key(cfg: DictConfig, method: str, split_name: str, dataset_sig: str, checkpoint_path: Optional[Path], model_weight_fingerprint: Optional[str] = None) -> str:
    checkpoint_tag = "pretrained"
    if checkpoint_path is not None:
        checkpoint_tag = f"{checkpoint_path}|{sha256_file(checkpoint_path)}"
    payload = {
        "method": method,
        "split": split_name,
        "dataset": dataset_sig,
        "model_type": cfg.model.type,
        "mode": cfg.model.mode,
        "no_background": bool(cfg.dataset.no_background),
        **build_dataset_cache_identity(cfg.dataset),
        "checkpoint": checkpoint_tag,
        "model_weight_fingerprint": model_weight_fingerprint or "unknown",
        "wildfusion_local_top_k": (
            int(getattr(cfg.benchmark.methods.wildfusion, "local_top_k", 512))
            if method == "wildfusion"
            else None
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def extract_deep_features_with_cache(
    dataset: WildlifeDataset,
    split_name: str,
    model: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cache: FeatureCache,
    cfg: DictConfig,
    method_name: str,
    checkpoint_path: Optional[Path],
) -> np.ndarray:
    dataset_sig = dataset_digest(dataset, cfg.dataset.label_col, Path(str(cfg.dataset.root)))
    weight_fingerprint = getattr(model, "_reid_weight_fingerprint", None)
    if weight_fingerprint is None:
        weight_fingerprint = model_fingerprint(model, revision=str(cfg.model.type))
        setattr(model, "_reid_weight_fingerprint", weight_fingerprint)
    cache_key = make_cache_key(cfg, method_name, split_name, dataset_sig, checkpoint_path, weight_fingerprint)

    def _compute():
        extractor = DeepFeatures(model, device=device, batch_size=batch_size, num_workers=num_workers)
        return extractor(dataset)

    return cache.get_or_compute(cache_key, _compute)


def make_dataset_view(cfg: DictConfig, dataset: Any, transform: Optional[Any]) -> BenchmarkDatasetView:
    return BenchmarkDatasetView(
        base_dataset=dataset,
        label_col=cfg.dataset.label_col,
        transform=transform,
        no_background=bool(cfg.dataset.no_background),
        mask_col=cfg.dataset.mask_col,
    )


class CachedDeepExtractor:
    def __init__(
        self,
        model: Any,
        device: torch.device,
        batch_size: int,
        num_workers: int,
        cache: FeatureCache,
        cfg: DictConfig,
        method_name: str,
        checkpoint_path: Optional[Path],
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache = cache
        self.cfg = cfg
        self.method_name = method_name
        self.checkpoint_path = checkpoint_path
        self.model_weight_fingerprint = model_fingerprint(model, revision=str(cfg.model.type))

    def __call__(self, dataset: WildlifeDataset) -> np.ndarray:
        dataset_sig = dataset_digest(dataset, self.cfg.dataset.label_col, Path(str(self.cfg.dataset.root)))
        cache_key = make_cache_key(self.cfg, self.method_name, "dynamic_split", dataset_sig, self.checkpoint_path, self.model_weight_fingerprint)

        def _compute():
            extractor = DeepFeatures(self.model, device=self.device, batch_size=self.batch_size, num_workers=self.num_workers)
            return extractor(dataset)

        features = self.cache.get_or_compute(cache_key, _compute)
        labels_string = get_labels_string(dataset, self.cfg.dataset.label_col)
        return FeatureContainer(features=features, labels_string=labels_string)


def run_linear_probe(
    cfg: DictConfig,
    model: Any,
    embedding_size: int,
    device: torch.device,
    dataset_query: Any,
    dataset_database: Any,
    run_dir: Path,
    wandb_run: Any = None,
    method_artifacts: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    timings: Dict[str, float] = {}
    method_metrics: Dict[str, Any] = {}
    t0 = time.perf_counter()

    open_set_policy, embedding_retrieval = resolve_classifier_evaluation(cfg)
    label_to_index = _build_label_mapping(dataset_database, cfg.dataset.label_col)
    db_labels_idx = dataset_database.df[cfg.dataset.label_col].astype(str).map(label_to_index).to_numpy(dtype=np.int64)
    query_labels_raw = dataset_query.df[cfg.dataset.label_col].astype(str).tolist()
    query_labels_idx = _label_indices(query_labels_raw, label_to_index)

    train_ds = EncodedLabelDataset(dataset_database, label_to_index)
    query_ds = EncodedLabelDataset(dataset_query, label_to_index, allow_unknown=open_set_policy != "closed")

    _set_trainable_params(model, cfg, method_key="linear_probe")
    lp_cfg = cfg.benchmark.methods.linear_probe
    class_order = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
    class_weights_np, class_weight_metadata = compute_identity_class_weights(
        dataset_database.df[cfg.dataset.label_col].astype(str).tolist(),
        class_order,
        weighting=str(getattr(lp_cfg, "class_weighting", "none")),
        normalize=bool(getattr(lp_cfg, "class_weight_normalize", True)),
        max_weight=float(getattr(lp_cfg, "class_weight_max", 5.0)),
    )
    class_weights = (
        torch.as_tensor(class_weights_np, dtype=torch.float32, device=device)
        if class_weights_np is not None
        else None
    )
    objective = SoftmaxLoss(
        num_classes=len(label_to_index),
        embedding_size=embedding_size,
        class_weights=class_weights,
    )
    objective.to(device)
    method_metrics.update(
        {
            "linear_probe_class_weighting": class_weight_metadata["mode"],
            "linear_probe_class_weight_formula": class_weight_metadata["formula"],
            "linear_probe_class_weight_normalize": class_weight_metadata["normalize_to_mean_one"],
            "linear_probe_class_weight_cap": class_weight_metadata["max_weight"],
            "linear_probe_class_weight_min": class_weight_metadata["min"],
            "linear_probe_class_weight_max": class_weight_metadata["max"],
            "linear_probe_class_weight_mean": class_weight_metadata["mean"],
            "linear_probe_class_weight_median": class_weight_metadata["median"],
            "linear_probe_class_weight_num_identities": class_weight_metadata["num_identities"],
            "linear_probe_class_weight_stats": class_weight_metadata,
        }
    )
    if method_artifacts is not None:
        method_artifacts["linear_probe_class_weighting"] = class_weight_metadata

    trainable_backbone = [p for p in model.parameters() if p.requires_grad]
    params = list(trainable_backbone) + list(objective.parameters())
    optimizer = _build_optimizer(params, lp_cfg, method_key="linear_probe")
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(lp_cfg.epochs),
        eta_min=float(lp_cfg.lr) * float(lp_cfg.eta_min_scale),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(lp_cfg.batch_size),
        num_workers=int(lp_cfg.num_workers),
        shuffle=True,
    )
    query_loader = DataLoader(
        query_ds,
        batch_size=int(lp_cfg.eval_batch_size),
        num_workers=int(lp_cfg.eval_num_workers),
        shuffle=False,
    )

    start_epoch = 0
    if lp_cfg.resume_checkpoint:
        resume_file = Path(str(lp_cfg.resume_checkpoint))
        ensure_file(resume_file, "Linear probe resume checkpoint")
        ckpt = torch.load(resume_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        objective.load_state_dict(ckpt["objective"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0))

    t_train = time.perf_counter()
    log_every = int(lp_cfg.log_every) if "log_every" in lp_cfg else 1
    for epoch in range(start_epoch, int(lp_cfg.epochs)):
        _set_probe_training_mode(model, objective, str(lp_cfg.train_mode))
        losses: List[float] = []
        train_probs_list: List[np.ndarray] = []
        train_targets_list: List[np.ndarray] = []
        optimizer.zero_grad(set_to_none=True)
        total_batches = len(train_loader)
        train_iter = tqdm(
            train_loader,
            desc=f"[linear_probe][train] epoch {epoch+1}/{int(lp_cfg.epochs)}",
            mininterval=1,
            ncols=120,
        )
        for i, batch in enumerate(train_iter):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            embeddings = model(x)
            loss = objective(embeddings, y)
            train_probs = _predict_class_probabilities(objective, embeddings).detach().cpu().numpy()
            train_probs_list.append(train_probs)
            train_targets_list.append(y.detach().cpu().numpy())
            (loss / float(accumulation_group_size(i, total_batches, int(lp_cfg.accumulation_steps)))).backward()
            if should_step_accumulated_gradients(i, total_batches, int(lp_cfg.accumulation_steps)):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            train_iter.set_postfix(loss=f"{losses[-1]:.4f}")

        scheduler.step()

        model.eval()
        objective.eval()
        probs_list: List[np.ndarray] = []
        val_losses: List[float] = []
        with torch.no_grad():
            val_iter = tqdm(
                query_loader,
                desc=f"[linear_probe][val] epoch {epoch+1}/{int(lp_cfg.epochs)}",
                mininterval=1,
                ncols=120,
            )
            for xq, yq in val_iter:
                xq = xq.to(device)
                yq = yq.to(device)
                emb = model(xq)
                probs = _predict_class_probabilities(objective, emb)
                known = yq >= 0
                if bool(known.any()):
                    val_loss = objective.unweighted_loss(emb[known], yq[known])
                    val_losses.append(float(val_loss.detach().cpu()))
                probs_list.append(probs.detach().cpu().numpy())
                displayed_loss = val_losses[-1] if val_losses else float("nan")
                val_iter.set_postfix(loss=f"{displayed_loss:.4f}")
        probs_query = np.concatenate(probs_list, axis=0)
        probs_train = np.concatenate(train_probs_list, axis=0)
        train_targets = np.concatenate(train_targets_list, axis=0)

        train_cls_metrics = _classification_topk_accuracy(probs_train, train_targets, [1, 5, 10])
        cls_metrics = _classifier_metrics(
            probs_query, query_labels_idx, query_labels_raw, label_to_index, open_set_policy
        )
        retrieval_metrics = _probe_retrieval_metrics(cfg, dataset_query, dataset_database, probs_query, db_labels_idx, query_labels_idx)

        if wandb_run is not None:
            lr = float(optimizer.param_groups[0].get("lr", 0.0))
            wandb_run.log(
                {
                    "linear_probe/epoch": epoch + 1,
                    "linear_probe/train_loss": float(np.mean(losses)) if losses else float("nan"),
                    "linear_probe/val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
                    "linear_probe/lr": lr,
                    **{f"linear_probe/train_{k}": v for k, v in train_cls_metrics.items()},
                    **{f"linear_probe/{k}": v for k, v in cls_metrics.items()},
                    **{f"linear_probe/{k}": v for k, v in retrieval_metrics.items()},
                },
                step=epoch + 1,
            )

        if (epoch + 1) % log_every == 0:
            print(
                f"[linear_probe] epoch {epoch+1}/{int(lp_cfg.epochs)} "
                f"train_loss={float(np.mean(losses)) if losses else float('nan'):.6f} "
                f"train_top1={train_cls_metrics.get('classification_top_1', float('nan')):.4f} "
                f"train_top5={train_cls_metrics.get('classification_top_5', float('nan')):.4f} "
                f"train_top10={train_cls_metrics.get('classification_top_10', float('nan')):.4f} "
                f"val_loss={float(np.mean(val_losses)) if val_losses else float('nan'):.6f} "
                f"val_top1={cls_metrics.get('classification_top_1', float('nan')):.4f} "
                f"val_top5={cls_metrics.get('classification_top_5', float('nan')):.4f} "
                f"val_top10={cls_metrics.get('classification_top_10', float('nan')):.4f}"
            )

        if bool(lp_cfg.save_checkpoint) and ((epoch + 1) % int(lp_cfg.save_every) == 0):
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = run_dir / f"linear_probe_epoch_{epoch+1}.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "objective": objective.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "label_to_index": label_to_index,
                },
                ckpt_path,
            )

    timings["linear_probe_train_sec"] = time.perf_counter() - t_train

    model.eval()
    objective.eval()
    t_eval = time.perf_counter()
    probs_list = []
    with torch.no_grad():
        eval_iter = tqdm(
            query_loader,
            desc="[linear_probe][final_eval]",
            mininterval=1,
            ncols=120,
        )
        for xq, _ in eval_iter:
            xq = xq.to(device)
            emb = model(xq)
            probs = _predict_class_probabilities(objective, emb)
            probs_list.append(probs.detach().cpu().numpy())
    probs_query = np.concatenate(probs_list, axis=0)
    timings["linear_probe_eval_sec"] = time.perf_counter() - t_eval

    similarity = _similarity_from_class_probs(probs_query, db_labels_idx)
    method_metrics.update(
        _classifier_metrics(probs_query, query_labels_idx, query_labels_raw, label_to_index, open_set_policy)
    )
    method_metrics["classification_embedding_retrieval_enabled"] = bool(embedding_retrieval)
    if embedding_retrieval:
        embedding_metrics, embedding_sec = _embedding_retrieval_diagnostic(
            cfg=cfg,
            model=model,
            dataset_query=query_ds,
            dataset_database=train_ds,
            device=device,
            batch_size=int(lp_cfg.eval_batch_size),
            num_workers=int(lp_cfg.eval_num_workers),
        )
        method_metrics.update(embedding_metrics)
        timings["linear_probe_embedding_retrieval_sec"] = float(embedding_sec)
    method_metrics.update(_probe_retrieval_metrics(cfg, dataset_query, dataset_database, probs_query, db_labels_idx, query_labels_idx))

    if bool(lp_cfg.save_checkpoint):
        run_dir.mkdir(parents=True, exist_ok=True)
        final_path = run_dir / str(lp_cfg.final_checkpoint_name)
        torch.save(
            {
                "model": model.state_dict(),
                "objective": objective.state_dict(),
                "label_to_index": label_to_index,
            },
            final_path,
        )

    timings["total_method_sec"] = time.perf_counter() - t0
    return similarity, timings, method_metrics


def run_efficient_probe(
    cfg: DictConfig,
    model: Any,
    embedding_size: int,
    number_of_patches: int,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    device: torch.device,
    dataset_query: Any,
    dataset_database: Any,
    run_dir: Path,
    wandb_run: Any = None,
    method_artifacts: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    timings: Dict[str, float] = {}
    method_metrics: Dict[str, Any] = {}
    t0 = time.perf_counter()

    open_set_policy, embedding_retrieval = resolve_classifier_evaluation(cfg)
    label_to_index = _build_label_mapping(dataset_database, cfg.dataset.label_col)
    db_labels_idx = dataset_database.df[cfg.dataset.label_col].astype(str).map(label_to_index).to_numpy(dtype=np.int64)
    query_labels_raw = dataset_query.df[cfg.dataset.label_col].astype(str).tolist()
    query_labels_idx = _label_indices(query_labels_raw, label_to_index)

    train_ds = EncodedLabelDataset(dataset_database, label_to_index)
    query_ds = EncodedLabelDataset(dataset_query, label_to_index, allow_unknown=open_set_policy != "closed")

    ep_cfg = cfg.benchmark.methods.efficient_probe
    _set_trainable_params(model, cfg, method_key="efficient_probe")
    class_order = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
    class_weights_np, class_weight_metadata = compute_identity_class_weights(
        dataset_database.df[cfg.dataset.label_col].astype(str).tolist(),
        class_order,
        weighting=str(getattr(ep_cfg, "class_weighting", "none")),
        normalize=bool(getattr(ep_cfg, "class_weight_normalize", True)),
        max_weight=float(getattr(ep_cfg, "class_weight_max", 5.0)),
    )
    class_weights = (
        torch.as_tensor(class_weights_np, dtype=torch.float32, device=device)
        if class_weights_np is not None
        else None
    )
    objective = SoftmaxLossEP(
        num_classes=len(label_to_index),
        embedding_size=embedding_size,
        dropout_rate=float(ep_cfg.dropout_rate),
        num_queries=int(ep_cfg.num_queries),
        d_out=int(ep_cfg.d_out),
        class_weights=class_weights,
    )
    objective.to(device)
    method_metrics.update(
        {
            "efficient_probe_class_weighting": class_weight_metadata["mode"],
            "efficient_probe_class_weight_formula": class_weight_metadata["formula"],
            "efficient_probe_class_weight_normalize": class_weight_metadata["normalize_to_mean_one"],
            "efficient_probe_class_weight_cap": class_weight_metadata["max_weight"],
            "efficient_probe_class_weight_min": class_weight_metadata["min"],
            "efficient_probe_class_weight_max": class_weight_metadata["max"],
            "efficient_probe_class_weight_mean": class_weight_metadata["mean"],
            "efficient_probe_class_weight_median": class_weight_metadata["median"],
            "efficient_probe_class_weight_num_identities": class_weight_metadata["num_identities"],
            "efficient_probe_class_weight_stats": class_weight_metadata,
        }
    )
    if method_artifacts is not None:
        method_artifacts["efficient_probe_class_weighting"] = class_weight_metadata

    trainable_backbone = [p for p in model.parameters() if p.requires_grad]
    params = list(trainable_backbone) + list(objective.parameters())
    optimizer = _build_optimizer(params, ep_cfg, method_key="efficient_probe")
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(ep_cfg.epochs),
        eta_min=float(ep_cfg.lr) * float(ep_cfg.eta_min_scale),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(ep_cfg.batch_size),
        num_workers=int(ep_cfg.num_workers),
        shuffle=True,
    )
    query_loader = DataLoader(
        query_ds,
        batch_size=int(ep_cfg.eval_batch_size),
        num_workers=int(ep_cfg.eval_num_workers),
        shuffle=False,
    )

    start_epoch = 0
    if ep_cfg.resume_checkpoint:
        resume_file = Path(str(ep_cfg.resume_checkpoint))
        ensure_file(resume_file, "Efficient probe resume checkpoint")
        ckpt = torch.load(resume_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        objective.load_state_dict(ckpt["objective"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0))

    t_train = time.perf_counter()
    log_every = int(ep_cfg.log_every) if "log_every" in ep_cfg else 1
    for epoch in range(start_epoch, int(ep_cfg.epochs)):
        _set_probe_training_mode(model, objective, str(ep_cfg.train_mode))
        losses: List[float] = []
        train_probs_list: List[np.ndarray] = []
        train_targets_list: List[np.ndarray] = []
        optimizer.zero_grad(set_to_none=True)
        total_batches = len(train_loader)
        train_iter = tqdm(
            train_loader,
            desc=f"[efficient_probe][train] epoch {epoch+1}/{int(ep_cfg.epochs)}",
            mininterval=1,
            ncols=120,
        )
        for i, batch in enumerate(train_iter):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            patch_tokens = _forward_patch_tokens(model, x, number_of_patches=number_of_patches)
            loss = objective(patch_tokens, y)
            train_probs = _predict_class_probabilities(objective, patch_tokens).detach().cpu().numpy()
            train_probs_list.append(train_probs)
            train_targets_list.append(y.detach().cpu().numpy())
            (loss / float(accumulation_group_size(i, total_batches, int(ep_cfg.accumulation_steps)))).backward()
            if should_step_accumulated_gradients(i, total_batches, int(ep_cfg.accumulation_steps)):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            train_iter.set_postfix(loss=f"{losses[-1]:.4f}")

        scheduler.step()

        model.eval()
        objective.eval()
        probs_list: List[np.ndarray] = []
        val_losses: List[float] = []
        with torch.no_grad():
            val_iter = tqdm(
                query_loader,
                desc=f"[efficient_probe][val] epoch {epoch+1}/{int(ep_cfg.epochs)}",
                mininterval=1,
                ncols=120,
            )
            for xq, yq in val_iter:
                xq = xq.to(device)
                yq = yq.to(device)
                patch_tokens = _forward_patch_tokens(model, xq, number_of_patches=number_of_patches)
                probs = _predict_class_probabilities(objective, patch_tokens)
                known = yq >= 0
                if bool(known.any()):
                    val_loss = objective.unweighted_loss(patch_tokens[known], yq[known])
                    val_losses.append(float(val_loss.detach().cpu()))
                probs_list.append(probs.detach().cpu().numpy())
                displayed_loss = val_losses[-1] if val_losses else float("nan")
                val_iter.set_postfix(loss=f"{displayed_loss:.4f}")
        probs_query = np.concatenate(probs_list, axis=0)
        probs_train = np.concatenate(train_probs_list, axis=0)
        train_targets = np.concatenate(train_targets_list, axis=0)

        train_cls_metrics = _classification_topk_accuracy(probs_train, train_targets, [1, 5, 10])
        cls_metrics = _classifier_metrics(
            probs_query, query_labels_idx, query_labels_raw, label_to_index, open_set_policy
        )
        retrieval_metrics = _probe_retrieval_metrics(cfg, dataset_query, dataset_database, probs_query, db_labels_idx, query_labels_idx)

        if wandb_run is not None:
            lr = float(optimizer.param_groups[0].get("lr", 0.0))
            wandb_run.log(
                {
                    "efficient_probe/epoch": epoch + 1,
                    "efficient_probe/train_loss": float(np.mean(losses)) if losses else float("nan"),
                    "efficient_probe/val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
                    "efficient_probe/lr": lr,
                    **{f"efficient_probe/train_{k}": v for k, v in train_cls_metrics.items()},
                    **{f"efficient_probe/{k}": v for k, v in cls_metrics.items()},
                    **{f"efficient_probe/{k}": v for k, v in retrieval_metrics.items()},
                },
                step=epoch + 1,
            )

        if (epoch + 1) % log_every == 0:
            print(
                f"[efficient_probe] epoch {epoch+1}/{int(ep_cfg.epochs)} "
                f"train_loss={float(np.mean(losses)) if losses else float('nan'):.6f} "
                f"train_top1={train_cls_metrics.get('classification_top_1', float('nan')):.4f} "
                f"train_top5={train_cls_metrics.get('classification_top_5', float('nan')):.4f} "
                f"train_top10={train_cls_metrics.get('classification_top_10', float('nan')):.4f} "
                f"val_loss={float(np.mean(val_losses)) if val_losses else float('nan'):.6f} "
                f"val_top1={cls_metrics.get('classification_top_1', float('nan')):.4f} "
                f"val_top5={cls_metrics.get('classification_top_5', float('nan')):.4f} "
                f"val_top10={cls_metrics.get('classification_top_10', float('nan')):.4f}"
            )

        if bool(ep_cfg.save_checkpoint) and ((epoch + 1) % int(ep_cfg.save_every) == 0):
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = run_dir / f"efficient_probe_epoch_{epoch+1}.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "objective": objective.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "label_to_index": label_to_index,
                },
                ckpt_path,
            )

    timings["efficient_probe_train_sec"] = time.perf_counter() - t_train

    model.eval()
    objective.eval()
    t_eval = time.perf_counter()
    probs_list = []
    collect_attention = bool(cfg.visualization.enabled)
    attention_indices = _sample_query_indices(
        total=len(query_ds),
        num_examples=int(getattr(cfg.visualization, "attention_num_examples", cfg.visualization.num_examples)),
        seed=int(cfg.benchmark.seed),
    )
    sampled_images: List[torch.Tensor] = []
    sampled_attention_maps: List[torch.Tensor] = []
    sampled_gt: List[int] = []
    sampled_pred: List[int] = []
    seen = 0
    with torch.no_grad():
        eval_iter = tqdm(
            query_loader,
            desc="[efficient_probe][final_eval]",
            mininterval=1,
            ncols=120,
        )
        for xq, yq in eval_iter:
            xq_cpu = xq.detach().cpu()
            xq = xq.to(device)
            patch_tokens = _forward_patch_tokens(model, xq, number_of_patches=number_of_patches)
            probs = _predict_class_probabilities(objective, patch_tokens)
            probs_list.append(probs.detach().cpu().numpy())
            if collect_attention and attention_indices and hasattr(objective, "attention_map"):
                attn = objective.attention_map.detach().cpu()
                pred = torch.argmax(probs.detach().cpu(), dim=1)
                yq_cpu = yq.detach().cpu()
                batch_size = xq_cpu.shape[0]
                for local_idx in range(batch_size):
                    global_idx = seen + local_idx
                    if global_idx not in attention_indices:
                        continue
                    sampled_images.append(xq_cpu[local_idx])
                    sampled_attention_maps.append(attn[local_idx])
                    sampled_gt.append(int(yq_cpu[local_idx].item()))
                    sampled_pred.append(int(pred[local_idx].item()))
            seen += xq_cpu.shape[0]
    probs_query = np.concatenate(probs_list, axis=0)
    timings["efficient_probe_eval_sec"] = time.perf_counter() - t_eval

    similarity = _similarity_from_class_probs(probs_query, db_labels_idx)
    method_metrics.update(
        _classifier_metrics(probs_query, query_labels_idx, query_labels_raw, label_to_index, open_set_policy)
    )
    method_metrics["classification_embedding_retrieval_enabled"] = bool(embedding_retrieval)
    if embedding_retrieval:
        embedding_metrics, embedding_sec = _embedding_retrieval_diagnostic(
            cfg=cfg,
            model=model,
            dataset_query=query_ds,
            dataset_database=train_ds,
            device=device,
            batch_size=int(ep_cfg.eval_batch_size),
            num_workers=int(ep_cfg.eval_num_workers),
            number_of_patches=number_of_patches,
        )
        method_metrics.update(embedding_metrics)
        timings["efficient_probe_embedding_retrieval_sec"] = float(embedding_sec)
    method_metrics.update(_probe_retrieval_metrics(cfg, dataset_query, dataset_database, probs_query, db_labels_idx, query_labels_idx))

    if bool(ep_cfg.save_checkpoint):
        run_dir.mkdir(parents=True, exist_ok=True)
        final_path = run_dir / str(ep_cfg.final_checkpoint_name)
        torch.save(
            {
                "model": model.state_dict(),
                "objective": objective.state_dict(),
                "label_to_index": label_to_index,
            },
            final_path,
        )

    if collect_attention and sampled_images:
        vis_dir = Path(run_dir) / "visualizations"
        attention_path = _save_attention_overlay_grid(
            out_path=vis_dir / "efficient_probe_attention_map.png",
            images=sampled_images,
            attention_maps=sampled_attention_maps,
            gt_indices=sampled_gt,
            pred_indices=sampled_pred,
            index_to_label={v: k for k, v in label_to_index.items()},
            mean=mean,
            std=std,
            average_queries=bool(getattr(cfg.visualization, "attention_average_queries", True)),
        )
        if method_artifacts is not None and attention_path is not None:
            method_artifacts["attention_map_path"] = attention_path

    timings["total_method_sec"] = time.perf_counter() - t0
    return similarity, timings, method_metrics


def run_method(
    cfg: DictConfig,
    method: str,
    model: Any,
    embedding_size: int,
    arch: str,
    number_of_patches: Optional[int],
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    device: torch.device,
    dataset_query: WildlifeDataset,
    dataset_database: WildlifeDataset,
    dataset_calibration: WildlifeDataset,
    transform_model: T.Compose,
    transform_aliked: T.Compose,
    checkpoint_path: Optional[Path],
    cache: FeatureCache,
    run_dir: Path,
    wandb_run: Any = None,
    method_artifacts: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    if method == "rdd":
        raise ValueError(
            "The public probe method 'rdd' was removed. Use method='vismatch' with "
            "benchmark.methods.vismatch.matcher='rdd-lightglue'."
        )
    timings: Dict[str, float] = {}
    method_metrics: Dict[str, Any] = {}
    t0 = time.perf_counter()

    def _call_similarity(matcher_obj, query_ds, database_ds, b_value):
        try:
            return matcher_obj(query_ds, database_ds, B=b_value)
        except TypeError as exc:
            if "unexpected keyword argument 'B'" not in str(exc):
                raise
            return matcher_obj(query_ds, database_ds)

    if method == "cosine":
        t_extract = time.perf_counter()
        features_database = extract_deep_features_with_cache(
            dataset_database,
            "database",
            model,
            device,
            cfg.model.batch_size,
            cfg.model.num_workers,
            cache,
            cfg,
            method,
            checkpoint_path,
        )
        features_query = extract_deep_features_with_cache(
            dataset_query,
            "query",
            model,
            device,
            cfg.model.batch_size,
            cfg.model.num_workers,
            cache,
            cfg,
            method,
            checkpoint_path,
        )
        timings["feature_extraction_sec"] = time.perf_counter() - t_extract
        t_sim = time.perf_counter()
        similarity = CosineSimilarity()(
            FeatureContainer(features=features_query, labels_string=get_labels_string(dataset_query, cfg.dataset.label_col)),
            FeatureContainer(features=features_database, labels_string=get_labels_string(dataset_database, cfg.dataset.label_col)),
        )
        timings["similarity_sec"] = time.perf_counter() - t_sim

    elif method == "wildfusion":
        settings = cfg.benchmark.methods.wildfusion
        t_setup = time.perf_counter()
        matcher_aliked = SimilarityPipeline(
            matcher=MatchLightGlue(features="aliked", device=device, batch_size=settings.local_batch_size),
            extractor=AlikedExtractor(
                max_num_keypoints=int(getattr(settings, "local_top_k", 512)),
                force_num_keypoints=True,
            ),
            transform=transform_aliked,
            calibration=IsotonicCalibration(),
        )
        matcher_mega = SimilarityPipeline(
            matcher=CosineSimilarity(),
            extractor=CachedDeepExtractor(
                model=model,
                device=device,
                batch_size=settings.deep_batch_size,
                num_workers=settings.deep_num_workers,
                cache=cache,
                cfg=cfg,
                method_name=method,
                checkpoint_path=checkpoint_path,
            ),
            transform=transform_model,
            calibration=IsotonicCalibration(),
        )
        wildfusion = WildFusion(calibrated_pipelines=[matcher_aliked, matcher_mega], priority_pipeline=matcher_mega)
        timings["model_setup_sec"] = time.perf_counter() - t_setup
        calibration_cfg = getattr(cfg.benchmark, "calibration", {})
        t_calibration = time.perf_counter()
        calibration_info = fit_wildfusion_calibration(wildfusion, dataset_calibration, dataset_calibration, exclude_self_pairs=bool(getattr(calibration_cfg, "exclude_self_pairs", True)), official_same_set=bool(getattr(calibration_cfg, "official_same_set", False)))
        timings["calibration_sec"] = time.perf_counter() - t_calibration
        if method_artifacts is not None: method_artifacts["wildfusion_calibration"] = calibration_info
        _instrument_local_pipeline(matcher_aliked, timings)
        t_sim = time.perf_counter()
        candidate_k = resolve_candidate_k(cfg, len(dataset_database))
        similarity = _call_similarity(wildfusion, dataset_query, dataset_database, candidate_k)
        timings["similarity_sec"] = time.perf_counter() - t_sim
        # WildFusion refines only its top-B pairs with the local pipeline and keeps
        # priority-pipeline scores elsewhere. Record the shared budget so the
        # refinement budget is auditable alongside Vismatch and mAP@k.
        timings["benchmark_candidate_k"] = float(candidate_k)
        method_metrics["wildfusion_B"] = float(candidate_k)
        method_metrics["wildfusion_refined_fraction"] = (
            float(candidate_k / len(dataset_database)) if len(dataset_database) else float("nan")
        )

    elif method == "local_lightglue":
        settings = cfg.benchmark.methods.local_lightglue
        t_setup = time.perf_counter()
        matcher_local = SimilarityPipeline(
            matcher=MatchLightGlue(features="aliked", device=device, batch_size=settings.local_batch_size),
            extractor=AlikedExtractor(),
            transform=transform_aliked,
            calibration=IsotonicCalibration(),
        )
        timings["model_setup_sec"] = time.perf_counter() - t_setup
        calibration_cfg = getattr(cfg.benchmark, "calibration", {})
        t_calibration = time.perf_counter()
        calibration_info = fit_pipeline_calibration(matcher_local, dataset_calibration, dataset_calibration, exclude_self_pairs=bool(getattr(calibration_cfg, "exclude_self_pairs", True)) and not bool(getattr(calibration_cfg, "official_same_set", False)))
        timings["calibration_sec"] = time.perf_counter() - t_calibration
        if method_artifacts is not None: method_artifacts["local_calibration"] = calibration_info
        _instrument_local_pipeline(matcher_local, timings)
        t_sim = time.perf_counter()
        candidate_k = resolve_candidate_k(cfg, len(dataset_database))
        similarity = _call_similarity(matcher_local, dataset_query, dataset_database, candidate_k)
        timings["similarity_sec"] = time.perf_counter() - t_sim
        timings["benchmark_candidate_k"] = float(candidate_k)
        method_metrics["local_lightglue_B"] = float(candidate_k)

    elif method == "linear_probe":
        similarity, lp_timings, method_metrics = run_linear_probe(
            cfg=cfg,
            model=model,
            embedding_size=embedding_size,
            device=device,
            dataset_query=dataset_query,
            dataset_database=dataset_database,
            run_dir=run_dir,
            wandb_run=wandb_run,
            method_artifacts=method_artifacts,
        )
        timings.update(lp_timings)
    elif method == "efficient_probe":
        if arch != "vit":
            raise ValueError("efficient_probe is supported only for ViT backbones")
        if number_of_patches is None or int(number_of_patches) <= 0:
            raise ValueError(f"Invalid number_of_patches for efficient_probe: {number_of_patches}")
        similarity, ep_timings, method_metrics = run_efficient_probe(
            cfg=cfg,
            model=model,
            embedding_size=embedding_size,
            number_of_patches=int(number_of_patches),
            mean=mean,
            std=std,
            device=device,
            dataset_query=dataset_query,
            dataset_database=dataset_database,
            run_dir=run_dir,
            wandb_run=wandb_run,
            method_artifacts=method_artifacts,
        )
        timings.update(ep_timings)
    elif method == "vismatch":
        vismatch_cfg = cfg.benchmark.methods.vismatch
        stage_a_method = str(vismatch_cfg.stage_a_method)
        if stage_a_method == "vismatch":
            raise ValueError("benchmark.methods.vismatch.stage_a_method cannot be 'vismatch'")

        configured_candidate_k = resolve_configured_candidate_k(cfg)
        candidate_k = resolve_candidate_k(cfg, len(dataset_database))

        t_stage_a = time.perf_counter()
        stage_similarity, stage_timings, stage_metrics = run_method(
            cfg=cfg,
            method=stage_a_method,
            model=model,
            embedding_size=embedding_size,
            arch=arch,
            number_of_patches=number_of_patches,
            mean=mean,
            std=std,
            device=device,
            dataset_query=dataset_query,
            dataset_database=dataset_database,
            dataset_calibration=dataset_calibration,
            transform_model=transform_model,
            transform_aliked=transform_aliked,
            checkpoint_path=checkpoint_path,
            cache=cache,
            run_dir=run_dir,
            wandb_run=wandb_run,
            method_artifacts=None,
        )
        stage_a_sec = time.perf_counter() - t_stage_a
        candidate_indices = stable_rank_indices(stage_similarity)[:, :candidate_k]
        timings["vismatch_stage_a_sec"] = float(stage_a_sec)
        timings["benchmark_candidate_k"] = float(configured_candidate_k)
        timings["vismatch_candidate_k"] = float(candidate_k)
        for k, v in stage_timings.items():
            if k == "total_method_sec":
                continue
            timings[f"stage_a_{stage_a_method}_{k}"] = float(v)

        similarity, vismatch_timings, method_metrics = run_vismatch_benchmark(
            cfg=cfg,
            dataset_query=dataset_query,
            dataset_database=dataset_database,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            mean=mean,
            std=std,
            candidate_indices=candidate_indices,
            stage_similarity=stage_similarity,
            method_artifacts=method_artifacts,
        )
        timings.update(vismatch_timings)
        # `run_vismatch_benchmark` returns a fresh metrics dict, so Stage-A metrics are
        # merged after it. Prefixing mirrors the Stage-A timing keys above and keeps the
        # shortlist provenance of the Stage-A method, such as `wildfusion_B`, on the run.
        # Values are not coerced: some metrics are string diagnostics.
        for key, value in stage_metrics.items():
            method_metrics[f"stage_a_{stage_a_method}_{key}"] = value

    else:
        raise ValueError(
            f"Unsupported method '{method}'. Supported: cosine, wildfusion, local_lightglue, linear_probe, efficient_probe, vismatch"
        )

    timings["total_method_sec"] = time.perf_counter() - t0
    timings["feature_cache_lookup_sec"] = float(timings.get("feature_cache_lookup_sec", 0.0)) + float(cache.lookup_sec)
    timings["feature_cache_hits"] = float(timings.get("feature_cache_hits", 0.0)) + float(cache.hits)
    timings["feature_cache_misses"] = float(timings.get("feature_cache_misses", 0.0)) + float(cache.misses)
    timings["feature_extraction_compute_sec"] = float(timings.get("feature_extraction_compute_sec", 0.0)) + float(cache.compute_sec)
    timings.setdefault(
        "feature_extraction_sec",
        timings["feature_extraction_compute_sec"] + timings["feature_cache_lookup_sec"],
    )
    timings.setdefault("model_setup_sec", 0.0)
    timings.setdefault("calibration_sec", 0.0)
    if method == "cosine":
        timings["method_compute_runtime_sec"] = float(timings.get("similarity_sec", 0.0))
    else:
        timings["method_compute_runtime_sec"] = float(timings["total_method_sec"])
    set_primary_compute_runtime(method, timings)
    return np.asarray(similarity), timings, method_metrics


def visualize_predictions(
    cfg: DictConfig,
    similarity: np.ndarray,
    dataset_query_display: WildlifeDataset,
    dataset_database_display: WildlifeDataset,
    run_dir: Path,
) -> List[str]:
    vis_cfg = cfg.visualization
    out_dir = Path(run_dir) / "visualizations" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked_idx = stable_rank_indices(similarity)
    rng = np.random.default_rng(cfg.benchmark.seed)
    num_queries = len(dataset_query_display)
    num_examples = min(vis_cfg.num_examples, num_queries)
    query_indices = rng.choice(np.arange(num_queries), num_examples, replace=False)

    saved_files: List[str] = []
    title_wrap = int(getattr(vis_cfg, "title_wrap_chars", 28))

    def _wrap_title(text: Any) -> str:
        return textwrap.fill(str(text), width=max(8, title_wrap), break_long_words=True, break_on_hyphens=True)

    for query_idx in query_indices:
        database_idx = ranked_idx[query_idx, : vis_cfg.top_k]
        scores = similarity[query_idx, database_idx]
        fig, ax = plt.subplots(1, 1 + vis_cfg.top_k, figsize=((1 + vis_cfg.top_k) * 3, 3))

        query_data = dataset_query_display[query_idx]
        ax[0].imshow(query_data[0])
        ax[0].set_title(_wrap_title(query_data[1]), fontsize=9)
        ax[0].axis("off")

        for i, db_idx in enumerate(database_idx):
            database_data = dataset_database_display[db_idx]
            ax[i + 1].imshow(database_data[0])
            ax[i + 1].set_title(f"{_wrap_title(database_data[1])}\nscore: {scores[i]:.3f}", fontsize=9)
            ax[i + 1].axis("off")

        out_path = out_dir / f"query_{int(query_idx):06d}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        saved_files.append(str(out_path))

    return saved_files


def run_probe(cfg: DictConfig) -> None:
    run_started = datetime.utcnow()
    context = build_run_context(cfg, "probe", run_started=run_started)
    context.write_config(cfg)
    context.write_manifest(
        {
            "dataset": str(cfg.dataset.name),
            "animal": str(cfg.dataset.animal),
            "split_protocol": str(cfg.dataset.split_col),
            "model": str(cfg.model.type),
            "method": str(cfg.benchmark.method),
            "variant": str(cfg.benchmark.methods.vismatch.matcher)
            if str(cfg.benchmark.method) == "vismatch"
            else "default",
        },
        status="running",
    )
    try:
        # Safety checks, dataset digests, and Vismatch cache keys each hash every image.
        # The dataset is read-only for the duration of a run, so hash each file once.
        with file_digest_cache():
            return _run_probe(cfg, context)
    except Exception as exc:
        context.write_manifest(
            {
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            },
            status="failed",
        )
        index_path = Path(str(getattr(cfg.reporting, "index_path", "reports/runs.csv")))
        upsert_run_index(
            index_path,
            run_index_row(
                context,
                {
                    "status": "failed",
                    "dataset": str(cfg.dataset.name),
                    "animal": str(cfg.dataset.animal),
                    "split_protocol": str(cfg.dataset.split_col),
                    "model": str(cfg.model.type),
                    "method": str(cfg.benchmark.method),
                    "variant": context.run_dir.parts[-2],
                },
            ),
        )
        raise


def _run_probe(cfg: DictConfig, context: Any) -> None:
    run_t0 = time.perf_counter()
    set_reproducible(int(cfg.benchmark.seed), bool(cfg.benchmark.deterministic))
    method = str(cfg.benchmark.method)
    if method == "rdd":
        raise ValueError(
            "The public probe method 'rdd' was removed. Use method='vismatch' with "
            "benchmark.methods.vismatch.matcher='rdd-lightglue'."
        )
    run_id = context.run_id
    run_dir = context.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    validate_evaluation_cutoffs(cfg, method)

    dataset, dataset_database_raw, dataset_query_raw = load_dataset_splits(cfg)
    classifier_methods = {"linear_probe", "efficient_probe"}
    classifier_active = method in classifier_methods
    if method == "vismatch":
        classifier_active = str(cfg.benchmark.methods.vismatch.stage_a_method) in classifier_methods
    classifier_policy = "open"
    if classifier_active:
        classifier_policy, _ = resolve_classifier_evaluation(cfg)
        safety_enabled = bool(getattr(cfg, "safety_checks", {}).get("enabled", True))
        validate_classifier_open_set_labels(
            dataset_database_raw,
            dataset_query_raw,
            str(cfg.dataset.label_col),
            classifier_policy,
            emit_warning=not safety_enabled,
        )
    if bool(getattr(cfg, "safety_checks", {}).get("enabled", True)):
        require_closed_set = classifier_active and classifier_policy == "closed"
        run_split_safety_checks(
            df_a=dataset_database_raw.df,
            df_b=dataset_query_raw.df,
            split_a_name="database",
            split_b_name="query",
            root=Path(cfg.dataset.root),
            label_col=str(cfg.dataset.label_col),
            run_dir=run_dir,
            fail_on_overlap=True,
            require_b_labels_in_a=require_closed_set,
        )

    use_backbone = method != "vismatch"
    backbone_setup_sec = 0.0
    if method == "vismatch":
        stage_a_method = str(cfg.benchmark.methods.vismatch.stage_a_method)
        stage_a_needs_backbone = stage_a_method in {"cosine", "wildfusion", "linear_probe", "efficient_probe"}
        use_backbone = stage_a_needs_backbone
    if use_backbone:
        device = choose_device(cfg.model.device)
        t_backbone_setup = time.perf_counter()
        model, embedding_size, mean, std, img_size, arch, number_of_patches, checkpoint_path = load_backbone(cfg)
        model.to(device)
        model.eval()
        backbone_setup_sec = time.perf_counter() - t_backbone_setup
    else:
        device = choose_device(str(cfg.benchmark.methods.vismatch.device))
        model = None
        embedding_size = 0
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        img_size = 224
        arch = "vismatch"
        number_of_patches = None
        checkpoint_path = None

    transform_display, transform_model, transform_aliked = build_transforms(mean, std, img_size)
    dataset_calibration_raw = get_calibration_dataset(
        dataset_database=dataset_database_raw,
        size=int(cfg.dataset.calibration_size),
        root=cfg.dataset.root,
        label_col=cfg.dataset.label_col,
        source_dataset=dataset,
        split_col=str(cfg.dataset.split_col),
        split_value=getattr(cfg.benchmark.calibration, "split_value", None),
    )

    if method in {"cosine", "linear_probe", "efficient_probe"}:
        dataset_database = make_dataset_view(cfg, dataset_database_raw, transform=transform_model)
        dataset_query = make_dataset_view(cfg, dataset_query_raw, transform=transform_model)
        dataset_calibration = make_dataset_view(cfg, dataset_calibration_raw, transform=transform_model)
    elif method == "vismatch":
        stage_a_method = str(cfg.benchmark.methods.vismatch.stage_a_method)
        if stage_a_method in {"cosine", "wildfusion", "linear_probe", "efficient_probe"}:
            dataset_database = make_dataset_view(cfg, dataset_database_raw, transform=transform_model)
            dataset_query = make_dataset_view(cfg, dataset_query_raw, transform=transform_model)
            dataset_calibration = make_dataset_view(cfg, dataset_calibration_raw, transform=transform_model)
        else:
            dataset_database = make_dataset_view(cfg, dataset_database_raw, transform=None)
            dataset_query = make_dataset_view(cfg, dataset_query_raw, transform=None)
            dataset_calibration = make_dataset_view(cfg, dataset_calibration_raw, transform=None)
    else:
        dataset_database = make_dataset_view(cfg, dataset_database_raw, transform=None)
        dataset_query = make_dataset_view(cfg, dataset_query_raw, transform=None)
        dataset_calibration = make_dataset_view(cfg, dataset_calibration_raw, transform=None)

    cache = FeatureCache(
        enabled=bool(cfg.benchmark.cache.enabled),
        cache_dir=Path(cfg.benchmark.cache.dir),
        fmt=str(cfg.benchmark.cache.format),
    )

    print(f"Running method={method} on device={device.type}")
    print(f"Query images: {len(dataset_query)} | Database images: {len(dataset_database)}")
    # Print the number of unique identities in the query and database
    print(f"Length of unique identities in query: {len(dataset_query.df[cfg.dataset.label_col].unique().tolist())}")
    print(f"Length of unique identities in database: {len(dataset_database.df[cfg.dataset.label_col].unique().tolist())}")

    wandb_run = None
    if bool(cfg.wandb.enabled):
        try:
            import wandb  # type: ignore
        except Exception as exc:
            raise RuntimeError("wandb is enabled but not installed") from exc
        wandb_run = wandb.init(
            project=str(cfg.wandb.project),
            entity=cfg.wandb.entity if cfg.wandb.entity else None,
            group=cfg.wandb.group if cfg.wandb.group else None,
            tags=list(cfg.wandb.tags) if cfg.wandb.tags else None,
            name=cfg.wandb.name if cfg.wandb.name else probe_wandb_name(cfg, run_id),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    method_artifacts: Dict[str, Any] = {}
    similarity, timings, method_metrics = run_method(
        cfg=cfg,
        method=method,
        model=model,
        embedding_size=embedding_size,
        arch=arch,
        number_of_patches=number_of_patches,
        mean=mean,
        std=std,
        device=device,
        dataset_query=dataset_query,
        dataset_database=dataset_database,
        dataset_calibration=dataset_calibration,
        transform_model=transform_model,
        transform_aliked=transform_aliked,
        checkpoint_path=checkpoint_path,
        cache=cache,
        run_dir=run_dir,
        wandb_run=wandb_run,
        method_artifacts=method_artifacts,
    )
    timings["model_setup_sec"] = float(timings.get("model_setup_sec", 0.0)) + float(backbone_setup_sec)

    top_k_values = [int(k) for k in cfg.benchmark.top_k]
    metrics = compute_metrics(
        dataset_query=dataset_query,
        dataset_database=dataset_database,
        similarity=similarity,
        top_k_values=top_k_values,
        compute_map=bool(cfg.benchmark.compute_map),
        map_at_k=resolve_map_at_k(cfg, len(dataset_database)),
    )
    metrics.update(method_metrics)
    scores_path = save_score_matrix(run_dir / "scores.npz", similarity)
    if scores_path is None:
        print("[probe] score matrix too dense to persist; metrics cannot be recomputed offline")
    else:
        print(f"[probe] saved sparse score matrix: {scores_path}")
    elapsed_sec = time.perf_counter() - run_t0
    timings["total_run_sec"] = float(elapsed_sec)
    timings["total_run_min"] = float(elapsed_sec / 60.0)

    prediction_visuals: List[str] = []
    extra_visuals: Dict[str, List[str]] = {}
    if bool(cfg.visualization.enabled):
        dataset_database_display = make_dataset_view(cfg, dataset_database_raw, transform=transform_display)
        dataset_query_display = make_dataset_view(cfg, dataset_query_raw, transform=transform_display)
        prediction_visuals = visualize_predictions(
            cfg=cfg,
            similarity=similarity,
            dataset_query_display=dataset_query_display,
            dataset_database_display=dataset_database_display,
            run_dir=run_dir,
        )
        if method == "vismatch":
            extra_visuals["vismatch_match"] = [
                str(p) for p in method_artifacts.get("vismatch_match_paths", [])
            ]
        attention_map_path = method_artifacts.get("attention_map_path")
        if attention_map_path:
            extra_visuals["attention_map"] = [str(attention_map_path)]

    visualization_artifacts = finalize_visualizations(
        context,
        prediction_paths=prediction_visuals,
        similarity=similarity,
        dataset_query=dataset_query_raw,
        dataset_database=dataset_database_raw,
        label_col=str(cfg.dataset.label_col),
        top_k=int(cfg.visualization.top_k),
        path_col=str(getattr(cfg.benchmark.methods.vismatch, "path_col", "path"))
        if method == "vismatch"
        else "path",
        extra_paths=extra_visuals,
    )
    visuals = list(prediction_visuals)
    for paths in extra_visuals.values():
        visuals.extend(paths)
    visuals.extend(str(path) for path in visualization_artifacts.values())
    if wandb_run is not None and visuals:
        try:
            import wandb  # type: ignore

            wandb_run.log(
                {"visualizations": [wandb.Image(path) for path in visuals if Path(path).is_file()]},
                step=1,
            )
        except Exception:
            pass

    run_dir.mkdir(parents=True, exist_ok=True)
    output_json = run_dir / "result.json"
    config_snapshot = context.config_snapshot_path
    context.write_config(cfg)

    result = {
        "run_id": run_id,
        "run_utc": context.run_utc,
        "method": method,
        "device": device.type,
        "model_type": cfg.model.type,
        "model_mode": cfg.model.mode,
        "no_background": bool(cfg.dataset.no_background),
        "image_variant": str(cfg.dataset.image_variant),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "dataset_root": cfg.dataset.root,
        "metadata_file": cfg.dataset.metadata_file,
        "split_col": cfg.dataset.split_col,
        "database_split_value": cfg.dataset.database_split_value,
        "query_split_value": cfg.dataset.query_split_value,
        "num_query": len(dataset_query),
        "num_database": len(dataset_database),
        "metrics": metrics,
        "score_matrix_policy": metrics.get("score_matrix_policy"),
        "timings": timings,
        "visualizations": visuals,
        "cache_fingerprints": list(cache.used_keys),
        "vismatch_checkpoint": method_artifacts.get("vismatch_checkpoint"),
        "linear_probe_class_weighting": method_artifacts.get("linear_probe_class_weighting"),
        "efficient_probe_class_weighting": method_artifacts.get("efficient_probe_class_weighting"),
        "classifier_open_set_policy": metrics.get("classifier_open_set_policy"),
        "classifier_evaluation": {
            key: value
            for key, value in metrics.items()
            if key.startswith("classification_") or key == "classifier_open_set_policy"
        },
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    context.write_metrics(metrics)
    context.write_timings(timings)
    context.write_manifest(
        {
            "dataset": str(cfg.dataset.name),
            "animal": str(cfg.dataset.animal),
            "split_protocol": str(cfg.dataset.split_col),
            "model": str(cfg.model.type),
            "method": method,
            "image_variant": str(cfg.dataset.image_variant),
            "variant": str(cfg.benchmark.methods.vismatch.matcher)
            if method == "vismatch"
            else "default",
            "checkpoint": file_identity(checkpoint_path),
            "num_query": len(dataset_query),
            "num_database": len(dataset_database),
            "metrics": metrics,
            "score_matrix_policy": metrics.get("score_matrix_policy"),
            "timings": timings,
            "visualizations": visuals,
            "cache_fingerprints": list(cache.used_keys),
            "vismatch_checkpoint": method_artifacts.get("vismatch_checkpoint"),
            "linear_probe_class_weighting": method_artifacts.get("linear_probe_class_weighting"),
            "efficient_probe_class_weighting": method_artifacts.get("efficient_probe_class_weighting"),
            "classifier_open_set_policy": metrics.get("classifier_open_set_policy"),
            "classifier_evaluation": {
                key: value
                for key, value in metrics.items()
                if key.startswith("classification_") or key == "classifier_open_set_policy"
            },
            "calibration": method_artifacts.get("wildfusion_calibration", method_artifacts.get("local_calibration")),
            "status": "completed",
        },
        status="completed",
    )
    upsert_run_index(
        Path(str(getattr(cfg.reporting, "index_path", "reports/runs.csv"))),
        run_index_row(
            context,
            {
                "status": "completed",
                "dataset": str(cfg.dataset.name),
                "animal": str(cfg.dataset.animal),
                "split_protocol": str(cfg.dataset.split_col),
                "model": str(cfg.model.type),
                "method": method,
                "image_variant": str(cfg.dataset.image_variant),
                "variant": str(cfg.benchmark.methods.vismatch.matcher)
                if method == "vismatch"
                else "default",
                "num_query": len(dataset_query),
                "num_database": len(dataset_database),
                "feature_extraction_sec": timings.get("vismatch_feature_extraction_sec", timings.get("feature_extraction_sec", "")),
                "feature_extraction_compute_sec": timings.get("feature_extraction_compute_sec", ""),
                "feature_cache_lookup_sec": timings.get("feature_cache_lookup_sec", ""),
                "primary_compute_runtime_sec": timings.get("primary_compute_runtime_sec", ""),
                "matcher_runtime_sec": timings.get("matcher_runtime_sec", ""),
                "method_compute_runtime_sec": timings.get("method_compute_runtime_sec", ""),
                "model_setup_sec": timings.get("model_setup_sec", ""),
                "calibration_sec": timings.get("calibration_sec", ""),
                "matching_sec": timings.get("matcher_runtime_sec", timings.get("primary_compute_runtime_sec", "")),
                "total_runtime_sec": elapsed_sec,
                **metrics,
                **timings,
            },
        ),
    )

    csv_base: Dict[str, Any] = {
        "run_id": run_id,
        "run_utc": result["run_utc"],
        "method": method,
        "device": device.type,
        "model_type": cfg.model.type,
        "model_mode": cfg.model.mode,
        "no_background": bool(cfg.dataset.no_background),
        "image_variant": str(cfg.dataset.image_variant),
        "checkpoint_path": result["checkpoint_path"],
        "dataset_root": cfg.dataset.root,
        "metadata_file": cfg.dataset.metadata_file,
        "num_query": len(dataset_query),
        "num_database": len(dataset_database),
        "config_hash": context.config_hash,
        "run_dir": str(context.run_dir),
        "manifest_path": str(context.manifest_path),
        "metrics_path": str(context.metrics_path),
        "visualization_dir": str(context.visualization_dir),
    }
    csv_row = _build_probe_csv_row(csv_base, metrics, timings)
    append_csv_row(Path(cfg.output.csv_path), csv_row)

    print("Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {_format_metric_value(metric_value)}")
    print(f"Saved JSON: {output_json}")
    print(f"Appended CSV row: {cfg.output.csv_path}")
    if visuals:
        print(f"Saved {len(visuals)} visualizations under {context.visualization_dir}")

    if wandb_run is not None:
        wandb_run.log(
            {
                "method": method,
                "model_type": cfg.model.type,
                "model_mode": cfg.model.mode,
                "no_background": bool(cfg.dataset.no_background),
                "image_variant": str(cfg.dataset.image_variant),
                "num_query": len(dataset_query),
                "num_database": len(dataset_database),
                **metrics,
                **timings,
            },
            step=1,
        )
        wandb_run.summary["run_id"] = run_id
        wandb_run.summary["dataset_root"] = cfg.dataset.root
        wandb_run.summary["metadata_file"] = cfg.dataset.metadata_file
        wandb_run.summary["image_variant"] = str(cfg.dataset.image_variant)
        wandb_run.summary["split_col"] = cfg.dataset.split_col
        wandb_run.summary["database_split_value"] = cfg.dataset.database_split_value
        wandb_run.summary["query_split_value"] = cfg.dataset.query_split_value
        elapsed_sec = time.perf_counter() - run_t0
        wandb_run.summary["elapsed_sec"] = float(elapsed_sec)
        wandb_run.summary["elapsed_min"] = float(elapsed_sec / 60.0)
        wandb_run.finish()

    print(f"Elapsed time: {elapsed_sec / 60.0:.2f} min ({elapsed_sec:.1f} sec)")
