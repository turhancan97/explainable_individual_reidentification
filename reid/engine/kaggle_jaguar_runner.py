from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from wildlife_datasets.datasets import WildlifeDataset
from wildlife_tools.features.local import AlikedExtractor
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.calibration import IsotonicCalibration
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion

from models.model import get_model
from reid.engine.finetune_runner import run_finetune
from reid.features.containers import FeatureContainer, get_labels_string
from reid.methods.rdd import run_rdd_benchmark
from reid.training.checkpointing import resolve_model_checkpoint


@dataclass
class SplitMetadata:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    metadata_path: Path


def _choose_device(device_cfg: str) -> torch.device:
    device_cfg = str(device_cfg).lower()
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            print("[kaggle] CUDA requested but unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _resolve_filenames(base_dir: Path, filenames: Sequence[str]) -> List[Path]:
    _ensure_exists(base_dir, f"Image directory base ({base_dir})")
    index_exact: Dict[str, Path] = {}
    index_lower: Dict[str, Path] = {}

    for p in base_dir.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if name not in index_exact:
            index_exact[name] = p
        low = name.lower()
        if low not in index_lower:
            index_lower[low] = p

    resolved: List[Path] = []
    missing: List[str] = []
    for fn in filenames:
        direct = base_dir / fn
        if direct.is_file():
            resolved.append(direct)
            continue
        p = index_exact.get(fn)
        if p is None:
            p = index_lower.get(str(fn).lower())
        if p is None:
            missing.append(str(fn))
            continue
        resolved.append(p)

    if missing:
        sample = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Could not resolve {len(missing)} filenames under {base_dir}. "
            f"Sample missing: {sample}"
        )
    return resolved


def _materialize_alpha_masked_images(
    src_paths: Sequence[Path],
    *,
    cache_dir: Path,
    split_name: str,
) -> List[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[str] = []
    for src in tqdm(src_paths, desc=f"[kaggle][alpha-mask:{split_name}]", mininterval=1, ncols=120):
        src_key = f"{src}|{src.stat().st_size}|{int(src.stat().st_mtime)}"
        out_name = hashlib.sha256(src_key.encode("utf-8")).hexdigest() + ".png"
        dst = cache_dir / out_name
        if not dst.is_file():
            img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {src}")
            if img.ndim == 2:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] >= 4:
                bgr = img[:, :, :3].astype(np.float32)
                alpha = (img[:, :, 3:4].astype(np.float32) / 255.0)
                bgr = (bgr * alpha).astype(np.uint8)
            elif img.ndim == 3:
                bgr = img[:, :, :3]
            else:
                raise ValueError(f"Unsupported image shape for {src}: {img.shape}")
            ok = cv2.imwrite(str(dst), bgr)
            if not ok:
                raise IOError(f"Failed to write alpha-masked image: {dst}")
        out_paths.append(dst.as_posix())
    return out_paths


def _save_alpha_debug_samples(
    src_paths: Sequence[Path],
    masked_paths: Sequence[str],
    *,
    out_dir: Path,
    num_samples: int,
) -> None:
    n = min(int(num_samples), len(src_paths), len(masked_paths))
    if n <= 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        src = src_paths[i]
        dst = Path(masked_paths[i])
        raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        masked = cv2.imread(str(dst), cv2.IMREAD_COLOR)
        if raw is None or masked is None:
            continue

        if raw.ndim == 2:
            raw_rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        elif raw.shape[2] >= 4:
            raw_rgb = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(raw_rgb)
        ax[0].set_title(f"Original\\n{src.name}", fontsize=9)
        ax[0].axis("off")
        ax[1].imshow(masked_rgb)
        ax[1].set_title(f"Alpha-masked\\n{dst.name}", fontsize=9)
        ax[1].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"alpha_mask_debug_{i:03d}.png", dpi=140)
        plt.close(fig)


def _stratified_train_val_split(
    train_csv: pd.DataFrame,
    *,
    seed: int,
    val_ratio: float,
    min_val_per_identity: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "filename" not in train_csv.columns or "ground_truth" not in train_csv.columns:
        raise KeyError("train.csv must contain columns: filename, ground_truth")

    rng = np.random.default_rng(seed)
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []

    for _, group in train_csv.groupby("ground_truth", sort=True):
        idx = np.arange(len(group))
        rng.shuffle(idx)
        n = len(group)
        proposed = int(round(n * val_ratio))
        val_n = max(int(min_val_per_identity), proposed)
        val_n = min(val_n, n - 1) if n > 1 else 0
        val_idx = idx[:val_n]
        tr_idx = idx[val_n:]

        if len(tr_idx) == 0:
            tr_idx = idx[-1:]
            val_idx = idx[:-1]

        train_parts.append(group.iloc[tr_idx])
        val_parts.append(group.iloc[val_idx])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    return train_df, val_df


def _prepare_split_metadata(cfg: DictConfig, run_dir: Path) -> SplitMetadata:
    data_dir = Path(cfg.kaggle.data_dir)
    train_csv_path = data_dir / cfg.kaggle.train_csv
    _ensure_exists(train_csv_path, "Kaggle train.csv")
    train_csv = pd.read_csv(train_csv_path)

    train_df, val_df = _stratified_train_val_split(
        train_csv,
        seed=int(cfg.kaggle.seed),
        val_ratio=float(cfg.validation.val_ratio),
        min_val_per_identity=int(cfg.validation.min_val_per_identity),
    )
    train_base_dir = data_dir / str(cfg.kaggle.train_dir)

    unique_train_filenames = sorted(set(train_csv["filename"].astype(str).tolist()))
    resolved_train_files = _resolve_filenames(train_base_dir, unique_train_filenames)
    filename_to_resolved: Dict[str, str]
    if bool(cfg.alpha_mask.enabled):
        masked_train_paths = _materialize_alpha_masked_images(
            resolved_train_files,
            cache_dir=Path(cfg.alpha_mask.cache_dir) / "train",
            split_name="train",
        )
        filename_to_resolved = {name: path for name, path in zip(unique_train_filenames, masked_train_paths)}
        _save_alpha_debug_samples(
            resolved_train_files,
            masked_train_paths,
            out_dir=run_dir / "alpha_mask_debug" / "train",
            num_samples=int(cfg.alpha_mask.debug_samples),
        )
    else:
        filename_to_resolved = {name: path.as_posix() for name, path in zip(unique_train_filenames, resolved_train_files)}

    def _to_meta(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        abs_paths = [filename_to_resolved[str(x)] for x in df["filename"].astype(str).tolist()]
        return pd.DataFrame(
            {
                "path": abs_paths,
                "identity": df["ground_truth"].astype(str).tolist(),
                "split": split_name,
            }
        )

    meta_train = _to_meta(train_df, "train")
    meta_val = _to_meta(val_df, "val")
    metadata = pd.concat([meta_train, meta_val], ignore_index=True)
    metadata_path = run_dir / "kaggle_trainval_metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    return SplitMetadata(train_df=meta_train, val_df=meta_val, metadata_path=metadata_path)


def _find_latest_checkpoint(results_dir: Path) -> Path:
    return resolve_model_checkpoint(results_dir=results_dir, filename="checkpoint-final.pth")


def _build_finetune_cfg(cfg: DictConfig, metadata_path: Path) -> DictConfig:
    base = OmegaConf.load(cfg.finetune.base_config)
    base.dataset.root = str(metadata_path.parent)
    base.dataset.metadata_file = str(metadata_path.name)
    base.dataset.label_col = "identity"
    base.dataset.split_col = "split"
    base.dataset.train_split_value = "train"
    base.dataset.val_split_value = "val"
    base.dataset.no_background = False
    base.dataset.mask_col = "mask"

    base.model.type = str(cfg.model.type)
    base.train.seed = int(cfg.kaggle.seed)
    base.train.epochs = int(cfg.finetune.epochs)
    base.train.batch_size = int(cfg.finetune.batch_size)
    base.train.num_workers = int(cfg.finetune.num_workers)
    base.train.accumulation_steps = int(cfg.finetune.accumulation_steps)
    base.train.lr = float(cfg.finetune.lr)
    base.train.weight_decay = float(cfg.finetune.weight_decay)
    base.train.amp = str(cfg.finetune.amp)
    base.train.deterministic = bool(cfg.finetune.deterministic)
    base.train.log_every = int(cfg.finetune.log_every)
    base.train.resume_checkpoint = cfg.finetune.resume_checkpoint

    base.loss.margin = float(cfg.finetune.arcface_margin)
    base.loss.scale = float(cfg.finetune.arcface_scale)
    base.scheduler.eta_min_scale = float(cfg.finetune.eta_min_scale)

    base.output.run_dir = str(cfg.finetune.output_run_dir)
    base.output.csv_path = str(Path(cfg.finetune.output_run_dir) / "train_metrics.csv")
    base.output.save_every = int(cfg.finetune.save_every)
    base.output.save_best = bool(cfg.finetune.save_best)
    base.output.best_metric = str(cfg.finetune.best_metric)

    base.benchmark.top_k = [1, 5, 10]
    base.benchmark.compute_map = True
    base.benchmark.val_batch_size = int(cfg.finetune.val_batch_size)
    base.benchmark.val_num_workers = int(cfg.finetune.val_num_workers)

    base.wandb.enabled = bool(cfg.wandb.enabled and cfg.wandb.finetune_enabled)
    if base.wandb.enabled:
        base.wandb.project = str(cfg.wandb.project)
        base.wandb.entity = cfg.wandb.entity
        base.wandb.group = cfg.wandb.group
        base.wandb.tags = list(cfg.wandb.tags)
        base.wandb.name = cfg.wandb.finetune_name if cfg.wandb.finetune_name else None

    base.safety_checks.enabled = bool(cfg.safety_checks.enabled)
    return base


def _build_image_dataset(
    data_dir: Path,
    image_paths: Sequence[str],
    transform: Optional[Any],
    labels: Optional[Sequence[str]] = None,
) -> WildlifeDataset:
    if labels is None:
        labels = list(image_paths)
    if len(labels) != len(image_paths):
        raise ValueError("labels length must match image_paths length")
    df = pd.DataFrame({"path": list(image_paths), "identity": list(labels)})
    return WildlifeDataset(str(data_dir), df, transform=transform, load_label=True, col_label="identity")


def _dataset_signature(image_paths: Sequence[str], tag: str) -> str:
    payload = "\n".join(image_paths) + "\n" + tag
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_embeddings_with_cache(
    *,
    model: Any,
    dataset: WildlifeDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    tta_hflip: bool,
    cache_dir: Path,
    cache_tag: str,
    split_name: str,
    image_paths: Sequence[str],
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = _dataset_signature(image_paths, tag=f"{cache_tag}|{split_name}")
    cache_path = cache_dir / f"{split_name}_{sig}.npz"
    if cache_path.is_file():
        arr = np.load(cache_path)["embeddings"].astype(np.float32)
        print(f"[kaggle] loaded {split_name} embeddings from cache: {cache_path}")
        return arr

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    feats: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[kaggle][extract:{split_name}]", mininterval=1, ncols=120):
            x = batch[0].to(device)
            out = model(x)
            if tta_hflip:
                out_flip = model(torch.flip(x, dims=[3]))
                out = 0.5 * (out + out_flip)
            out = torch.nn.functional.normalize(out, dim=1)
            feats.append(out.detach().cpu().numpy())

    emb = np.concatenate(feats, axis=0).astype(np.float32)
    np.savez_compressed(cache_path, embeddings=emb)
    print(f"[kaggle] saved {split_name} embeddings cache: {cache_path}")
    return emb


def _cosine_0_1_matrix(embeddings: np.ndarray) -> np.ndarray:
    sim = embeddings @ embeddings.T
    sim = np.clip((sim + 1.0) / 2.0, 0.0, 1.0)
    return sim.astype(np.float32)


def _build_candidate_indices(scores: np.ndarray, candidate_k: int) -> np.ndarray:
    n = scores.shape[0]
    kk = min(max(1, int(candidate_k)), max(1, n - 1))
    candidates = np.empty((n, kk), dtype=np.int64)
    ranked = np.argsort(scores, axis=1)[:, ::-1]
    for i in range(n):
        row = ranked[i]
        row = row[row != i]
        candidates[i] = row[:kk]
    return candidates


def _normalize_minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    vmin = float(np.min(x))
    vmax = float(np.max(x))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.full_like(x, 0.5)
    if vmax - vmin < 1e-12:
        return np.full_like(x, 0.5)
    return (x - vmin) / (vmax - vmin)


def _fuse_stage_a_and_rdd(
    stage_scores: np.ndarray,
    rdd_scores_raw: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    mode: str = "delta",
    alpha: float = 0.02,
    min_stage_score: float = 0.0,
    symmetrize: bool = True,
) -> np.ndarray:
    if mode not in {"delta", "blend", "replace"}:
        raise ValueError("RDD fusion mode must be one of: delta, blend, replace")

    alpha = float(np.clip(alpha, 0.0, 1.0))
    min_stage_score = float(np.clip(min_stage_score, 0.0, 1.0))

    final = stage_scores.astype(np.float32).copy()
    n = final.shape[0]
    for i in range(n):
        cands = candidate_indices[i]
        vals = rdd_scores_raw[i, cands]
        valid_mask = vals > -1e8
        if not np.any(valid_mask):
            continue

        idx = cands[valid_mask]
        vals_valid = vals[valid_mask]
        stage_vals = stage_scores[i, idx].astype(np.float32)

        if min_stage_score > 0.0:
            stage_gate = stage_vals >= min_stage_score
            if not np.any(stage_gate):
                continue
            idx = idx[stage_gate]
            vals_valid = vals_valid[stage_gate]
            stage_vals = stage_vals[stage_gate]

        norm_vals = _normalize_minmax(vals_valid).astype(np.float32)
        if mode == "replace":
            fused = norm_vals
        elif mode == "blend":
            fused = (1.0 - alpha) * stage_vals + alpha * norm_vals
        else:
            fused = stage_vals + alpha * (norm_vals - 0.5)
        final[i, idx] = np.clip(fused, 0.0, 1.0)

    np.fill_diagonal(final, 1.0)
    if bool(symmetrize) and final.shape[0] == final.shape[1]:
        final = (0.5 * (final + final.T)).astype(np.float32)
        np.fill_diagonal(final, 1.0)
    return np.clip(final, 0.0, 1.0)


def _identity_balanced_map_all_vs_all(scores: np.ndarray, labels: Sequence[str]) -> float:
    labels_arr = np.asarray(labels)
    n = len(labels_arr)
    if n == 0:
        return float("nan")

    ranked = np.argsort(scores, axis=1)[:, ::-1]
    ap_by_identity: Dict[str, List[float]] = {}

    for i in range(n):
        order = ranked[i]
        order = order[order != i]
        rel = (labels_arr[order] == labels_arr[i]).astype(np.float32)
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue
        cum_rel = np.cumsum(rel)
        ranks = np.arange(1, len(rel) + 1, dtype=np.float32)
        precision = cum_rel / ranks
        ap = float((precision * rel).sum() / n_rel)
        ap_by_identity.setdefault(str(labels_arr[i]), []).append(ap)

    if not ap_by_identity:
        return float("nan")
    per_identity = [float(np.mean(v)) for v in ap_by_identity.values() if len(v) > 0]
    return float(np.mean(per_identity)) if per_identity else float("nan")


def _validate_submission(submission: pd.DataFrame, test_df: pd.DataFrame, strict_rows: bool) -> None:
    if list(submission.columns) != ["row_id", "similarity"]:
        raise ValueError(f"Submission columns must be ['row_id','similarity'], got {list(submission.columns)}")

    if strict_rows and len(submission) != 137270:
        raise ValueError(f"Submission must have exactly 137270 rows, got {len(submission)}")

    if len(submission) != len(test_df):
        raise ValueError(f"Submission rows ({len(submission)}) must equal test rows ({len(test_df)})")

    if submission["row_id"].tolist() != test_df["row_id"].tolist():
        raise ValueError("row_id values/order do not match test.csv exactly")

    sims = submission["similarity"].to_numpy(dtype=np.float64)
    if not np.isfinite(sims).all():
        raise ValueError("Submission similarity contains NaN/Inf")
    if (sims < 0).any() or (sims > 1).any():
        raise ValueError("Submission similarity values must be in [0, 1]")


def _apply_ensemble(submission_vec: np.ndarray, test_df: pd.DataFrame, additional_paths: Sequence[str]) -> np.ndarray:
    vectors = [_normalize_minmax(submission_vec)]
    for p in additional_paths:
        add_df = pd.read_csv(p)
        if list(add_df.columns) != ["row_id", "similarity"]:
            raise ValueError(f"Ensemble file must have columns row_id,similarity: {p}")
        if add_df["row_id"].tolist() != test_df["row_id"].tolist():
            raise ValueError(f"Ensemble file row_id mismatch: {p}")
        vec = add_df["similarity"].to_numpy(dtype=np.float64)
        vectors.append(_normalize_minmax(vec))
    return np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)


class _StageADeepExtractor:
    def __init__(
        self,
        *,
        model: Any,
        device: torch.device,
        batch_size: int,
        num_workers: int,
        tta_hflip: bool,
        cache_dir: Path,
        cache_tag: str,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tta_hflip = tta_hflip
        self.cache_dir = cache_dir
        self.cache_tag = cache_tag

    def __call__(self, dataset: WildlifeDataset) -> FeatureContainer:
        image_paths = dataset.df["path"].astype(str).tolist()
        emb = _extract_embeddings_with_cache(
            model=self.model,
            dataset=dataset,
            device=self.device,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            tta_hflip=self.tta_hflip,
            cache_dir=self.cache_dir,
            cache_tag=self.cache_tag,
            split_name="stage_a_wildfusion_dynamic",
            image_paths=image_paths,
        )
        return FeatureContainer(features=emb, labels_string=get_labels_string(dataset, "identity"))


def _call_similarity(matcher_obj: Any, query_ds: WildlifeDataset, database_ds: WildlifeDataset, b_value: int) -> np.ndarray:
    try:
        return np.asarray(matcher_obj(query_ds, database_ds, B=int(b_value)))
    except TypeError as exc:
        if "unexpected keyword argument 'B'" not in str(exc):
            raise
        return np.asarray(matcher_obj(query_ds, database_ds))


def run_kaggle_jaguar(cfg: DictConfig) -> None:
    t0 = time.perf_counter()
    run_started = datetime.utcnow()
    run_id = run_started.strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(cfg.kaggle.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    mode = str(getattr(cfg.submission, "mode", "stage_a_plus_rdd"))
    if mode not in {"stage_a_only", "stage_a_plus_rdd"}:
        raise ValueError("submission.mode must be one of: stage_a_only, stage_a_plus_rdd")
    stage_a_method = str(getattr(cfg.stage_a, "method", "cosine"))
    if stage_a_method not in {"cosine", "wildfusion"}:
        raise ValueError("stage_a.method must be one of: cosine, wildfusion")
    rdd_fusion_mode = str(getattr(cfg.submission, "rdd_fusion_mode", "delta"))
    if rdd_fusion_mode not in {"delta", "blend", "replace"}:
        raise ValueError("submission.rdd_fusion_mode must be one of: delta, blend, replace")
    rdd_fusion_alpha = float(getattr(cfg.submission, "rdd_fusion_alpha", 0.02))
    rdd_fusion_min_stage = float(getattr(cfg.submission, "rdd_min_stage_score", 0.0))
    rdd_fusion_sym = bool(getattr(cfg.submission, "rdd_fusion_symmetrize", True))
    print(
        f"[kaggle] workflow mode={mode}: finetune -> stage_a({stage_a_method})"
        + (" + rdd -> submission" if mode == "stage_a_plus_rdd" else " -> submission")
    )
    OmegaConf.save(cfg, run_dir / "config.snapshot.yaml")

    data_dir = Path(cfg.kaggle.data_dir)
    _ensure_exists(data_dir, "Kaggle data_dir")
    test_csv_path = data_dir / cfg.kaggle.test_csv
    _ensure_exists(test_csv_path, "Kaggle test.csv")
    test_df_all = pd.read_csv(test_csv_path)

    if bool(cfg.kaggle.dry_run.enabled):
        pair_limit = int(cfg.kaggle.dry_run.pair_limit)
        test_df = test_df_all.head(pair_limit).copy()
        print(f"[kaggle][dry-run] using first {len(test_df)} rows from test.csv")
    else:
        test_df = test_df_all

    split_meta = _prepare_split_metadata(cfg, run_dir)

    ft_cfg = _build_finetune_cfg(cfg, split_meta.metadata_path)
    if bool(cfg.finetune.run_finetune):
        print("[kaggle] starting finetune...")
        run_finetune(ft_cfg)

    if cfg.finetune.checkpoint_path:
        checkpoint_path = Path(cfg.finetune.checkpoint_path)
    else:
        checkpoint_path = _find_latest_checkpoint(Path(ft_cfg.output.run_dir))

    _ensure_exists(checkpoint_path, "finetuned checkpoint")
    print(f"[kaggle] using checkpoint: {checkpoint_path}")

    device = _choose_device(str(cfg.kaggle.device))
    model, embedding_size, mean, std, img_size, arch, patch_size, number_of_patches = get_model(cfg.model.type)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    checkpoint_stat = checkpoint_path.stat()
    cache_tag = (
        f"model={cfg.model.type}|checkpoint={checkpoint_path}|size={checkpoint_stat.st_size}|mtime={int(checkpoint_stat.st_mtime)}|"
        f"img={img_size}|tta={bool(cfg.stage_a.tta_hflip)}"
    )

    test_images = sorted(set(test_df["query_image"].astype(str).tolist()) | set(test_df["gallery_image"].astype(str).tolist()))
    test_base_dir = data_dir / str(cfg.kaggle.test_dir)
    resolved_test = _resolve_filenames(test_base_dir, test_images)
    if bool(cfg.alpha_mask.enabled):
        test_paths = _materialize_alpha_masked_images(
            resolved_test,
            cache_dir=Path(cfg.alpha_mask.cache_dir) / "test",
            split_name="test",
        )
        _save_alpha_debug_samples(
            resolved_test,
            test_paths,
            out_dir=run_dir / "alpha_mask_debug" / "test",
            num_samples=int(cfg.alpha_mask.debug_samples),
        )
    else:
        test_paths = [p.as_posix() for p in resolved_test]
    image_to_idx = {img: i for i, img in enumerate(test_images)}

    transform = T.Compose([T.Resize([img_size, img_size]), T.ToTensor(), T.Normalize(mean=mean, std=std)])
    test_dataset_stage_a = _build_image_dataset(data_dir, test_paths, transform=transform)
    test_dataset_rdd = _build_image_dataset(data_dir, test_paths, transform=None)

    timings: Dict[str, float] = {}

    t_stage_a = time.perf_counter()
    wildfusion_matcher = None
    if stage_a_method == "cosine":
        test_embeddings = _extract_embeddings_with_cache(
            model=model,
            dataset=test_dataset_stage_a,
            device=device,
            batch_size=int(cfg.stage_a.batch_size),
            num_workers=int(cfg.stage_a.num_workers),
            tta_hflip=bool(cfg.stage_a.tta_hflip),
            cache_dir=Path(cfg.stage_a.cache_dir),
            cache_tag=cache_tag,
            split_name="test",
            image_paths=test_paths,
        )
        stage_a_scores = _cosine_0_1_matrix(test_embeddings)
    else:
        wf_cfg = cfg.stage_a.wildfusion
        calib_paths = split_meta.train_df["path"].astype(str).tolist()
        calib_labels = split_meta.train_df["identity"].astype(str).tolist()
        calib_size = int(getattr(wf_cfg, "calibration_size", 0))
        if calib_size > 0:
            calib_paths = calib_paths[:calib_size]
            calib_labels = calib_labels[:calib_size]
        dataset_calibration = _build_image_dataset(data_dir, calib_paths, transform=None, labels=calib_labels)

        transform_aliked = T.Compose([T.Resize([512, 512]), T.ToTensor()])
        matcher_aliked = SimilarityPipeline(
            matcher=MatchLightGlue(features="aliked", device=device, batch_size=int(wf_cfg.local_batch_size)),
            extractor=AlikedExtractor(),
            transform=transform_aliked,
            calibration=IsotonicCalibration(),
        )
        matcher_mega = SimilarityPipeline(
            matcher=CosineSimilarity(),
            extractor=_StageADeepExtractor(
                model=model,
                device=device,
                batch_size=int(wf_cfg.deep_batch_size),
                num_workers=int(wf_cfg.deep_num_workers),
                tta_hflip=bool(cfg.stage_a.tta_hflip),
                cache_dir=Path(cfg.stage_a.cache_dir),
                cache_tag=cache_tag,
            ),
            transform=transform,
            calibration=IsotonicCalibration(),
        )
        wildfusion_matcher = WildFusion(
            calibrated_pipelines=[matcher_aliked, matcher_mega],
            priority_pipeline=matcher_mega,
        )
        wildfusion_matcher.fit_calibration(dataset_calibration, dataset_calibration)
        stage_a_scores = np.asarray(
            _call_similarity(
                wildfusion_matcher,
                test_dataset_stage_a,
                test_dataset_stage_a,
                int(wf_cfg.B),
            ),
            dtype=np.float32,
        )
        stage_a_scores = np.clip(stage_a_scores, 0.0, 1.0)

    if bool(cfg.kaggle.fast_mode):
        candidate_k = int(cfg.stage_a.fast_candidate_k)
        rdd_top_k = int(cfg.rdd.fast_top_k)
        print(f"[kaggle][fast_mode] candidate_k={candidate_k}, rdd.top_k={rdd_top_k}")
    else:
        candidate_k = int(cfg.stage_a.candidate_k)
        rdd_top_k = int(cfg.rdd.top_k)

    timings["stage_a_sec"] = time.perf_counter() - t_stage_a

    rdd_cfg = OmegaConf.create(
        {
            "dataset": {
                "root": str(data_dir),
                "mask_col": "mask",
                "no_background": False,
                "label_col": "identity",
            },
            "benchmark": {
                "methods": {
                    "rdd": {
                        "repo_dir": str(cfg.rdd.repo_dir),
                        "config_path": str(cfg.rdd.config_path),
                        "weights": str(cfg.rdd.weights),
                        "cache_dir": str(cfg.rdd.cache_dir),
                        "device": str(cfg.rdd.device),
                        "path_col": "path",
                        "resize_max": int(cfg.rdd.resize_max),
                        "top_k": int(rdd_top_k),
                        "stage_a_method": "cosine",
                        "candidate_k": int(candidate_k),
                    }
                },
                "seed": int(cfg.kaggle.seed),
            },
            "model": {"type": str(cfg.model.type), "mode": "finetuned"},
            "visualization": {
                "enabled": bool(cfg.visualization.enabled),
                "dir": str(run_dir / "visualizations"),
                "num_examples": int(cfg.visualization.num_examples),
                "rdd_max_matches": int(cfg.visualization.rdd_max_matches),
            },
        }
    )

    candidate_indices: Optional[np.ndarray] = None
    final_scores: np.ndarray
    rdd_scores_raw: Optional[np.ndarray] = None
    if mode == "stage_a_plus_rdd":
        candidate_indices = _build_candidate_indices(stage_a_scores, candidate_k=candidate_k)
        t_rdd = time.perf_counter()
        rdd_scores_raw, rdd_timings, _ = run_rdd_benchmark(
            cfg=rdd_cfg,
            dataset_query=test_dataset_rdd,
            dataset_database=test_dataset_rdd,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            mean=mean,
            std=std,
            candidate_indices=candidate_indices,
            method_artifacts={},
        )
        timings.update({f"rdd_{k}": float(v) for k, v in rdd_timings.items()})
        timings["rdd_stage_sec"] = time.perf_counter() - t_rdd

        t_fuse = time.perf_counter()
        final_scores = _fuse_stage_a_and_rdd(
            stage_scores=stage_a_scores,
            rdd_scores_raw=rdd_scores_raw,
            candidate_indices=candidate_indices,
            mode=rdd_fusion_mode,
            alpha=rdd_fusion_alpha,
            min_stage_score=rdd_fusion_min_stage,
            symmetrize=rdd_fusion_sym,
        )
        timings["fusion_sec"] = time.perf_counter() - t_fuse
        print(
            f"[kaggle][fusion] mode={rdd_fusion_mode} alpha={rdd_fusion_alpha:.3f} "
            f"min_stage_score={rdd_fusion_min_stage:.3f} symmetrize={rdd_fusion_sym}"
        )
    else:
        final_scores = stage_a_scores.copy()
        timings["rdd_stage_sec"] = 0.0
        timings["fusion_sec"] = 0.0

    np.save(run_dir / "stage_a_scores.npy", stage_a_scores)
    if candidate_indices is not None:
        np.save(run_dir / "candidate_indices.npy", candidate_indices)
    if rdd_scores_raw is not None:
        np.save(run_dir / "rdd_scores_raw.npy", rdd_scores_raw)
    np.save(run_dir / "final_scores.npy", final_scores)
    pd.DataFrame({"image": test_images, "index": np.arange(len(test_images))}).to_csv(run_dir / "test_image_index.csv", index=False)

    if bool(cfg.validation.enabled):
        print("[kaggle] running local validation...")
        val_paths = split_meta.val_df["path"].astype(str).tolist()
        val_images = [Path(p).name for p in val_paths]
        val_labels = split_meta.val_df["identity"].astype(str).tolist()
        val_dataset_stage_a = _build_image_dataset(data_dir, val_paths, transform=transform, labels=val_labels)
        val_dataset_rdd = _build_image_dataset(data_dir, val_paths, transform=None)

        if stage_a_method == "cosine":
            val_embeddings = _extract_embeddings_with_cache(
                model=model,
                dataset=val_dataset_stage_a,
                device=device,
                batch_size=int(cfg.stage_a.batch_size),
                num_workers=int(cfg.stage_a.num_workers),
                tta_hflip=bool(cfg.stage_a.tta_hflip),
                cache_dir=Path(cfg.stage_a.cache_dir),
                cache_tag=cache_tag,
                split_name="val",
                image_paths=val_paths,
            )
            val_stage = _cosine_0_1_matrix(val_embeddings)
        else:
            wf_cfg = cfg.stage_a.wildfusion
            if wildfusion_matcher is None:
                raise RuntimeError("wildfusion matcher was expected but not initialized")
            val_stage = np.asarray(
                _call_similarity(
                    wildfusion_matcher,
                    val_dataset_stage_a,
                    val_dataset_stage_a,
                    int(wf_cfg.B),
                ),
                dtype=np.float32,
            )
            val_stage = np.clip(val_stage, 0.0, 1.0)
        if mode == "stage_a_plus_rdd":
            val_candidates = _build_candidate_indices(val_stage, candidate_k=min(candidate_k, max(1, len(val_images) - 1)))
            val_rdd_raw, _, _ = run_rdd_benchmark(
                cfg=rdd_cfg,
                dataset_query=val_dataset_rdd,
                dataset_database=val_dataset_rdd,
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                mean=mean,
                std=std,
                candidate_indices=val_candidates,
                method_artifacts={},
            )
            val_final = _fuse_stage_a_and_rdd(
                stage_scores=val_stage,
                rdd_scores_raw=val_rdd_raw,
                candidate_indices=val_candidates,
                mode=rdd_fusion_mode,
                alpha=rdd_fusion_alpha,
                min_stage_score=rdd_fusion_min_stage,
                symmetrize=rdd_fusion_sym,
            )
        else:
            val_final = val_stage
        val_ib_map = _identity_balanced_map_all_vs_all(val_final, val_labels)
        validation_report = {
            "val_num_images": int(len(val_images)),
            "val_num_identities": int(split_meta.val_df["identity"].nunique()),
            "identity_balanced_map": float(val_ib_map),
            "submission_mode": mode,
            "rdd_fusion_mode": rdd_fusion_mode,
            "rdd_fusion_alpha": float(rdd_fusion_alpha),
            "rdd_min_stage_score": float(rdd_fusion_min_stage),
            "rdd_fusion_symmetrize": bool(rdd_fusion_sym),
        }
        with (run_dir / "validation_report.json").open("w", encoding="utf-8") as f:
            json.dump(validation_report, f, indent=2)
        print(f"[kaggle][validation] identity_balanced_mAP={val_ib_map:.6f}")

    t_submit = time.perf_counter()
    pair_scores = np.empty(len(test_df), dtype=np.float32)
    for i, row in enumerate(tqdm(test_df.itertuples(index=False), total=len(test_df), desc="[kaggle][pairs]", mininterval=1, ncols=120)):
        q = str(getattr(row, "query_image"))
        g = str(getattr(row, "gallery_image"))
        qi = image_to_idx[q]
        gi = image_to_idx[g]
        pair_scores[i] = float(final_scores[qi, gi])

    pair_scores = np.clip(pair_scores, 0.0, 1.0)

    add_submissions = [str(p) for p in list(cfg.ensemble.additional_submissions)] if bool(cfg.ensemble.enabled) else []
    if add_submissions:
        pair_scores = _apply_ensemble(pair_scores, test_df=test_df, additional_paths=add_submissions)
        pair_scores = np.clip(pair_scores, 0.0, 1.0)

    submission = pd.DataFrame({"row_id": test_df["row_id"].astype(int), "similarity": pair_scores.astype(np.float32)})
    strict_rows = bool(cfg.submission.strict_row_count) and not bool(cfg.kaggle.dry_run.enabled)
    _validate_submission(submission, test_df=test_df, strict_rows=strict_rows)

    submission_path = run_dir / f"submission_{mode}.csv"
    submission.to_csv(submission_path, index=False)
    if cfg.submission.output_csv:
        out_csv_base = Path(cfg.submission.output_csv)
        out_csv = out_csv_base.with_name(f"{out_csv_base.stem}_{mode}{out_csv_base.suffix}")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(out_csv, index=False)

    print(f"[kaggle] submission saved: {submission_path}")

    timings["submission_sec"] = time.perf_counter() - t_submit
    timings["total_sec"] = time.perf_counter() - t0

    manifest = {
        "run_id": run_id,
        "run_utc": run_started.isoformat() + "Z",
        "submission_mode": mode,
        "rdd_fusion_mode": rdd_fusion_mode,
        "rdd_fusion_alpha": float(rdd_fusion_alpha),
        "rdd_min_stage_score": float(rdd_fusion_min_stage),
        "rdd_fusion_symmetrize": bool(rdd_fusion_sym),
        "timings": {k: float(v) for k, v in timings.items()},
        "git_commit": _git_commit_hash(),
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "hostname": platform.node(),
            "pid": os.getpid(),
        },
    }
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[kaggle] elapsed: {timings['total_sec'] / 60.0:.2f} min ({timings['total_sec']:.1f} sec)")
