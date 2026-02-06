import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from wildlife_datasets.datasets import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.features.local import AlikedExtractor
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.calibration import IsotonicCalibration
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion

from models.model import get_model
from reid.data.dataset_view import BenchmarkDatasetView
from reid.evaluation.metrics import compute_metrics
from reid.features.containers import FeatureContainer, get_labels_string
from reid.utils.io import append_csv_row, ensure_dir, ensure_file
from reid.utils.repro import set_reproducible


def choose_device(device_cfg: str) -> torch.device:
    if device_cfg == "cpu":
        return torch.device("cpu")
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_latest_checkpoint(results_dir: Path, filename: str) -> Path:
    ensure_dir(results_dir, "Results directory")
    run_dirs = [p for p in results_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {results_dir}")
    run_dirs.sort(key=lambda p: p.stat().st_mtime)
    checkpoint_path = run_dirs[-1] / filename
    ensure_file(checkpoint_path, "Checkpoint")
    return checkpoint_path


def load_backbone(cfg: DictConfig) -> Tuple[Any, int, Tuple[float, ...], Tuple[float, ...], int, Optional[Path]]:
    model, embedding_size, mean, std, img_size = get_model(cfg.model.type)
    checkpoint_path: Optional[Path] = None

    if cfg.model.mode == "finetuned":
        if cfg.model.checkpoint.path:
            checkpoint_path = Path(cfg.model.checkpoint.path)
        elif cfg.model.checkpoint.from_latest_results:
            checkpoint_path = find_latest_checkpoint(Path(cfg.model.checkpoint.results_dir), cfg.model.checkpoint.filename)
        else:
            raise ValueError("Finetuned mode requires checkpoint.path or checkpoint.from_latest_results=true")

        ensure_file(checkpoint_path, "Checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint)

    return model, embedding_size, mean, std, img_size, checkpoint_path


def build_transforms(mean: Tuple[float, ...], std: Tuple[float, ...], img_size: int) -> Tuple[T.Compose, T.Compose, T.Compose]:
    transform_display = T.Compose([T.Resize([img_size, img_size])])
    transform_model = T.Compose([*transform_display.transforms, T.ToTensor(), T.Normalize(mean=mean, std=std)])
    transform_aliked = T.Compose([T.Resize([512, 512]), T.ToTensor()])
    return transform_display, transform_model, transform_aliked


def load_dataset_splits(cfg: DictConfig) -> Tuple[WildlifeDataset, WildlifeDataset, WildlifeDataset]:
    root = Path(cfg.dataset.root)
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


def get_calibration_dataset(dataset_database: WildlifeDataset, size: int, root: str, label_col: str) -> WildlifeDataset:
    if size <= 0:
        raise ValueError("dataset.calibration_size must be > 0")
    if len(dataset_database.metadata) < size:
        size = len(dataset_database.metadata)
    return WildlifeDataset(root, df=dataset_database.metadata.iloc[:size], load_label=True, col_label=label_col)


def dataset_digest(dataset: WildlifeDataset, label_col: str) -> str:
    df = dataset.df if hasattr(dataset, "df") else dataset.metadata
    candidate_cols = ["path", "filepath", "file", "image_path", label_col]
    cols = [c for c in candidate_cols if c in df.columns]
    if not cols:
        cols = [label_col] if label_col in df.columns else []
    if cols:
        payload = "\n".join(df[cols].astype(str).agg("|".join, axis=1).tolist())
    else:
        payload = f"n={len(df)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FeatureCache:
    def __init__(self, enabled: bool, cache_dir: Path, fmt: str):
        self.enabled = enabled
        self.cache_dir = cache_dir
        self.fmt = fmt
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.fmt not in {"pt", "npz"}:
            raise ValueError("benchmark.cache.format must be 'pt' or 'npz'")

    def _path_for(self, key: str) -> Path:
        ext = "pt" if self.fmt == "pt" else "npz"
        return self.cache_dir / f"{key}.{ext}"

    def get_or_compute(self, key: str, compute_fn) -> np.ndarray:
        path = self._path_for(key)
        if self.enabled and path.is_file():
            return self._load(path)
        data = self._normalize_features(compute_fn())
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


def make_cache_key(cfg: DictConfig, method: str, split_name: str, dataset_sig: str, checkpoint_path: Optional[Path]) -> str:
    checkpoint_tag = "pretrained"
    if checkpoint_path is not None:
        stat = checkpoint_path.stat()
        checkpoint_tag = f"{checkpoint_path}|{stat.st_size}|{int(stat.st_mtime)}"
    payload = {
        "method": method,
        "split": split_name,
        "dataset": dataset_sig,
        "model_type": cfg.model.type,
        "mode": cfg.model.mode,
        "no_background": bool(cfg.dataset.no_background),
        "checkpoint": checkpoint_tag,
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
    dataset_sig = dataset_digest(dataset, cfg.dataset.label_col)
    cache_key = make_cache_key(cfg, method_name, split_name, dataset_sig, checkpoint_path)

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

    def __call__(self, dataset: WildlifeDataset) -> np.ndarray:
        dataset_sig = dataset_digest(dataset, self.cfg.dataset.label_col)
        cache_key = make_cache_key(self.cfg, self.method_name, "dynamic_split", dataset_sig, self.checkpoint_path)

        def _compute():
            extractor = DeepFeatures(self.model, device=self.device, batch_size=self.batch_size, num_workers=self.num_workers)
            return extractor(dataset)

        features = self.cache.get_or_compute(cache_key, _compute)
        labels_string = get_labels_string(dataset, self.cfg.dataset.label_col)
        return FeatureContainer(features=features, labels_string=labels_string)


def run_method(
    cfg: DictConfig,
    method: str,
    model: Any,
    device: torch.device,
    dataset_query: WildlifeDataset,
    dataset_database: WildlifeDataset,
    dataset_calibration: WildlifeDataset,
    transform_model: T.Compose,
    transform_aliked: T.Compose,
    checkpoint_path: Optional[Path],
    cache: FeatureCache,
) -> Tuple[np.ndarray, Dict[str, float]]:
    timings: Dict[str, float] = {}
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
        t_extract = time.perf_counter()
        matcher_aliked = SimilarityPipeline(
            matcher=MatchLightGlue(features="aliked", device=device, batch_size=settings.local_batch_size),
            extractor=AlikedExtractor(),
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
        wildfusion.fit_calibration(dataset_calibration, dataset_calibration)
        timings["feature_extraction_sec"] = time.perf_counter() - t_extract
        t_sim = time.perf_counter()
        similarity = _call_similarity(wildfusion, dataset_query, dataset_database, settings.B)
        timings["similarity_sec"] = time.perf_counter() - t_sim

    elif method == "local_lightglue":
        settings = cfg.benchmark.methods.local_lightglue
        t_extract = time.perf_counter()
        matcher_local = SimilarityPipeline(
            matcher=MatchLightGlue(features="aliked", device=device, batch_size=settings.local_batch_size),
            extractor=AlikedExtractor(),
            transform=transform_aliked,
            calibration=IsotonicCalibration(),
        )
        matcher_local.fit_calibration(dataset_calibration, dataset_calibration)
        timings["feature_extraction_sec"] = time.perf_counter() - t_extract
        t_sim = time.perf_counter()
        similarity = _call_similarity(matcher_local, dataset_query, dataset_database, settings.B)
        timings["similarity_sec"] = time.perf_counter() - t_sim

    else:
        raise ValueError(f"Unsupported method '{method}'. Supported: cosine, wildfusion, local_lightglue")

    timings["total_method_sec"] = time.perf_counter() - t0
    return np.asarray(similarity), timings


def visualize_predictions(
    cfg: DictConfig,
    similarity: np.ndarray,
    dataset_query_display: WildlifeDataset,
    dataset_database_display: WildlifeDataset,
    run_id: str,
) -> List[str]:
    vis_cfg = cfg.visualization
    out_dir = Path(vis_cfg.dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked_idx = np.argsort(similarity, axis=1)[:, ::-1]
    rng = np.random.default_rng(cfg.benchmark.seed)
    num_queries = len(dataset_query_display)
    num_examples = min(vis_cfg.num_examples, num_queries)
    query_indices = rng.choice(np.arange(num_queries), num_examples, replace=False)

    saved_files: List[str] = []
    for query_idx in query_indices:
        database_idx = ranked_idx[query_idx, : vis_cfg.top_k]
        scores = similarity[query_idx, database_idx]
        fig, ax = plt.subplots(1, 1 + vis_cfg.top_k, figsize=((1 + vis_cfg.top_k) * 3, 3))

        query_data = dataset_query_display[query_idx]
        ax[0].imshow(query_data[0])
        ax[0].set_title(str(query_data[1]))
        ax[0].axis("off")

        for i, db_idx in enumerate(database_idx):
            database_data = dataset_database_display[db_idx]
            ax[i + 1].imshow(database_data[0])
            ax[i + 1].set_title(f"{database_data[1]}: {scores[i]:.3f}")
            ax[i + 1].axis("off")

        out_path = out_dir / f"predictions_{query_idx}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        saved_files.append(str(out_path))

    return saved_files


def run_probe(cfg: DictConfig) -> None:
    set_reproducible(int(cfg.benchmark.seed), bool(cfg.benchmark.deterministic))
    device = choose_device(cfg.model.device)
    model, _, mean, std, img_size, checkpoint_path = load_backbone(cfg)
    model.eval()

    transform_display, transform_model, transform_aliked = build_transforms(mean, std, img_size)
    _, dataset_database_raw, dataset_query_raw = load_dataset_splits(cfg)
    dataset_calibration_raw = get_calibration_dataset(
        dataset_database=dataset_database_raw,
        size=int(cfg.dataset.calibration_size),
        root=cfg.dataset.root,
        label_col=cfg.dataset.label_col,
    )

    method = str(cfg.benchmark.method)
    if method == "cosine":
        dataset_database = make_dataset_view(cfg, dataset_database_raw, transform=transform_model)
        dataset_query = make_dataset_view(cfg, dataset_query_raw, transform=transform_model)
        dataset_calibration = make_dataset_view(cfg, dataset_calibration_raw, transform=transform_model)
    else:
        dataset_database = make_dataset_view(cfg, dataset_database_raw, transform=None)
        dataset_query = make_dataset_view(cfg, dataset_query_raw, transform=None)
        dataset_calibration = make_dataset_view(cfg, dataset_calibration_raw, transform=None)

    cache = FeatureCache(
        enabled=bool(cfg.benchmark.cache.enabled),
        cache_dir=Path(cfg.benchmark.cache.dir),
        fmt=str(cfg.benchmark.cache.format),
    )

    run_started = datetime.utcnow()
    run_id = run_started.strftime("run_%Y%m%d_%H%M%S")
    print(f"Running method={method} on device={device.type}")
    print(f"Query images: {len(dataset_query)} | Database images: {len(dataset_database)}")

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
            name=cfg.wandb.name if cfg.wandb.name else run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    similarity, timings = run_method(
        cfg=cfg,
        method=method,
        model=model,
        device=device,
        dataset_query=dataset_query,
        dataset_database=dataset_database,
        dataset_calibration=dataset_calibration,
        transform_model=transform_model,
        transform_aliked=transform_aliked,
        checkpoint_path=checkpoint_path,
        cache=cache,
    )

    top_k_values = [int(k) for k in cfg.benchmark.top_k]
    metrics = compute_metrics(
        dataset_query=dataset_query,
        dataset_database=dataset_database,
        similarity=similarity,
        top_k_values=top_k_values,
        compute_map=bool(cfg.benchmark.compute_map),
    )

    visuals: List[str] = []
    if bool(cfg.visualization.enabled):
        dataset_database_display = make_dataset_view(cfg, dataset_database_raw, transform=transform_display)
        dataset_query_display = make_dataset_view(cfg, dataset_query_raw, transform=transform_display)
        visuals = visualize_predictions(
            cfg=cfg,
            similarity=similarity,
            dataset_query_display=dataset_query_display,
            dataset_database_display=dataset_database_display,
            run_id=run_id,
        )
        if wandb_run is not None:
            try:
                import wandb  # type: ignore

                wandb_run.log(
                    {
                        "visualizations": [wandb.Image(path) for path in visuals],
                    },
                    step=1,
                )
            except Exception:
                pass

    run_dir = Path(cfg.output.run_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_json = run_dir / "result.json"
    config_snapshot = run_dir / "config.snapshot.yaml"
    OmegaConf.save(cfg, config_snapshot)

    result = {
        "run_id": run_id,
        "run_utc": run_started.isoformat() + "Z",
        "method": method,
        "device": device.type,
        "model_type": cfg.model.type,
        "model_mode": cfg.model.mode,
        "no_background": bool(cfg.dataset.no_background),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "dataset_root": cfg.dataset.root,
        "metadata_file": cfg.dataset.metadata_file,
        "split_col": cfg.dataset.split_col,
        "database_split_value": cfg.dataset.database_split_value,
        "query_split_value": cfg.dataset.query_split_value,
        "num_query": len(dataset_query),
        "num_database": len(dataset_database),
        "metrics": metrics,
        "timings": timings,
        "visualizations": visuals,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    csv_row: Dict[str, Any] = {
        "run_id": run_id,
        "run_utc": result["run_utc"],
        "method": method,
        "device": device.type,
        "model_type": cfg.model.type,
        "model_mode": cfg.model.mode,
        "no_background": bool(cfg.dataset.no_background),
        "checkpoint_path": result["checkpoint_path"],
        "dataset_root": cfg.dataset.root,
        "metadata_file": cfg.dataset.metadata_file,
        "num_query": len(dataset_query),
        "num_database": len(dataset_database),
    }
    csv_row.update({k: float(v) for k, v in metrics.items()})
    csv_row.update({k: float(v) for k, v in timings.items()})
    append_csv_row(Path(cfg.output.csv_path), csv_row)

    print("Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.6f}")
    print(f"Saved JSON: {output_json}")
    print(f"Appended CSV row: {cfg.output.csv_path}")
    if visuals:
        print(f"Saved {len(visuals)} visualizations under {Path(cfg.visualization.dir) / run_id}")

    if wandb_run is not None:
        wandb_run.log(
            {
                "method": method,
                "model_type": cfg.model.type,
                "model_mode": cfg.model.mode,
                "no_background": bool(cfg.dataset.no_background),
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
        wandb_run.summary["split_col"] = cfg.dataset.split_col
        wandb_run.summary["database_split_value"] = cfg.dataset.database_split_value
        wandb_run.summary["query_split_value"] = cfg.dataset.query_split_value
        wandb_run.finish()
