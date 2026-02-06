import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pycocotools import mask as mask_utils
from timm.data import create_transform
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from wildlife_tools.data import WildlifeDataset as WildlifeDatasetWildlifeTools
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.train import ArcFaceLoss
from wildlife_tools.train.trainer import set_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import get_model


@dataclass
class FeatureContainer:
    features: np.ndarray
    labels_string: Optional[np.ndarray] = None


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


DEFAULT_CONFIG = {
    "dataset": {
        "root": "/shared/sets/datasets/vision/czechlynx/CzechLynx_v2",
        "metadata_file": "CzechLynxDataset-Metadata-Real.csv",
        "label_col": "unique_name",
        "mask_col": "mask",
        "no_background": False,
        "split_col": "split-time_closed",
        "train_split_value": "train",
        "val_split_value": "test",
    },
    "model": {
        "type": "megadescriptor",
    },
    "train": {
        "seed": 0,
        "epochs": 20,
        "batch_size": 16,
        "num_workers": 4,
        "accumulation_steps": 4,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "amp": "auto",
        "deterministic": True,
        "resume_checkpoint": None,
        "log_every": 1,
    },
    "loss": {
        "margin": 0.25,
        "scale": 16.0,
    },
    "scheduler": {
        "type": "cosine",
        "eta_min_scale": 1e-3,
    },
    "output": {
        "run_dir": "results",
        "save_every": 10,
        "save_best": True,
        "best_metric": "top_1",
        "csv_path": "results/train_metrics.csv",
    },
    "benchmark": {
        "top_k": [1, 5, 10],
        "compute_map": True,
        "val_batch_size": 32,
        "val_num_workers": 4,
    },
    "wandb": {
        "enabled": False,
        "project": "explainable-reid",
        "entity": None,
        "group": None,
        "tags": [],
        "name": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune backbone on ReID dataset")
    parser.add_argument("--config", type=str, default="train/finetune_config.yaml", help="Path to YAML config")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--metadata-file", type=str, default=None)
    parser.add_argument("--no-background", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> DictConfig:
    cfg = OmegaConf.create(DEFAULT_CONFIG)
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg_path))

    if args.dataset_root is not None:
        cfg.dataset.root = args.dataset_root
    if args.metadata_file is not None:
        cfg.dataset.metadata_file = args.metadata_file
    if args.no_background:
        cfg.dataset.no_background = True
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.resume is not None:
        cfg.train.resume_checkpoint = args.resume
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    return cfg


def ensure_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def ensure_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} not found: {path}")


def set_reproducible(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_labels_string(dataset: Any, label_col: str) -> np.ndarray:
    if hasattr(dataset, "labels_string"):
        labels = getattr(dataset, "labels_string")
        return np.asarray(labels)
    if hasattr(dataset, "df") and label_col in dataset.df.columns:
        return dataset.df[label_col].astype(str).to_numpy()
    if hasattr(dataset, "metadata") and label_col in dataset.metadata.columns:
        return dataset.metadata[label_col].astype(str).to_numpy()
    return np.array([], dtype=str)


def compute_metrics(
    dataset_query: Any,
    dataset_database: Any,
    similarity: np.ndarray,
    top_k_values: List[int],
    compute_map: bool,
) -> Dict[str, float]:
    if similarity.shape != (len(dataset_query), len(dataset_database)):
        raise ValueError(
            f"Invalid similarity shape {similarity.shape}, expected {(len(dataset_query), len(dataset_database))}"
        )

    ranked_idx = np.argsort(similarity, axis=1)[:, ::-1]
    query_labels = dataset_query.df[dataset_query.col_label].to_numpy()
    db_labels = dataset_database.df[dataset_database.col_label].to_numpy()

    metrics: Dict[str, float] = {}
    max_k = max(top_k_values)
    if max_k > len(dataset_database):
        raise ValueError(f"Requested top-k includes {max_k}, but database has only {len(dataset_database)} samples")

    for k in top_k_values:
        hits = []
        for q_idx in range(len(dataset_query)):
            top_db_idx = ranked_idx[q_idx, :k]
            top_labels = db_labels[top_db_idx]
            hits.append(query_labels[q_idx] in top_labels)
        metrics[f"top_{k}"] = float(np.mean(hits))

    if compute_map:
        aps: List[float] = []
        for q_idx in range(len(dataset_query)):
            relevant = (db_labels[ranked_idx[q_idx]] == query_labels[q_idx]).astype(np.float32)
            n_relevant = int(relevant.sum())
            if n_relevant == 0:
                continue
            cum_rel = np.cumsum(relevant)
            ranks = np.arange(1, len(relevant) + 1, dtype=np.float32)
            precision = cum_rel / ranks
            ap = float((precision * relevant).sum() / n_relevant)
            aps.append(ap)
        metrics["mAP"] = float(np.mean(aps)) if aps else float("nan")

    return metrics


def append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.is_file()
    fieldnames = sorted(row.keys())
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def normalize_features(raw: Any) -> np.ndarray:
    value = raw
    if isinstance(value, dict):
        for key in ("features", "embeddings", "vectors", "x"):
            if key in value:
                value = value[key]
                break

    if isinstance(value, tuple) and len(value) >= 1:
        first = value[0]
        if isinstance(first, (np.ndarray, list, tuple)) or torch.is_tensor(first):
            value = first

    def _to_numpy(v: Any) -> np.ndarray:
        if torch.is_tensor(v):
            return v.detach().cpu().numpy()
        return np.asarray(v)

    def _extract_from_sequence_rows(seq: Any) -> Optional[np.ndarray]:
        if not isinstance(seq, (list, tuple)) or len(seq) == 0:
            return None
        first = seq[0]
        if isinstance(first, (list, tuple)) and len(first) >= 1:
            first_item = first[0]
            if isinstance(first_item, (np.ndarray, list, tuple)) or torch.is_tensor(first_item):
                rows = [_to_numpy(row[0]) for row in seq]
                return np.stack(rows, axis=0)
        return None

    maybe_rows = _extract_from_sequence_rows(value)
    if maybe_rows is not None:
        arr = maybe_rows
    elif isinstance(value, (list, tuple)):
        rows = [_to_numpy(v) for v in value]
        try:
            arr = np.stack(rows, axis=0)
        except ValueError:
            arr = np.asarray(rows, dtype=np.float32)
    elif torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        try:
            arr = np.asarray(value)
        except ValueError:
            try:
                seq = list(value)
            except Exception as exc:
                raise ValueError(f"Could not convert features to numpy. type={type(raw)}") from exc
            maybe_rows = _extract_from_sequence_rows(seq)
            if maybe_rows is None:
                raise
            arr = maybe_rows

    if arr.dtype == object:
        arr = np.stack([np.asarray(x) for x in arr], axis=0)

    if arr.ndim == 1:
        raise ValueError(f"Features must be 2D (N, D). Got shape={arr.shape}")
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    return arr.astype(np.float32, copy=False)


def train_one_epoch(
    model: Any,
    objective: Any,
    optimizer: Any,
    scheduler: Any,
    loader: DataLoader,
    device: torch.device,
    accumulation_steps: int,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler,
) -> Dict[str, float]:
    model.train()
    objective.train()
    losses: List[float] = []

    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        if amp_enabled:
            with torch.amp.autocast():
                out = model(x)
                loss = objective(out, y)
            scaler.scale(loss).backward()
        else:
            out = model(x)
            loss = objective(out, y)
            loss.backward()

        if (i + 1) % accumulation_steps == 0:
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().cpu()))

    if scheduler:
        scheduler.step()

    return {"train_loss_epoch_avg": float(np.mean(losses)) if losses else float("nan")}


def save_full_checkpoint(
    path: Path,
    model: Any,
    objective: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
) -> None:
    state = {
        "model": model.state_dict(),
        "objective": objective.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)


def load_full_checkpoint(
    path: Path,
    model: Any,
    objective: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Optional[torch.amp.GradScaler],
) -> int:
    state = torch.load(path, map_location="cpu")
    if "model" in state:
        model.load_state_dict(state["model"])
    if "objective" in state:
        objective.load_state_dict(state["objective"])
    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if "scheduler" in state and scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if "scaler" in state and scaler is not None:
        scaler.load_state_dict(state["scaler"])
    return int(state.get("epoch", 0))

def evaluate(
    model: Any,
    device: torch.device,
    dataset_query: Any,
    dataset_database: Any,
    batch_size: int,
    num_workers: int,
    top_k: List[int],
    compute_map: bool,
) -> Dict[str, float]:
    original_device = next(model.parameters()).device
    extractor = DeepFeatures(model, device=device, batch_size=batch_size, num_workers=num_workers)
    features_database = normalize_features(extractor(dataset_database))
    features_query = normalize_features(extractor(dataset_query))
    similarity = CosineSimilarity()(
        FeatureContainer(features=features_query, labels_string=get_labels_string(dataset_query, dataset_query.col_label)),
        FeatureContainer(features=features_database, labels_string=get_labels_string(dataset_database, dataset_database.col_label)),
    )
    if original_device != device:
        model.to(original_device)
    return compute_metrics(dataset_query, dataset_database, np.asarray(similarity), top_k, compute_map)


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    set_reproducible(int(cfg.train.seed), bool(cfg.train.deterministic))
    device = choose_device()

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

    model, embedding_size, mean, std, img_size = get_model(cfg.model.type)
    model.to(device)

    transform_display = T.Compose([T.Resize([img_size, img_size])])
    transform_train = create_transform(input_size=img_size, is_training=True, auto_augment="rand-m10-n2-mstd1")
    transform_val = T.Compose([*transform_display.transforms, T.ToTensor(), T.Normalize(mean=mean, std=std)])

    train_metadata = metadata[metadata[cfg.dataset.split_col] == cfg.dataset.train_split_value]
    val_metadata = metadata[metadata[cfg.dataset.split_col] == cfg.dataset.val_split_value]

    dataset_train_raw = WildlifeDatasetWildlifeTools(
        root=str(root),
        metadata=train_metadata,
        transform=None,
        col_label=cfg.dataset.label_col,
    )
    dataset_val_raw = WildlifeDatasetWildlifeTools(
        root=str(root),
        metadata=val_metadata,
        transform=None,
        col_label=cfg.dataset.label_col,
    )

    dataset_train = BenchmarkDatasetView(
        base_dataset=dataset_train_raw,
        label_col=cfg.dataset.label_col,
        transform=transform_train,
        no_background=bool(cfg.dataset.no_background),
        mask_col=cfg.dataset.mask_col,
    )
    dataset_val = BenchmarkDatasetView(
        base_dataset=dataset_val_raw,
        label_col=cfg.dataset.label_col,
        transform=transform_val,
        no_background=bool(cfg.dataset.no_background),
        mask_col=cfg.dataset.mask_col,
    )

    objective = ArcFaceLoss(
        num_classes=dataset_train_raw.num_classes,
        embedding_size=embedding_size,
        margin=float(cfg.loss.margin),
        scale=float(cfg.loss.scale),
    )

    optimizer = AdamW(
        params=list(model.parameters()) + list(objective.parameters()),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg.train.epochs), eta_min=float(cfg.train.lr) * float(cfg.scheduler.eta_min_scale))

    run_started = datetime.utcnow()
    run_id = run_started.strftime("run_%Y%m%d_%H%M%S")
    output_folder = Path(cfg.output.run_dir) / run_id
    output_folder.mkdir(parents=True, exist_ok=True)

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

    amp_enabled = bool(cfg.train.amp == "auto" and device.type == "cuda") or bool(cfg.train.amp is True)
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 0
    if cfg.train.resume_checkpoint:
        resume_path = Path(cfg.train.resume_checkpoint)
        ensure_file(resume_path, "Resume checkpoint")
        start_epoch = load_full_checkpoint(resume_path, model, objective, optimizer, scheduler, scaler)

    best_metric_name = str(cfg.output.best_metric)
    best_metric_value = -float("inf")

    train_loader = DataLoader(
        dataset_train,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        shuffle=True,
    )

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        set_seed(int(cfg.train.seed) + epoch)
        train_metrics = train_one_epoch(
            model=model,
            objective=objective,
            optimizer=optimizer,
            scheduler=scheduler,
            loader=train_loader,
            device=device,
            accumulation_steps=int(cfg.train.accumulation_steps),
            amp_enabled=amp_enabled,
            scaler=scaler,
        )

        metrics = evaluate(
            model=model,
            device=device,
            dataset_query=dataset_val,
            dataset_database=dataset_train,
            batch_size=int(cfg.benchmark.val_batch_size),
            num_workers=int(cfg.benchmark.val_num_workers),
            top_k=[int(k) for k in cfg.benchmark.top_k],
            compute_map=bool(cfg.benchmark.compute_map),
        )
        model.to(device)
        objective.to(device)

        metric_value = float(metrics.get(best_metric_name, -float("inf")))
        if metric_value > best_metric_value and bool(cfg.output.save_best):
            best_metric_value = metric_value
            torch.save(model.state_dict(), output_folder / "checkpoint-best.pth")
            save_full_checkpoint(
                output_folder / "checkpoint-best-full.pth",
                model,
                objective,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
            )

        if (epoch + 1) % int(cfg.output.save_every) == 0:
            torch.save(model.state_dict(), output_folder / f"checkpoint-epoch-{epoch+1}.pth")
        save_full_checkpoint(
            output_folder / "checkpoint-latest-full.pth",
            model,
            objective,
            optimizer,
            scheduler,
            scaler,
            epoch + 1,
        )

        row = {
            "run_id": run_id,
            "epoch": epoch + 1,
            "best_metric": best_metric_name,
            "best_metric_value": best_metric_value,
            "no_background": bool(cfg.dataset.no_background),
        }
        row.update(train_metrics)
        row.update(metrics)
        append_csv_row(Path(cfg.output.csv_path), row)

        if wandb_run is not None:
            lr = float(optimizer.param_groups[0].get("lr", 0.0))
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss_epoch_avg": row.get("train_loss_epoch_avg"),
                    "lr": lr,
                    **metrics,
                },
                step=epoch + 1,
            )

        if (epoch + 1) % int(cfg.train.log_every) == 0:
            metrics_str = " ".join([f"{k}={metrics[k]:.6f}" for k in metrics])
            print(
                f"Epoch {epoch+1}: "
                f"train_loss={row.get('train_loss_epoch_avg', float('nan')):.6f} "
                f"{metrics_str}"
            )

    torch.save(model.state_dict(), output_folder / "checkpoint-final.pth")
    save_full_checkpoint(
        output_folder / "checkpoint-final-full.pth",
        model,
        objective,
        optimizer,
        scheduler,
        scaler,
        int(cfg.train.epochs),
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
