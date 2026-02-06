from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from timm.data import create_transform
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from wildlife_tools.data import WildlifeDataset as WildlifeDatasetWildlifeTools
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.train.trainer import set_seed

from models.model import get_model
from models.objective import ArcFaceLoss
from reid.data.dataset_view import BenchmarkDatasetView
from reid.evaluation.metrics import compute_metrics
from reid.features.containers import FeatureContainer, get_labels_string, normalize_features
from reid.training.checkpointing import load_full_checkpoint, save_full_checkpoint
from reid.utils.io import append_csv_row, ensure_dir, ensure_file
from reid.utils.repro import set_reproducible


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def run_finetune(cfg: DictConfig) -> None:
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

    model, embedding_size, mean, std, img_size, arch, patch_size, number_of_patches = get_model(cfg.model.type)
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
