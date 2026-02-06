import argparse
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reid.engine.finetune_runner import run_finetune


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
    parser.add_argument("--config", type=str, default="config/finetune_config.yaml", help="Path to YAML config")
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    run_finetune(cfg)


if __name__ == "__main__":
    main()
