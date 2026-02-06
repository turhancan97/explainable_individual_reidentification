import argparse
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reid.engine.probe_runner import run_probe


DEFAULT_CONFIG = {
    "dataset": {
        "root": "/shared/sets/datasets/vision/czechlynx/CzechLynx_v2",
        "metadata_file": "CzechLynxDataset-Metadata-Real.csv",
        "label_col": "unique_name",
        "mask_col": "mask",
        "no_background": False,
        "split_col": "split-time_closed",
        "database_split_value": "train",
        "query_split_value": "test",
        "calibration_size": 100,
    },
    "model": {
        "type": "megadescriptor",
        "mode": "finetuned",
        "device": "auto",
        "batch_size": 32,
        "num_workers": 4,
        "checkpoint": {
            "path": None,
            "from_latest_results": True,
            "results_dir": "results",
            "filename": "checkpoint-final.pth",
        },
    },
    "benchmark": {
        "method": "cosine",
        "top_k": [1, 5, 10],
        "compute_map": True,
        "seed": 0,
        "deterministic": True,
        "cache": {
            "enabled": True,
            "dir": "cache/features",
            "format": "pt",
        },
        "methods": {
            "cosine": {},
            "wildfusion": {
                "B": 10,
                "deep_batch_size": 16,
                "deep_num_workers": 4,
                "local_batch_size": 16,
            },
            "local_lightglue": {
                "B": 10,
                "local_batch_size": 16,
            },
            "linear_probe": {
                "train_mode": "classifier",
                "epochs": 10,
                "log_every": 1,
                "batch_size": 64,
                "num_workers": 4,
                "accumulation_steps": 1,
                "optimizer": "sgd",
                "lr": 1e-3,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "eta_min_scale": 1e-3,
                "eval_batch_size": 128,
                "eval_num_workers": 4,
                "resume_checkpoint": None,
                "save_checkpoint": False,
                "save_every": 1,
                "final_checkpoint_name": "linear_probe_final.pth",
                "partial_rules": {
                    "default": ["layers.3", "norm"],
                },
            },
            "efficient_probe": {
                "train_mode": "classifier",
                "epochs": 10,
                "log_every": 1,
                "batch_size": 64,
                "num_workers": 4,
                "accumulation_steps": 1,
                "optimizer": "sgd",
                "lr": 1e-3,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "eta_min_scale": 1e-3,
                "dropout_rate": 0.0,
                "num_queries": 4,
                "d_out": 2,
                "eval_batch_size": 128,
                "eval_num_workers": 4,
                "resume_checkpoint": None,
                "save_checkpoint": False,
                "save_every": 1,
                "final_checkpoint_name": "efficient_probe_final.pth",
                "partial_rules": {
                    "default": ["layers.3", "norm"],
                },
            },
            "rdd": {
                "repo_dir": "/home/kargin/Projects/repositories/rdd",
                "config_path": "/home/kargin/Projects/repositories/rdd/configs/default.yaml",
                "weights": "/home/kargin/Projects/repositories/rdd/weights/RDD-v2.pth",
                "cache_dir": "cache/rdd_features",
                "device": "auto",
                "path_col": "path",
                "resize_max": 448,
                "top_k": 2048,
            },
        },
    },
    "visualization": {
        "enabled": False,
        "dir": "visualizations",
        "num_examples": 8,
        "rdd_max_matches": 200,
        "attention_num_examples": 8,
        "attention_average_queries": True,
        "top_k": 3,
    },
    "output": {
        "run_dir": "benchmark_runs",
        "csv_path": "benchmark_runs/benchmark_results.csv",
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
    parser = argparse.ArgumentParser(description="ReID benchmark probe runner")
    parser.add_argument("--config", type=str, default="config/probe_config.yaml", help="Path to YAML config")
    parser.add_argument("--method", type=str, default=None, help="Override benchmark method")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override dataset root")
    parser.add_argument("--metadata-file", type=str, default=None, help="Override metadata file name")
    parser.add_argument("--backbone-mode", type=str, default=None, choices=["pretrained", "finetuned"])
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Checkpoint path for finetuned mode")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> DictConfig:
    cfg = OmegaConf.create(DEFAULT_CONFIG)
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg_path))

    if args.method is not None:
        cfg.benchmark.method = args.method
    if args.dataset_root is not None:
        cfg.dataset.root = args.dataset_root
    if args.metadata_file is not None:
        cfg.dataset.metadata_file = args.metadata_file
    if args.backbone_mode is not None:
        cfg.model.mode = args.backbone_mode
    if args.checkpoint_path is not None:
        cfg.model.checkpoint.path = args.checkpoint_path
        cfg.model.checkpoint.from_latest_results = False
    if args.seed is not None:
        cfg.benchmark.seed = args.seed
    if args.visualize:
        cfg.visualization.enabled = True
    if args.no_visualize:
        cfg.visualization.enabled = False

    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    run_probe(cfg)


if __name__ == "__main__":
    main()
