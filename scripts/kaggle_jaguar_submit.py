import argparse
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle Jaguar Re-ID: finetune + RDD submission pipeline")
    parser.add_argument("--config", type=str, default="config/kaggle_jaguar.yaml", help="Path to YAML config")
    parser.add_argument("--data-dir", type=str, default=None, help="Kaggle dataset directory")
    parser.add_argument("--dry-run", action="store_true", help="Run only on first N pairs")
    parser.add_argument("--pair-limit", type=int, default=None, help="Dry-run pair limit")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode (smaller candidate_k/top_k)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Use an explicit finetuned checkpoint path")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> DictConfig:
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    if args.data_dir is not None:
        cfg.kaggle.data_dir = args.data_dir
    if args.dry_run:
        cfg.kaggle.dry_run.enabled = True
    if args.pair_limit is not None:
        cfg.kaggle.dry_run.pair_limit = int(args.pair_limit)
    if args.fast:
        cfg.kaggle.fast_mode = True
    if args.checkpoint is not None:
        cfg.finetune.checkpoint_path = args.checkpoint
        cfg.finetune.run_finetune = False

    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    from reid.engine.kaggle_jaguar_runner import run_kaggle_jaguar

    run_kaggle_jaguar(cfg)


if __name__ == "__main__":
    main()
