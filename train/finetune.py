import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@hydra.main(version_base="1.3", config_path="../conf", config_name="finetune")
def main(cfg: DictConfig) -> None:
    """Run one finetuning experiment with Hydra configuration overrides."""
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    from reid.engine.finetune_runner import run_finetune

    run_finetune(cfg)


if __name__ == "__main__":
    main()
