import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


_REMOVED_PROBE_BUDGET_KEYS = {
    "benchmark.map_at_k",
    "benchmark.methods.local_lightglue.B",
    "benchmark.methods.vismatch.candidate_k",
    "benchmark.methods.wildfusion.B",
}


def reject_removed_probe_budget_overrides(argv=None):
    for argument in sys.argv[1:] if argv is None else argv:
        key = argument.split("=", 1)[0]
        if key in _REMOVED_PROBE_BUDGET_KEYS:
            raise ValueError(
                f"'{key}' is no longer configurable; use benchmark.candidate_k instead."
            )


@hydra.main(version_base="1.3", config_path="../conf", config_name="probe")
def main(cfg: DictConfig) -> None:
    """Run a single retrieval benchmark with Hydra configuration overrides."""
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    from reid.engine.probe_runner import run_probe

    run_probe(cfg)


if __name__ == "__main__":
    reject_removed_probe_budget_overrides()
    main()
