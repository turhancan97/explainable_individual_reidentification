from pathlib import Path
from typing import Any, Optional

import torch


def resolve_configured_model_checkpoint(
    explicit_path: Optional[Path],
    results_dir: Path,
    filename: str = "checkpoint-final.pth",
    fallback_dirs: Optional[list[Path]] = None,
) -> Path:
    """Resolve an explicit model path before searching result directories."""
    if explicit_path is not None:
        explicit = Path(explicit_path)
        if _is_full_checkpoint_name(explicit.name):
            raise ValueError(
                f"Model-only inference cannot use a full checkpoint: {explicit}"
            )
        return explicit
    search_dirs = [Path(results_dir), *(Path(path) for path in (fallback_dirs or []))]
    errors = []
    for search_dir in search_dirs:
        try:
            return resolve_model_checkpoint(results_dir=search_dir, filename=filename)
        except FileNotFoundError as exc:
            errors.append(str(exc))
    raise FileNotFoundError("No model-only checkpoint found. " + " | ".join(errors))


def _is_full_checkpoint_name(filename: str) -> bool:
    return filename.endswith("-full.pth") or "-full_" in filename


def resolve_model_checkpoint(results_dir: Path, filename: str = "checkpoint-final.pth") -> Path:
    """Find the newest model-only checkpoint, including legacy tagged names."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    run_dirs = [candidate for candidate in results_dir.iterdir() if candidate.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {results_dir}")
    run_dirs.sort(key=lambda candidate: candidate.stat().st_mtime, reverse=True)

    if _is_full_checkpoint_name(str(filename)):
        raise ValueError(
            f"Model-only inference cannot search for a full checkpoint: {filename}"
        )

    candidate_names = [str(filename)]
    if str(filename) != "checkpoint-final.pth":
        candidate_names.append("checkpoint-final.pth")

    for run_dir in run_dirs:
        for candidate_name in candidate_names:
            candidate = run_dir / candidate_name
            if candidate.is_file():
                return candidate

        tagged = sorted(run_dir.glob("checkpoint-final_*.pth"))
        if tagged:
            return tagged[-1]

    searched = ", ".join(candidate_names + ["checkpoint-final_<dataset_tag>.pth"])
    raise FileNotFoundError(
        f"No model-only checkpoint found under {results_dir}. Searched: {searched}"
    )


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
