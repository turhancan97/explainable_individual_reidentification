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
    """Find a model-only checkpoint, preferring completed modern runs.

    Modern experiment runs may be nested several levels below the search root;
    legacy one-level result directories remain supported.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if _is_full_checkpoint_name(str(filename)):
        raise ValueError(
            f"Model-only inference cannot search for a full checkpoint: {filename}"
        )

    candidate_names = [str(filename)]
    if str(filename) != "checkpoint-final.pth":
        candidate_names.append("checkpoint-final.pth")

    candidates = []
    for candidate in results_dir.rglob("checkpoint*.pth"):
        if not candidate.is_file() or _is_full_checkpoint_name(candidate.name):
            continue
        canonical_rank = 0 if candidate.name in candidate_names else 1
        if canonical_rank == 1 and not candidate.name.startswith("checkpoint-final_"):
            continue
        manifest = candidate.parent / "run_manifest.json"
        completed = False
        if manifest.is_file():
            try:
                import json

                completed = json.loads(manifest.read_text()).get("status") == "completed"
            except (OSError, ValueError, TypeError):
                completed = False
            if not completed:
                continue
        elif candidate.parent.parent == results_dir:
            # Historical result directories have no manifest and are still valid.
            completed = True
        candidates.append(
            (
                0 if completed else 1,
                canonical_rank,
                -candidate.stat().st_mtime,
                str(candidate),
                candidate,
            )
        )
    if candidates:
        candidates.sort(key=lambda item: item[:4])
        return candidates[0][4]

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


def validate_resume_epochs(start_epoch: int, total_epochs: int, resume_path: Any) -> None:
    """Reject a resume that has no epochs left to run.

    A full checkpoint resumes at the epoch after the one it stored, so a checkpoint
    from a finished run leaves the training range empty. Training then silently
    produces no epochs while still writing final checkpoints and a completed manifest,
    which is almost always an unraised `train.epochs`. Failing here keeps that
    mistake visible instead of emitting a zero-epoch run.
    """
    start_epoch = int(start_epoch)
    total_epochs = int(total_epochs)
    if start_epoch < total_epochs:
        return
    raise ValueError(
        f"Resume checkpoint '{resume_path}' already completed {start_epoch} epochs, but "
        f"train.epochs is {total_epochs}, so there is nothing left to train. Set "
        f"train.epochs above {start_epoch} to continue training, or run train/probe.py "
        "to evaluate the checkpoint."
    )


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
