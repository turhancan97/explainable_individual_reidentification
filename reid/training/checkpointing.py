from pathlib import Path
from typing import Any, Optional

import torch


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
