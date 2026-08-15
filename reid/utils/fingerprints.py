"""Content fingerprints for split validation, caches, and manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


HASH_ALGORITHM = "sha256"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(paths: Iterable[Path]) -> list[dict[str, str]]:
    """Return deterministic path/digest records for existing files."""
    records: list[dict[str, str]] = []
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        records.append({"path": resolved.as_posix(), "sha256": sha256_file(resolved)})
    return records


def hash_mapping(value: Mapping[str, Any]) -> str:
    """Hash a JSON-serializable mapping with deterministic key ordering."""
    import json

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_state_dict(state_dict: Mapping[str, Any]) -> str:
    """Hash tensor-like state dictionaries without depending on torch imports."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        digest.update(str(name).encode("utf-8"))
        if hasattr(value, "detach"):
            value = value.detach().cpu().contiguous().numpy()
        if hasattr(value, "dtype"):
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(value.shape).encode("utf-8"))
            digest.update(value.tobytes())
        elif isinstance(value, bytes):
            digest.update(value)
        else:
            digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()


def model_fingerprint(model: Any, *, revision: Optional[str] = None) -> str:
    """Return a content fingerprint for a model plus optional source revision."""
    state_dict = model.state_dict() if hasattr(model, "state_dict") else {}
    return hash_mapping({"revision": revision or "unknown", "state": hash_state_dict(state_dict)})
