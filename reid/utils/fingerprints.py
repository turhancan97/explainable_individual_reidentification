"""Content fingerprints for split validation, caches, and manifests."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional


HASH_ALGORITHM = "sha256"

# Split safety checks, dataset digests, and Vismatch cache keys each hash every image
# once, so a probe run reads the entire dataset three times. Memoization is opt-in and
# scoped to a run rather than a silent process-wide default: it is only sound while the
# dataset files are known to be stable, and stale digests would defeat the very
# content-addressing these hashes exist to provide.
_FILE_DIGEST_CACHE: Optional[dict[tuple[int, int, int, int], str]] = None


@contextmanager
def file_digest_cache() -> Iterator[dict[tuple[int, int, int, int], str]]:
    """Memoize :func:`sha256_file` for the duration of one run.

    Callers assert that the files being hashed do not change while the block is
    active, which holds for a probe or finetune run over a read-only dataset. Nested
    blocks share the outermost cache. Outside such a block every call re-reads the
    file, so library and test behaviour is unchanged.
    """
    global _FILE_DIGEST_CACHE
    previous = _FILE_DIGEST_CACHE
    cache = previous if previous is not None else {}
    _FILE_DIGEST_CACHE = cache
    try:
        yield cache
    finally:
        _FILE_DIGEST_CACHE = previous


def _file_identity_key(path: Path) -> Optional[tuple[int, int, int, int]]:
    """Identify a file by inode plus size and modification time.

    Device and inode identify the file regardless of how the path was spelled, so the
    three call sites share entries even though each builds its paths differently.
    Size and mtime are a best-effort staleness signal only: some filesystems, tmpfs
    among them, reuse one ``st_mtime_ns`` for rapid same-size rewrites, which is why
    caching is confined to an explicit run scope. Filesystems without a usable inode
    are left uncached.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    if not stat.st_ino:
        return None
    return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))


def _sha256_file_uncached(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    resolved = Path(path)
    cache = _FILE_DIGEST_CACHE
    if cache is None:
        return _sha256_file_uncached(resolved, chunk_size)
    key = _file_identity_key(resolved)
    if key is None:
        return _sha256_file_uncached(resolved, chunk_size)
    cached = cache.get(key)
    if cached is not None:
        return cached
    digest = _sha256_file_uncached(resolved, chunk_size)
    cache[key] = digest
    return digest


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
