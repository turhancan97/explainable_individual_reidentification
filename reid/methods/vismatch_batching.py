"""Dependency-light planning and retry helpers for batched Vismatch work."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Iterator, Sequence, TypeVar


T = TypeVar("T")
R = TypeVar("R")


def iter_chunks(items: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    """Yield non-empty slices, including a final partial slice."""

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def feature_shape_key(left: object, right: object) -> tuple[int, int]:
    """Return the keypoint cardinalities required for safe feature batching."""

    left_keypoints = getattr(left, "keypoints")
    right_keypoints = getattr(right, "keypoints")
    return int(left_keypoints.shape[0]), int(right_keypoints.shape[0])


def grouped_pair_batches(
    pairs: Iterable[tuple[int, int]],
    query_features: Sequence[object],
    database_features: Sequence[object],
    batch_size: int,
) -> Iterator[list[tuple[int, int]]]:
    """Group candidate pairs by exact feature shape and split them into batches.

    Exact cardinality grouping is intentional. LoMa's assignment softmax does not
    accept a padding mask, so padding variable-length features would change scores.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    grouped: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for query_index, database_index in pairs:
        grouped[
            feature_shape_key(
                query_features[query_index], database_features[database_index]
            )
        ].append((int(query_index), int(database_index)))
    for group in grouped.values():
        yield from (list(chunk) for chunk in iter_chunks(group, batch_size))


def is_cuda_oom(error: BaseException) -> bool:
    """Recognize CUDA OOM errors without importing torch in unit-test helpers."""

    error_name = type(error).__name__.lower().replace("_", "")
    message = str(error).lower()
    return "outofmemory" in error_name or (
        "out of memory" in message and "cuda" in message
    )


def run_with_batch_backoff(
    items: Sequence[T],
    batch_size: int,
    process_batch: Callable[[Sequence[T]], R],
    *,
    oom_backoff: bool,
    is_oom: Callable[[BaseException], bool] = is_cuda_oom,
    clear_memory: Callable[[], None] | None = None,
) -> Iterator[tuple[R, int]]:
    """Process items in batches, halving the size after a configured OOM.

    The yielded second value is the effective batch size used for that batch.
    The reduced size is retained for all subsequent batches in the same call.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    effective_size = int(batch_size)
    position = 0
    while position < len(items):
        current = items[position : position + effective_size]
        try:
            result = process_batch(current)
        except BaseException as error:
            if not oom_backoff or not is_oom(error) or effective_size <= 1:
                raise
            effective_size = max(1, effective_size // 2)
            if clear_memory is not None:
                clear_memory()
            continue
        yield result, effective_size
        position += len(current)
