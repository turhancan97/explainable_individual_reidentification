"""Helpers for gradient-accumulation scheduling."""


def should_step_accumulated_gradients(
    batch_index: int,
    total_batches: int,
    accumulation_steps: int,
) -> bool:
    """Whether the optimizer should step after a zero-based batch index.

    The final batch is always a step boundary so gradients from a partial
    accumulation group are not discarded at the end of an epoch.
    """
    if total_batches <= 0:
        return False
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be > 0")
    if batch_index < 0 or batch_index >= total_batches:
        raise ValueError(
            f"batch_index must be in [0, {total_batches}), got {batch_index}"
        )
    batch_number = batch_index + 1
    return batch_number % accumulation_steps == 0 or batch_number == total_batches


def accumulation_group_size(batch_index: int, total_batches: int, accumulation_steps: int) -> int:
    """Return the number of microbatches represented by this optimizer step."""
    if total_batches <= 0:
        return 0
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be > 0")
    if batch_index < 0 or batch_index >= total_batches:
        raise ValueError(
            f"batch_index must be in [0, {total_batches}), got {batch_index}"
        )
    group_start = (batch_index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, total_batches - group_start)
