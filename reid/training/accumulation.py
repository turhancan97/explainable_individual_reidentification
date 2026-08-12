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
