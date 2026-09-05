"""Exact PPO rollout/evaluation budget helpers."""


def effective_timesteps(requested: int, epoch_size: int) -> int:
    """Round a requested rollout budget up to a complete PPO epoch."""
    if requested < 1 or epoch_size < 1:
        raise ValueError("requested timesteps and epoch size must be positive")
    return (requested + epoch_size - 1) // epoch_size * epoch_size


def compatible_num_evals(
    effective_steps: int, epoch_size: int, requested_num_evals: int
) -> int:
    """Select an eval cadence that cannot expand the training budget.

    Brax trains in ``num_evals - 1`` intervals and rounds each interval to a
    full epoch. The interval count must therefore divide the total epoch count.
    """
    if effective_steps < 1 or epoch_size < 1 or requested_num_evals < 2:
        raise ValueError("training budget, epoch size and eval count are invalid")
    if effective_steps % epoch_size:
        raise ValueError("effective timesteps must contain full PPO epochs")
    epochs = effective_steps // epoch_size
    requested_intervals = requested_num_evals - 1
    divisors = [candidate for candidate in range(1, epochs + 1) if epochs % candidate == 0]
    intervals = min(
        divisors,
        key=lambda candidate: (
            abs(candidate - requested_intervals),
            candidate < requested_intervals,
            candidate,
        ),
    )
    return intervals + 1
