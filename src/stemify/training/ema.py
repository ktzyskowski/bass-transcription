"""Exponential moving average of a scalar stream.

Pulled out of the logging callback on purpose: smoothing a noisy per-batch loss
is a numeric concern, not a tracking one, so it lives as a tiny testable unit the
callback composes rather than open-coding. ``beta`` is the retention -- each
update keeps ``beta`` of the running value and ``1 - beta`` of the new sample, so
higher beta smooths harder (0.98 ~= a 50-sample window). The first sample seeds
the average outright, so there is no zero-biased warmup to correct.
"""


class EMA:
    """Running exponential moving average; ``update`` folds in a sample and returns
    the current average. ``value`` is ``None`` until the first update."""

    def __init__(self, beta: float = 0.98) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        self.beta = beta
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.beta * self.value + (1 - self.beta) * x
        return self.value
