"""The metric contract: a pure function from aligned reference/estimate to numbers.

A ``Metric`` receives two frame-aligned pitch arrays and returns a
``MetricResult``. It does no resampling, no voicing inference, no aggregation --
``FrameEvaluation`` has already done all of that (see ``stemify.evaluation.frame``).
That is the whole point of the split: metrics stay stateless and comparable, and
alignment is implemented once rather than once per metric.

The contract for ``calculate(true, pred)``:

  - both arrays are 1-D, the same length, and share a common time grid
  - values are frequencies in Hz, with **0.0 meaning unvoiced**
  - the grid is the *reference's*, which is the mir_eval convention -- the
    reference defines where questions get asked, the estimate is resampled onto
    it

Aggregation across a dataset is *not* a metric concern. ``FrameEvaluation`` pools frames
from every track into one pair of arrays and calls ``calculate`` once, which
makes results frame-weighted by construction. Per-group breakdowns (per track,
per string) are the same call on a boolean-masked subset. There is deliberately
no ``combine`` step to get wrong.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    """What a metric computed: named scalars, plus any non-scalar artifacts.

    ``scalars`` is what gets printed, logged, and compared across runs.
    ``arrays`` carries anything that is not a single number -- a confusion
    matrix, a histogram -- for plotting or further analysis. Keeping them apart
    means a caller can always treat ``scalars`` as a flat comparable record
    regardless of which metrics produced it.
    """

    name: str
    scalars: dict[str, float] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def render(self) -> str:
        """Human-readable summary. Subclasses override for domain formatting."""
        return "\n".join(f"  {k:<28} {v:>9.4g}" for k, v in self.scalars.items())


class Metric(ABC):
    """Computes one family of numbers from an aligned (reference, estimate) pair."""

    name: str

    @abstractmethod
    def calculate(self, true: np.ndarray, pred: np.ndarray) -> MetricResult:
        """Score aligned reference vs. estimate. See the module docstring for the
        contract both arrays must satisfy."""

    def _check(self, true: np.ndarray, pred: np.ndarray) -> None:
        """Enforce the alignment contract. Cheap, and the failure it catches --
        silently comparing misaligned grids -- produces plausible-looking
        numbers that are entirely meaningless."""
        if true.shape != pred.shape:
            raise ValueError(
                f"{self.name}: expected aligned arrays, got {true.shape} vs "
                f"{pred.shape}. Align with stemify.evaluation.alignment first."
            )
        if true.ndim != 1:
            raise ValueError(f"{self.name}: expected 1-D arrays, got {true.ndim}-D")
