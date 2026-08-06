"""``MetricResult``: the one data type every metric returns, frame or note.

A metric is a pure function from a (reference, estimate) pair to numbers. There
are two families, split by the shape of that pair, and each has its own base
class:

  - ``FrameMetric`` (``metrics.frame``) scores two frame-aligned arrays (Hz for
    pitch, or a label/probability pair for onset and voicing). Its caller flattens
    every track's frames into one pair -- frame scoring is elementwise, so
    concatenation is lossless -- and scores once.
  - ``NoteMetric`` (``metrics.note``) scores two *lists of per-track* note-event
    lists. Note matching is a per-track set assignment, not elementwise, so the
    track structure is load-bearing and cannot be flattened away; the metric
    loops tracks and sums counts internally, then scores once.

What they share is only this result type: named scalars plus any non-scalar
artifacts, uniform to render, serialise and compare regardless of which family
produced it. It is just data -- computation lives in the metrics, alignment and
estimation in the harnesses (``stemify.evaluation.cues`` / ``.note``).
"""

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
