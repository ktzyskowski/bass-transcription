"""The one representation both sides of an evaluation speak.

A reference and an estimate are the same kind of object -- a pitch over time --
and they were previously modelled two different ways: estimates got a structured
``F0Contour``, references got a bare ``tuple[ndarray, ndarray]``. That asymmetry
is why alignment used to take four positional arrays and why every dataset
loader repeated the same return annotation.

``freqs`` follows the mir_eval convention: 0.0 marks an unvoiced frame, so a
contour drops straight into ``mir_eval.melody`` without a separate voicing array.

``confidence`` is optional because only estimators have one. A reference is
ground truth; there is nothing to be uncertain about.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Contour:
    """A pitch track: frequency per frame, with 0.0 meaning unvoiced."""

    times: np.ndarray  # seconds
    freqs: np.ndarray  # Hz, 0.0 where unvoiced
    confidence: np.ndarray | None = None  # 0..1, estimates only

    def __post_init__(self) -> None:
        if self.times.shape != self.freqs.shape:
            raise ValueError(
                f"times and freqs must be the same length, got "
                f"{self.times.shape} and {self.freqs.shape}"
            )

    def __len__(self) -> int:
        return int(self.times.size)

    @property
    def voiced(self) -> np.ndarray:
        return self.freqs > 0

    @property
    def is_silent(self) -> bool:
        """True when nothing sounds anywhere -- there is no question to ask."""
        return not self.voiced.any()

    @classmethod
    def empty(cls) -> "Contour":
        return cls(np.zeros(0), np.zeros(0))

    @classmethod
    def constant(cls, hz: float, times: np.ndarray, voiced: np.ndarray) -> "Contour":
        """One pitch held wherever ``voiced`` is true.

        This is what makes a single-note prediction expressible as a contour: a
        baseline that only answers "the pitch of this clip is X" produces a
        degenerate contour holding X across the clip, and is then scored by
        exactly the same metrics as a frame-level tracker.
        """
        return cls(times=times, freqs=np.where(voiced, hz, 0.0))
