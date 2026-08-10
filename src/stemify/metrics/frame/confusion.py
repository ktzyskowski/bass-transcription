"""Where the errors actually go: a pitch confusion matrix and its semitone-delta
summary.

RPA and RCA say *how much* is wrong and how much of that is octaves. This says
*what* is wrong. The full matrix goes in ``arrays`` for plotting; what gets
printed is the histogram of ``pred - true`` in semitones, because on bass the
errors are structured and that histogram makes the structure legible in one
glance:

  -  ``0``  correct
  - ``+12`` locked onto the 2nd harmonic (one octave up) -- the dominant bass
    failure, and the one already seen at ~46% of raw CQT peaks
  - ``+19`` locked onto the 3rd harmonic (octave + fifth) -- what the probe on a
    low F1 note showed PESTO doing
  - ``-12`` sub-octave, usually a tracker halving a period it could not resolve
  - scattered non-harmonic deltas -- genuine confusion rather than harmonic
    locking, and the only part that argues for more data

A tall spike at +12 or +19 next to a near-empty rest of the histogram means the
model is not guessing; it is tracking a real partial of the right note. That
distinction drives entirely different fixes.
"""

from dataclasses import dataclass, field

import librosa
import numpy as np

from stemify.metrics.base import MetricResult
from stemify.metrics.frame.base import PitchMetric

# The physical range of a 5-string bass with 24 frets: B0 (open low B) to G4
# (24th fret of the G string). Chosen from instrument physics rather than fitted
# to a dataset, but it does cover 99.97% of MDB-bass-synth's 3.6M reference
# frames -- and widening further to 21-72 covers no more, so the extra classes
# would be dead weight.
#
# This replaces an earlier 28-56 (4-string, 12th fret) which excluded 1.75% of
# MDB and would have been badly wrong for Slakh, whose bass MIDI reaches 67.
#
# The bounds are deliberately *fixed* rather than derived per dataset: a
# confusion matrix whose axes change between runs cannot be compared across
# them. Anything outside is counted in ``out_of_range_rate`` rather than clipped
# into the edge classes, so a mismatch stays visible. The delta histogram --
# the diagnostic that actually matters -- spans every answered frame regardless
# of these bounds.
MIDI_MIN = 23
MIDI_MAX = 67

HARMONIC_LABELS = {
    0: "correct",
    12: "2nd harmonic (octave up)",
    19: "3rd harmonic (octave+5th)",
    24: "4th harmonic (2 octaves)",
    28: "5th harmonic",
    -12: "sub-octave",
}


def _empty_int() -> np.ndarray:
    return np.array([], dtype=int)


@dataclass(frozen=True)
class ConfusionResult(MetricResult):
    name = "pitch-confusion"
    SCALARS = (
        "n_voiced",
        "unvoiced_rate",
        "out_of_range_rate",
        "reference_out_of_range_rate",
        "exact_rate",
        "octave_up_rate",
    )

    n_voiced: float
    unvoiced_rate: float

    out_of_range_rate: float
    """Share of reference-voiced frames whose *prediction* fell outside the matrix.

    Attribution used to be wrong here. The rate was ``answered & ~in_range`` where
    ``in_range`` required **both** sides to be in bounds, so a reference pitch outside
    [``MIDI_MIN``, ``MIDI_MAX``] was reported under a line reading "predicted out of
    range" -- a frame whose prediction was exactly correct could score 1.0 for
    ``out_of_range_rate`` and 1.0 for ``exact_rate`` at the same time. This counts the
    estimate only; the reference's own excursions are ``reference_out_of_range_rate``.
    """

    reference_out_of_range_rate: float
    """Share of reference-voiced frames whose *reference* fell outside the matrix.

    A property of the corpus, not the model, which is exactly why it is reported
    apart: it is the fraction of frames the matrix structurally cannot hold, and if it
    grows the bounds are wrong for the dataset rather than the model being poor.
    """

    exact_rate: float
    octave_up_rate: float

    matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    midi_labels: np.ndarray = field(default_factory=_empty_int)
    delta_values: np.ndarray = field(default_factory=_empty_int)
    delta_counts: np.ndarray = field(default_factory=_empty_int)

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "matrix": self.matrix,
            "midi_labels": self.midi_labels,
            "delta_values": self.delta_values,
            "delta_counts": self.delta_counts,
        }

    def render(self) -> str:
        total = self.delta_counts.sum()
        lines = [
            f"  frames compared              {total:>9,.0f}",
            f"  predicted unvoiced           {self.unvoiced_rate:>9.1%}",
            f"  predicted out of range       {self.out_of_range_rate:>9.1%}",
            f"  reference out of range       {self.reference_out_of_range_rate:>9.1%}",
            "",
            f"  {'delta':>6} {'share':>8}  interpretation",
        ]
        # Most common deltas first -- the shape of the error, not every bin.
        for i in np.argsort(self.delta_counts)[::-1][:8]:
            if self.delta_counts[i] == 0:
                continue
            d = int(self.delta_values[i])
            lines.append(
                f"  {d:>+6} {self.delta_counts[i] / total:>8.1%}  "
                f"{HARMONIC_LABELS.get(d, '')}"
            )
        return "\n".join(lines)


class PitchConfusion(PitchMetric):
    name = "pitch-confusion"

    def __init__(self, midi_min: int = MIDI_MIN, midi_max: int = MIDI_MAX) -> None:
        self.midi_min = midi_min
        self.midi_max = midi_max

    def compute(self, true: np.ndarray, pred: np.ndarray) -> ConfusionResult:
        self._check(true, pred)

        # Scored over reference-voiced frames only, matching RPA/RCA.
        voiced = true > 0
        true_midi = np.round(librosa.hz_to_midi(true[voiced])).astype(int)
        pred_hz = pred[voiced]

        answered = pred_hz > 0
        pred_midi = np.full(pred_hz.shape, np.nan)
        pred_midi[answered] = np.round(librosa.hz_to_midi(pred_hz[answered]))

        n = int(voiced.sum())
        n_classes = self.midi_max - self.midi_min + 1
        matrix = np.zeros((n_classes, n_classes), dtype=int)

        # Kept apart, because they answer different questions: one is the model's
        # error, the other is the corpus outrunning the matrix. Conflating them is
        # what made a perfect prediction score 100% "predicted out of range".
        pred_in_range = (
            answered & (pred_midi >= self.midi_min) & (pred_midi <= self.midi_max)
        )
        ref_in_range = (true_midi >= self.midi_min) & (true_midi <= self.midi_max)

        # A cell needs both axes, so the matrix is the one place the conjunction is
        # the right question.
        cell = pred_in_range & ref_in_range
        np.add.at(
            matrix,
            (
                true_midi[cell] - self.midi_min,
                pred_midi[cell].astype(int) - self.midi_min,
            ),
            1,
        )

        # The delta histogram spans every answered frame, in range or not --
        # a +19 harmonic error often lands above MIDI_MAX, and dropping it would
        # hide exactly the failure this metric exists to surface.
        delta = (pred_midi[answered] - true_midi[answered]).astype(int)
        values, counts = np.unique(delta, return_counts=True) if delta.size else (
            np.array([], dtype=int),
            np.array([], dtype=int),
        )

        def rate(mask: np.ndarray) -> float:
            return float(mask.sum() / n) if n else float("nan")

        return ConfusionResult(
            n_voiced=float(n),
            unvoiced_rate=rate(~answered),
            out_of_range_rate=rate(answered & ~pred_in_range),
            reference_out_of_range_rate=rate(~ref_in_range),
            exact_rate=rate(delta == 0),
            octave_up_rate=rate(delta == 12),
            matrix=matrix,
            midi_labels=np.arange(self.midi_min, self.midi_max + 1),
            delta_values=values,
            delta_counts=counts,
        )
