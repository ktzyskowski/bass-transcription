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

import librosa
import numpy as np

from stemify.core.metrics.base import MetricResult
from stemify.core.metrics.frame.base import FrameMetric

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


class ConfusionResult(MetricResult):
    def render(self) -> str:
        s = self.scalars
        deltas = self.arrays["delta_values"]
        counts = self.arrays["delta_counts"]
        total = counts.sum()

        lines = [
            f"  frames compared              {total:>9,.0f}",
            f"  predicted unvoiced           {s['unvoiced_rate']:>9.1%}",
            f"  predicted out of range       {s['out_of_range_rate']:>9.1%}",
            "",
            f"  {'delta':>6} {'share':>8}  interpretation",
        ]
        # Most common deltas first -- the shape of the error, not every bin.
        for i in np.argsort(counts)[::-1][:8]:
            if counts[i] == 0:
                continue
            d = int(deltas[i])
            lines.append(
                f"  {d:>+6} {counts[i] / total:>8.1%}  {HARMONIC_LABELS.get(d, '')}"
            )
        return "\n".join(lines)


class PitchConfusion(FrameMetric):
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

        in_range = (
            answered
            & (pred_midi >= self.midi_min)
            & (pred_midi <= self.midi_max)
            & (true_midi >= self.midi_min)
            & (true_midi <= self.midi_max)
        )
        np.add.at(
            matrix,
            (
                true_midi[in_range] - self.midi_min,
                pred_midi[in_range].astype(int) - self.midi_min,
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

        return ConfusionResult(
            name=self.name,
            scalars={
                "n_voiced": float(n),
                "unvoiced_rate": float((~answered).sum() / n) if n else float("nan"),
                "out_of_range_rate": (
                    float((answered & ~in_range).sum() / n) if n else float("nan")
                ),
                "exact_rate": float((delta == 0).sum() / n) if n else float("nan"),
                "octave_up_rate": float((delta == 12).sum() / n) if n else float("nan"),
            },
            arrays={
                "matrix": matrix,
                "midi_labels": np.arange(self.midi_min, self.midi_max + 1),
                "delta_values": values,
                "delta_counts": counts,
            },
        )
