"""Raw pitch / chroma accuracy -- the octave-error diagnostic.

  - **RPA** (raw pitch accuracy): fraction of reference-voiced frames answered
    within 50 cents. Exactly right, octave included.
  - **RCA** (raw chroma accuracy): the same with octaves folded out. Right pitch
    class, any octave.

**Their gap is the octave-error rate**, and on bass that is the number that
matters: the fundamental is often quieter than its own harmonics, so a tracker
locks onto 2x or 3x f0 and is confidently wrong by exactly 12 or 19 semitones.
A low RPA with a high RCA is not a pitch problem, it is an octave problem --
a targeted fix (octave-aware loss, harmonic-aware input) rather than a
data-volume one. See docs/pitch-primer.md.

Voicing is reported alongside because RPA and RCA are computed over
reference-voiced frames only: a baseline that declines to answer is not
penalised by them, so accuracy is uninterpretable without knowing how often it
answered at all.
"""

import mir_eval
import numpy as np

from stemify.evaluation.metrics.base import Metric, MetricResult

CENTS_PER_OCTAVE = 1200.0


class PitchAccuracyResult(MetricResult):
    def render(self) -> str:
        s = self.scalars
        return "\n".join(
            [
                f"  frames (reference-voiced)    {s['n_voiced']:>9,.0f}",
                f"  RPA (exact)                  {s['rpa']:>9.1%}",
                f"  RCA (octave-invariant)       {s['rca']:>9.1%}",
                f"  octave-error gap             {s['octave_gap']:>9.1%}",
                f"  voicing recall               {s['voicing_recall']:>9.1%}",
                f"  voicing false alarm          {s['voicing_false_alarm']:>9.1%}",
                f"  median cents err (in-chroma) {s['median_cents']:>9.1f}",
            ]
        )


class PitchAccuracy(Metric):
    name = "pitch-accuracy"

    def __init__(self, cent_tolerance: float = 50.0) -> None:
        # 50 cents = half a semitone, the mir_eval default. Worth knowing this
        # interacts with quantisation: a model that snaps predictions to the
        # semitone grid fails on a reference note sitting 40 cents flat, even
        # though its pitch class is right. Real basses do sit flat.
        self.cent_tolerance = cent_tolerance

    def calculate(self, true: np.ndarray, pred: np.ndarray) -> PitchAccuracyResult:
        self._check(true, pred)

        ref_freq, ref_voicing = mir_eval.melody.freq_to_voicing(true)
        est_freq, est_voicing = mir_eval.melody.freq_to_voicing(pred)
        ref_cent = mir_eval.melody.hz2cents(ref_freq)
        est_cent = mir_eval.melody.hz2cents(est_freq)

        args = (ref_voicing, ref_cent, est_voicing, est_cent)
        rpa = mir_eval.melody.raw_pitch_accuracy(*args, cent_tolerance=self.cent_tolerance)
        rca = mir_eval.melody.raw_chroma_accuracy(*args, cent_tolerance=self.cent_tolerance)
        recall, false_alarm = mir_eval.melody.voicing_measures(
            ref_voicing, est_voicing
        )

        return PitchAccuracyResult(
            name=self.name,
            scalars={
                "n_voiced": float(ref_voicing.sum()),
                "rpa": float(rpa),
                "rca": float(rca),
                "octave_gap": float(rca - rpa),
                "voicing_recall": float(recall),
                "voicing_false_alarm": float(false_alarm),
                "median_cents": self._median_chroma_cents(
                    ref_cent, est_cent, ref_voicing, est_voicing
                ),
            },
        )

    def _median_chroma_cents(
        self,
        ref_cent: np.ndarray,
        est_cent: np.ndarray,
        ref_voicing: np.ndarray,
        est_voicing: np.ndarray,
    ) -> float:
        """Tuning error once octaves are folded out, over frames both agree are
        voiced. Isolates 'how precisely in tune' from 'which octave' -- a large
        octave gap can coexist with excellent cents precision, and that
        combination is the signature of harmonic locking rather than noise."""
        both = (ref_voicing > 0) & (est_voicing > 0)
        if not both.any():
            return float("nan")
        err = est_cent[both] - ref_cent[both]
        err -= CENTS_PER_OCTAVE * np.round(err / CENTS_PER_OCTAVE)
        return float(np.median(np.abs(err)))
