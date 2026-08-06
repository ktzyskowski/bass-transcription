"""Raw pitch / chroma accuracy -- the octave-error diagnostic.

  - **RPA** (raw pitch accuracy): fraction of reference-voiced frames answered
    within 50 cents. Exactly right, octave included.
  - **RCA** (raw chroma accuracy): the same with octaves folded out. Right pitch
    class, any octave.

**Their gap is the octave-error rate.** The theory behind expecting it to matter:
a bass fundamental is often quieter than its own harmonics, so a tracker locks
onto 2x or 3x f0 and is confidently wrong by exactly 12 or 19 semitones. A low
RPA with a high RCA would be an octave problem rather than a pitch one, wanting
a targeted fix (octave-aware loss, harmonic-aware input) rather than more data.

**Measured, that theory has not held here.** pyin's octave gap is 0.1% on IDMT
isolated notes, 0.3% on clean MDB stems, 5.4% on raw mixes, and 1.1% after
Demucs separation -- everywhere a small fraction of the error budget. The ~46%
figure in docs/pitch-primer.md is an artifact of naive CQT peak-picking, not a
property of the task, and pyin's internal octave logic already defeats it. The
dominant failure on mixes is instead **voicing collapse**: recall fell 96.8% to
61.8% while the octave gap barely moved. Keep reporting the gap, but do not
design around it without re-measuring.

Voicing is reported alongside because RPA and RCA are computed over
reference-voiced frames only: a baseline that declines to answer is not
penalised by them, so accuracy is uninterpretable without knowing how often it
answered at all.

**Overall Accuracy** exists because RPA and RCA are structurally blind to an
entire class of error. Their denominator is reference-*voiced* frames, so what
happens during rests is simply not counted -- a model that hallucinates a pitch
through every silence scores *identical* RPA and RCA to one that correctly says
nothing. Overall Accuracy counts every frame, crediting a correct silence the
same as a correct pitch, and is therefore the number closest to "would the
transcription be right".

The two are companions, not substitutes. RPA answers "can you name the pitch",
Overall answers "are you right about what is happening". Their divergence is
itself the diagnostic: on MDB-bass-synth, Demucs-separated audio sits 6.8 points
below the clean stem under RPA but 16.2 points below under Overall, because
separation leaves bass-like residue in the rests that RPA cannot see. Note also
that Overall is denominator-sensitive -- a sparse bassline weights silence
handling more heavily than a busy one -- so it is not comparable across datasets
with different voiced fractions.
"""

import mir_eval
import numpy as np

from stemify.core.metrics.base import MetricResult
from stemify.core.metrics.frame.base import FrameMetric

CENTS_PER_OCTAVE = 1200.0


class PitchAccuracyResult(MetricResult):
    def render(self) -> str:
        s = self.scalars
        return "\n".join(
            [
                f"  frames (total / voiced)      "
                f"{s['n_frames']:>9,.0f} / {s['n_voiced']:,.0f}",
                f"  RPA (exact)                  {s['rpa']:>9.1%}",
                f"  RCA (octave-invariant)       {s['rca']:>9.1%}",
                f"  Overall (incl. silence)      {s['overall']:>9.1%}",
                f"  octave-error gap             {s['octave_gap']:>9.1%}",
                f"  voicing recall               {s['voicing_recall']:>9.1%}",
                f"  voicing false alarm          {s['voicing_false_alarm']:>9.1%}",
                f"  median cents err (in-chroma) {s['median_cents']:>9.1f}",
            ]
        )


class PitchAccuracy(FrameMetric):
    name = "pitch-accuracy"

    def __init__(self, cent_tolerance: float = 50.0) -> None:
        # 50 cents = half a semitone, the mir_eval default. Worth knowing this
        # interacts with quantisation: a model that snaps predictions to the
        # semitone grid fails on a reference note sitting 40 cents flat, even
        # though its pitch class is right. Real basses do sit flat.
        self.cent_tolerance = cent_tolerance

    def compute(self, true: np.ndarray, pred: np.ndarray) -> PitchAccuracyResult:
        self._check(true, pred)

        ref_freq, ref_voicing = mir_eval.melody.freq_to_voicing(true)
        est_freq, est_voicing = mir_eval.melody.freq_to_voicing(pred)
        ref_cent = mir_eval.melody.hz2cents(ref_freq)
        est_cent = mir_eval.melody.hz2cents(est_freq)

        args = (ref_voicing, ref_cent, est_voicing, est_cent)
        rpa = mir_eval.melody.raw_pitch_accuracy(*args, cent_tolerance=self.cent_tolerance)
        rca = mir_eval.melody.raw_chroma_accuracy(*args, cent_tolerance=self.cent_tolerance)
        overall = mir_eval.melody.overall_accuracy(
            *args, cent_tolerance=self.cent_tolerance
        )
        recall, false_alarm = mir_eval.melody.voicing_measures(
            ref_voicing, est_voicing
        )

        return PitchAccuracyResult(
            name=self.name,
            scalars={
                "n_frames": float(ref_voicing.size),
                "n_voiced": float(ref_voicing.sum()),
                "rpa": float(rpa),
                "rca": float(rca),
                "overall": float(overall),
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
