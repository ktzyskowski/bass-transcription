"""Where the false-positive notes come from: fragmentation vs voicing.

Answers a question P/R/F cannot: of the phantom notes we emit, how many are the
segmenter shredding one held note into fragments (fix: better note boundaries)
versus the tracker hearing bass where the reference is silent (fix: better
voicing). Every false positive lands in exactly one bucket, and the split says
which head is worth building:

  - ``fragmentation`` -- onset falls *inside* a reference note's span and carries
    that note's pitch: a leftover fragment of a real note the segmenter split.
  - ``rest`` -- onset falls in a reference *rest*: a voicing false alarm.
  - ``interior_offpitch`` -- inside a span but the *wrong* pitch (an octave slip
    or confusion while a real note sounds); kept apart so it inflates neither.

"Fragmentation" is the plain term for over-segmentation -- a held note broken
into pieces. Reuses ``matched_pairs`` from ``note`` so "matched" -- hence which
estimates are false positives -- means exactly what it does for ``PRF``.
"""

from stemify.core.metrics.base import MetricResult
from stemify.core.metrics.note.base import NoteMetric, matched_pairs
from stemify.core.notes import NoteEvent


class FragmentationResult(MetricResult):
    def render(self) -> str:
        s = self.scalars
        fp = s["fp"]

        def frac(n: float) -> float:
            return n / fp if fp else 0.0

        return "\n".join([
            f"  false positives              {int(fp):,}",
            f"    fragmentation              {int(s['fragmentation']):>7,} "
            f"({frac(s['fragmentation']):.1%})",
            f"    voicing (note in a rest)   {int(s['rest']):>7,} "
            f"({frac(s['rest']):.1%})",
            f"    interior, wrong pitch      {int(s['interior_offpitch']):>7,} "
            f"({frac(s['interior_offpitch']):.1%})",
        ])


class Fragmentation(NoteMetric):
    name = "fragmentation"

    def compute(
        self,
        references: list[list[NoteEvent]],
        estimates: list[list[NoteEvent]],
    ) -> MetricResult:
        semitone_tol = self.pitch_tolerance / 100.0
        fragmentation = interior_off = rest = fp = 0
        for ref, est in zip(references, estimates, strict=True):
            matched = {
                e for _, e in matched_pairs(
                    ref, est, self.onset_tolerance, self.pitch_tolerance
                )
            }
            for j, e in enumerate(est):
                if j in matched:
                    continue
                fp += 1
                # Reference notes sounding at this estimate's onset. The offset
                # is the MIDI note-off -- the honest span for "held here".
                covering = [r for r in ref if r.onset <= e.onset <= r.offset]
                if not covering:
                    rest += 1
                elif any(abs(e.midi_pitch - r.midi_pitch) <= semitone_tol for r in covering):
                    fragmentation += 1
                else:
                    interior_off += 1
        return FragmentationResult(
            self.name,
            {
                "fp": fp,
                "fragmentation": fragmentation,
                "interior_offpitch": interior_off,
                "rest": rest,
            },
        )
