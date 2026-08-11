"""``LabelSet``: what a segment can be labelled, and the integers the DP uses.

The DP has always had a label *alphabet*; it was two elements wide and spelled as three
module constants, so "how many labels are there" was answered by the literal ``2`` in
nine shapes and ``3`` in ``initial_window``. Widening the alphabet to name a pitch is
therefore not a change to the recurrence -- it is a change to this object, and the
recurrence has to stop hardcoding the width first.

Named ``labelset`` rather than ``labels`` because ``music.labels`` already exists and
answers a different question: that one builds *per-frame training targets* for the
heads, this one names *segment* labels for the DP.

## The alphabet

    note@p = p - lo            for p in [lo, hi]   -- a contiguous block from 0
    REST   = n_pitches
    START  = n_pitches + 1                          -- one past the alphabet

``START`` sits past the end rather than inside the alphabet because it is reachable
only at position 0 and no segment may carry it: it is a *row* of the transition matrix,
not a column. That is what lets the first segment fall out of the same recurrence step
as every other one, and it is why ``transitions()`` is ``(n_labels + 1, n_labels)``
rather than square.

**Pitches come first, and that ordering is load-bearing.** At one pitch the alphabet is
``{note: 0, rest: 1, START: 2}`` -- bit-for-bit the integers ``constants.NOTE/REST/START``
already hold. A scorer's transition table is *indexed by those integers* and rides in the
state dict, so any other ordering would load every existing checkpoint with its
transition rows permuted: no error, no shape mismatch, just a different model. Putting
rest first reads more naturally and is not worth that.

The pitch block being contiguous from 0 keeps ``label_of``/``midi_of`` arithmetic rather
than a lookup, and makes the pitch axis of a potential a plain prefix slice.

## Where the range comes from

Derived from the pitch head's own frequency axis, not declared. The head emits one
logit per CQT bin at ``BINS_PER_SEMITONE`` bins per semitone starting at ``FMIN``, and
harmonic stacking *crops* that axis (``_ConvFrontend.stack``), so the width is a
property of the model, not a constant someone maintains. A separately-declared range is
free to disagree with the head that has to supply the evidence -- and the failure is
silent, because a label whose pitch the head cannot express simply never wins.

Semitones, not bins. The metric matches pitch at +-50 cents against references that are
integer MIDI (Slakh is rendered from MIDI, FiloBass is transcribed to it), so a
semitone label is exact where a 33-cent bin label would trisect the alphabet for
resolution nothing scores. 48 labels rather than 144 is also the difference between a
2,401-entry transition matrix and a 21,316-entry one.
"""

import math
from dataclasses import dataclass

from stemify.audio.features import BINS_PER_SEMITONE, FMIN, midi_to_bin


@dataclass(frozen=True)
class LabelSet:
    """The segment alphabet: ``rest``, plus one label per representable pitch.

    ``lo``/``hi`` are inclusive MIDI semitones. A single-pitch set (``lo == hi``) is the
    degenerate case that reproduces the original ``{note, rest}`` alphabet, which is what
    lets the recurrence be generalised and tested before the alphabet is widened.
    """

    lo: int
    hi: int

    def __post_init__(self) -> None:
        if self.hi < self.lo:
            raise ValueError(f"empty pitch range: lo={self.lo} > hi={self.hi}")

    @property
    def n_pitches(self) -> int:
        return self.hi - self.lo + 1

    @property
    def REST(self) -> int:
        """The one non-pitched label, sitting just past the pitch block."""
        return self.n_pitches

    @property
    def n_labels(self) -> int:
        """Labels a *segment* may carry -- the pitches plus rest. Excludes ``START``."""
        return self.n_pitches + 1

    @property
    def START(self) -> int:
        """One past the alphabet: a transition *row*, never a segment's label."""
        return self.n_labels

    @property
    def n_rows(self) -> int:
        """Rows of the transition matrix -- the alphabet plus ``START``."""
        return self.n_labels + 1

    def label_of(self, midi: float) -> int:
        """MIDI pitch -> its label, **clamped** into range.

        Clamped rather than raising because this is called while building gold from
        reference notes, and a corpus that contains one note above the head's axis
        should not fail the run. The cost is that an out-of-range note is labelled as
        the nearest representable pitch, which is wrong but bounded -- and it is visible
        as a systematic error at the extremes rather than as a crash mid-epoch.
        """
        p = int(round(midi))
        return min(max(p, self.lo), self.hi) - self.lo

    def midi_of(self, label: int) -> int:
        """A pitch label -> its MIDI semitone. ``REST`` has no pitch and raises."""
        if label == self.REST:
            raise ValueError("REST has no pitch")
        if not 0 <= label < self.n_pitches:
            raise ValueError(
                f"label {label} is outside this alphabet "
                f"(pitches 0..{self.n_pitches - 1}, rest {self.REST})"
            )
        return self.lo + label

    def is_note(self, label: int) -> bool:
        return label != self.REST and label != self.START

    @property
    def names_pitch(self) -> bool:
        """Whether a ``note`` label says *which* pitch, or only that there is one.

        False exactly for ``BINARY``, whose single pitch label means `note` and
        nothing more -- ``midi_of`` would answer ``lo``, which is a number and not an
        answer. Consumers that read a pitch off a label ask this first and fall back
        to the contour, which is what lets one decoder serve both alphabets.
        """
        return self.n_pitches > 1

    def bin_of(self, label: int) -> int:
        """The bin whose centre frequency *is* this label's pitch.

        The bridge between the alphabet and the evidence. Exactly
        ``BINS_PER_SEMITONE * label``, since both axes are logarithmic from the same
        ``FMIN`` -- but derived through ``midi_to_bin`` rather than asserted, so it
        follows the frequency axis if that is ever reconfigured.

        Note this is the semitone's **centre**, not the bottom of a block. Anything
        pooling bins into semitones has to straddle it (``[b-1, b+2)`` at three
        bins/semitone); pooling ``[b, b+3)`` instead is half a semitone sharp, and a
        note tuned slightly flat then reads as the semitone below. See
        ``cumsums.pitch_logprob_cumsum``.
        """
        return int(round(midi_to_bin(float(self.midi_of(label)))))

    @classmethod
    def from_bins(cls, n_bins: int) -> "LabelSet":
        """The alphabet a pitch head with ``n_bins`` outputs can actually supply.

        ``FMIN`` is the axis's bottom, so the lowest whole semitone on it is
        ``round(midi of FMIN)`` and the axis spans ``n_bins / BINS_PER_SEMITONE`` of
        them. Deriving it here is what keeps the alphabet and the head in step when
        harmonic stacking crops the axis -- see the module docstring.
        """
        if n_bins < BINS_PER_SEMITONE:
            raise ValueError(
                f"n_bins={n_bins} is under one semitone at {BINS_PER_SEMITONE} "
                "bins/semitone; there is no pitch alphabet to derive"
            )
        # FMIN is D#1 by construction; recover its MIDI number rather than restating it.
        lo = int(round(69.0 + 12.0 * math.log2(FMIN / 440.0)))
        return cls(lo, lo + n_bins // BINS_PER_SEMITONE - 1)


#: The original two-label alphabet, as a degenerate ``LabelSet``. Every pitch-less
#: scorer and every existing checkpoint decodes under this, so the widening is opt-in
#: and nothing that ran before changes shape. The pitch is arbitrary -- nothing reads
#: it when a scorer has no pitch potential.
BINARY = LabelSet(lo=0, hi=0)
