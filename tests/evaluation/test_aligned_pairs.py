"""aligned_pairs: turn an item's reference + the encoder's Frames into aligned pairs.

This is where the grid alignment lives (onset impulses, voiced mask, contour onto
the reference grid), so it is exercised with a hand-built item and Frames -- no audio.
"""

import numpy as np

from stemify.audio.sources import Sources
from stemify.music.contour import Contour
from stemify.transcription.frames import Frames
from stemify.evaluation.cues import aligned_pairs
from stemify.music.notes import NoteEvent

FRAME_TIMES = np.arange(12) * 0.0116


class _FrameOnlyItem:
    """A corpus with a contour and no note truth -- MDB-bass-synth and IDMT.

    Implements ``AudioItem`` and stops there. It has to be *complete* at that protocol
    rather than only carrying the two members ``aligned_pairs`` reads, because the
    capability check downstream is an ``isinstance`` against a runtime-checkable
    protocol, which verifies every inherited member. A fake missing ``sources`` would
    be classified "no note truth" for the wrong reason and would pass this file's tests
    while proving nothing about the real corpora.
    """

    @property
    def name(self) -> str:
        return "fake"

    @property
    def groups(self) -> dict[str, str]:
        return {}

    def sources(self) -> Sources:
        return Sources(bass="/nonexistent.wav")

    def reference_contour(self) -> Contour:
        return Contour(times=FRAME_TIMES, freqs=np.full(FRAME_TIMES.size, 110.0))


class _FakeItem(_FrameOnlyItem):
    """The same, plus note-level truth -- Slakh and FiloBass."""

    def __init__(self, note_onset: float):
        self._onset = note_onset

    def reference_notes(self):
        return [NoteEvent(onset=self._onset, offset=self._onset + 0.05, midi_pitch=45.0)]


def _cues(with_voicing: bool) -> Frames:
    heads = {"onset": np.zeros(12, dtype=np.float32)}
    if with_voicing:
        heads["voicing"] = np.ones(12, dtype=np.float32)
    return Frames(
        heads=heads,
        frame_times=FRAME_TIMES,
        contour=Contour(times=FRAME_TIMES, freqs=np.full(12, 110.0)),
    )


def test_onset_target_impulse_lands_on_the_note_onset():
    ch = aligned_pairs(_cues(with_voicing=False), _FakeItem(note_onset=FRAME_TIMES[5]))
    _, onset_true = ch.onset
    # The impulse (widened by one frame) is centred on frame 5, and nowhere else.
    assert onset_true[5] == 1.0
    assert onset_true[[0, 1, 9, 10, 11]].sum() == 0.0


def test_voicing_dropped_without_a_head():
    assert aligned_pairs(_cues(with_voicing=False), _FakeItem(0.03)).voicing is None


def test_voicing_and_pitch_aligned_to_the_frame_grid():
    ch = aligned_pairs(_cues(with_voicing=True), _FakeItem(0.03))
    voi_pred, voi_true = ch.voicing
    assert voi_pred.shape == voi_true.shape == FRAME_TIMES.shape
    est_hz, ref_hz = ch.pitch
    assert est_hz.shape == ref_hz.shape  # both on the reference grid
    assert np.allclose(ref_hz, 110.0)


def test_a_frame_annotated_corpus_drops_onset_instead_of_raising():
    """The regression. MDB-bass-synth and IDMT have no note-level truth, and this used
    to call ``reference_notes()`` unconditionally -- so both raised
    ``NotImplementedError`` from inside the scoring loop, and neither could be scored at
    ``--level cues`` at all. Onset is now dropped on the same terms as the other two
    channels, and everything the corpus *does* have is still scored."""
    ch = aligned_pairs(_cues(with_voicing=True), _FrameOnlyItem())
    assert ch.onset is None
    assert ch.voicing is not None
    assert ch.pitch is not None
