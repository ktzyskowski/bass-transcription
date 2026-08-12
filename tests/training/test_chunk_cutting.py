"""``_Track.__getitem__``: the seam where track coordinates become a window's own.

Both of the live bugs recorded in this area came through here -- a chunk seeded at
``START`` when gold's own boundary was meant (5.18 nats/track, invisible to every
telescoping test because transitions were zero-initialised), and a seed-slot formula
that lived at its call site while only its range check lived beside the buffer.

The existing coverage is end-to-end: ``test_chunk_carry`` sums per-chunk losses and
compares against one uninterrupted whole-track scan, and ``test_chunked``'s tiling test
reassembles the span *endpoints*. Those are strong, and they share a blind spot -- they
pass whenever two copies of the same geometry agree with each other. What they do not
pin is the geometry against something outside it.

So this file makes the features *say which frame they are*. Bin 0 of the synthetic CQT
holds its own absolute frame index, which makes every alignment question answerable by
reading a number back out:

  - does DP frame ``k`` of a window hold track frame ``origin + k``?
  - does a gold segment's local index name the frames that segment actually covers?
  - do the targets sit on the same frames as the features they label?

None of those can be answered by comparing the pipeline to itself.

Three more properties are checked directly, each because it is what a downstream
component silently assumes:

**The ``prev`` chain.** Every chunk's first gold segment is scored against what the
*previous* chunk ended on. ``segmentation_scores`` is tested for handling ``prevs``
correctly; nothing tested the dataset *deriving* them, which is the half that was wrong.

**Labels survive the split.** The tiling test drops them (``for _, i, j in ...``). Under
a pitch alphabet a mis-assigned label is exactly the bug class -- it changes no shape
and trains notes toward the wrong semitone.

**Local gold lands where the scan looks.** The scan reads ends in
``(context, context + chunk]`` and starts no further back than ``dmax``. A gold segment
outside that is scored by ``score(gold)`` and never enumerated by ``log Z``, so the NLL
loses its lower bound.
"""

from pathlib import Path

import numpy as np
import pytest

from stemify.audio.features import SILENCE_DB
from stemify.persistence.serde import CachedItem
from stemify.semicrf.gold import gold_segments
from stemify.semicrf.labelset import BINARY, LabelSet
from stemify.semicrf.fit.dataset import _Track
from stemify.semicrf.fit.tracks import TrainItem
from stemify.music.notes import NoteEvent

N_BINS, HOP_S = 4, 0.0116
STRIDE = 3


def _cached(tmp: Path, n_cqt: int, heads=("onset",)) -> Path:
    """A track whose bin 0 holds its own absolute CQT frame index.

    That is the whole trick: every downstream slice can then be checked against where it
    claims to come from, rather than against another slice of the same pipeline. The
    other bins are constant so a transposed read is visible too, and every value is
    >= 0 while ``SILENCE_DB`` is negative -- so padding is never mistaken for content.
    """
    features = np.zeros((N_BINS, n_cqt), dtype=np.float32)
    features[0] = np.arange(n_cqt)
    features[1:] = 7.0
    targets = {head: np.arange(n_cqt, dtype=np.float32) for head in heads}
    path = tmp / "track.npz"
    CachedItem("track", features, targets, np.arange(n_cqt) * HOP_S).save(path)
    return path


def _notes(n_cqt: int) -> list[NoteEvent]:
    span = n_cqt * HOP_S
    return [
        NoteEvent(onset=float(t), offset=float(t) + 0.30, midi_pitch=40.0 + (k % 5))
        for k, t in enumerate(np.arange(0.05, span - 0.5, 0.42))
    ]


def _track(tmp: Path, n_cqt: int, chunk: int, dmax: int, right: int = 0,
           heads=("onset",), labels: LabelSet = BINARY) -> _Track:
    item = TrainItem(features=(_cached(tmp, n_cqt, heads),), notes=_notes(n_cqt))
    return _Track(item, chunk, dmax, STRIDE, right, heads, labels=labels)


GEOMETRIES = [
    # (n_cqt, chunk, dmax, right) -- including dmax < chunk (the production shape),
    # dmax > chunk (a segment can then span several chunks), and a right margin.
    (600, 40, 30, 0),
    (600, 40, 30, 5),
    (900, 20, 30, 4),
    (450, 12, 12, 0),
    (1200, 64, 20, 7),
]


# --------------------------------------------------------------------------- #
# Alignment: does a window hold the frames it says it does?                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_cqt,chunk,dmax,right", GEOMETRIES)
def test_a_window_holds_exactly_the_track_frames_its_origin_names(
    tmp_path, n_cqt, chunk, dmax, right
):
    """The features at window position ``k`` must be track frame ``origin*stride + k``.

    An off-by-one-stride here shifts every target against its features and every gold
    index against the audio it labels, and changes no shape at all -- the window is the
    right width either way, and the loss falls either way.
    """
    track = _track(tmp_path, n_cqt, chunk, dmax, right)
    width_cqt = (dmax + chunk + right) * STRIDE
    for c in range(track.n_chunks):
        features, times, _, _, _ = track[c]
        origin = c * chunk - dmax                      # in DP frames
        a = origin * STRIDE                            # in CQT frames
        assert features.shape == (1, N_BINS, width_cqt)

        want = np.arange(a, a + width_cqt)
        inside = (want >= 0) & (want < n_cqt)
        got = features[0, 0]
        assert np.array_equal(got[inside], want[inside].astype(np.float32)), (
            f"chunk {c} is not the frames starting at CQT {a}"
        )
        assert np.all(got[~inside] == SILENCE_DB), (
            f"chunk {c} pads with something other than silence off the track"
        )
        # `frame_times` must name the same frames, extrapolated past both ends.
        assert np.allclose(times, want * HOP_S), f"chunk {c}'s clock disagrees"


@pytest.mark.parametrize("n_cqt,chunk,dmax,right", GEOMETRIES)
def test_the_targets_sit_on_the_same_frames_as_the_features(
    tmp_path, n_cqt, chunk, dmax, right
):
    """Targets span the whole window, not just the scored range, so they stay index
    aligned with the features and the model's output -- the loss slices the scanned part
    out afterwards. Off the track they are **NaN**, not zero: zero is a meaningless
    feature and a meaningful *label*, and a zero-padded edge trains the pitch head
    toward MIDI 0 while the loss falls."""
    track = _track(tmp_path, n_cqt, chunk, dmax, right)
    for c in range(track.n_chunks):
        features, _, _, y, _ = track[c]
        onset = y["onset"]
        assert onset.shape == features.shape[-1:]
        real = features[0, 0] != SILENCE_DB
        assert np.array_equal(onset[real], features[0, 0][real]), (
            f"chunk {c}: targets and features name different frames"
        )
        assert np.all(np.isnan(onset[~real])), (
            f"chunk {c}: off-track targets are not NaN, so a padded frame carries a "
            "label the audio does not"
        )


# --------------------------------------------------------------------------- #
# Gold: the split, the labels, and where it lands                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_cqt,chunk,dmax,right", GEOMETRIES)
def test_the_chunk_golds_are_the_whole_track_gold_split(
    tmp_path, n_cqt, chunk, dmax, right
):
    """Every segment ending in the scanned range appears in exactly one chunk, in order,
    **with its label**, and mapping it back to track coordinates reproduces the original.

    The existing tiling test compares endpoints only. A label is the half a pitch
    alphabet makes load-bearing, and a mis-assigned one changes no shape.
    """
    labels = LabelSet(lo=38, hi=50)
    track = _track(tmp_path, n_cqt, chunk, dmax, right, labels=labels)
    whole = gold_segments(
        np.arange(n_cqt)[::STRIDE] * HOP_S, _notes(n_cqt), dmax, labels
    )
    scanned = track.n_chunks * chunk

    rebuilt = []
    for c in range(track.n_chunks):
        _, _, gold, _, _ = track[c]
        origin = c * chunk - dmax
        rebuilt += [(label, start + origin, end + origin) for label, start, end in gold]

    want = [seg for seg in whole if seg[2] <= scanned]
    assert rebuilt == want, (
        "per-chunk gold is not the whole-track gold split: a segment was dropped, "
        "duplicated, relabelled or rebased wrongly"
    )


@pytest.mark.parametrize("n_cqt,chunk,dmax,right", GEOMETRIES)
def test_every_gold_segment_lands_where_the_scan_will_look_for_it(
    tmp_path, n_cqt, chunk, dmax, right
):
    """``log Z`` enumerates spans ending in ``(context, context + chunk]`` whose starts
    reach at most ``dmax`` back. A gold segment outside that is scored by the other half
    of the NLL and never by this one, so the difference is unbounded below -- the loss
    goes negative and the model farms the span nothing normalises."""
    track = _track(tmp_path, n_cqt, chunk, dmax, right)
    context = track.geometry.context
    for c in range(track.n_chunks):
        _, _, gold, _, _ = track[c]
        for _label, start, end in gold:
            assert context < end <= context + chunk, (
                f"chunk {c}: segment ends at local {end}, outside the scanned "
                f"({context}, {context + chunk}]"
            )
            assert start >= 0, f"chunk {c}: segment starts before the window at {start}"
            assert 1 <= end - start <= dmax, (
                f"chunk {c}: segment is {end - start} frames, which the scan's "
                f"candidate starts (dmax={dmax}) cannot reach"
            )


@pytest.mark.parametrize("n_cqt,chunk,dmax,right", GEOMETRIES)
def test_the_first_segments_seed_slot_fits_the_alpha_window(
    tmp_path, n_cqt, chunk, dmax, right
):
    """The dataset's cut has to be compatible with the buffer the scan resumes into.

    ``initial_window`` refuses a slot outside ``[0, dmax)`` -- correctly, since an
    out-of-range one means the gold pieces and the scan disagree about where the chunk
    begins. That refusal is a *crash*; this is the check that the dataset never earns
    it, which is the same fact stated where it can be acted on.
    """
    track = _track(tmp_path, n_cqt, chunk, dmax, right)
    for c in range(track.n_chunks):
        _, _, gold, _, _ = track[c]
        if not gold:
            continue
        slot = track.geometry.seed_slot(gold[0][1], dmax)
        assert 0 <= slot < dmax, f"chunk {c}: seed slot {slot} outside [0, {dmax})"


# --------------------------------------------------------------------------- #
# prev: the recorded live bug                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_cqt,chunk,dmax,right", GEOMETRIES)
def test_each_chunk_is_told_what_the_previous_one_ended_on(
    tmp_path, n_cqt, chunk, dmax, right
):
    """The chain, checked as a chain rather than by re-deriving the implementation's
    index arithmetic.

    Gold tiles the track contiguously, so the segment preceding a chunk's first is
    simply the last segment some earlier chunk kept. ``START`` belongs to exactly one
    place -- before the track's own first segment -- and scoring every chunk against it
    was a live defect worth 5.18 nats per track: ``log Z`` applies the true carried
    transition, so the two halves of the NLL were measuring different quantities. It
    was invisible while transitions were zero-initialised, which is exactly where a
    telescoping test looks.
    """
    labels = LabelSet(lo=38, hi=50)
    track = _track(tmp_path, n_cqt, chunk, dmax, right, labels=labels)
    expected = labels.START
    seen_any = False
    for c in range(track.n_chunks):
        _, _, gold, _, prev = track[c]
        if not gold:
            continue
        assert prev == expected, (
            f"chunk {c} is scored against {prev}, but the previous gold segment was "
            f"{expected}"
        )
        expected = gold[-1][0]
        seen_any = True
    assert seen_any, "no chunk carried gold -- this fixture is not exercising anything"


def test_only_a_tracks_first_chunk_is_seeded_from_start(tmp_path):
    """``START`` is a row of the transition matrix reachable at position 0 only. A
    later chunk seeded there is claiming the track begins twice."""
    labels = LabelSet(lo=38, hi=50)
    track = _track(tmp_path, 900, 40, 30, labels=labels)
    prevs = []
    for c in range(track.n_chunks):
        _, _, gold, _, prev = track[c]
        if gold:
            prevs.append(prev)
    assert prevs[0] == labels.START
    assert labels.START not in prevs[1:], (
        f"START reappears at chunk(s) {[k for k, p in enumerate(prevs) if p == labels.START]}"
    )


# --------------------------------------------------------------------------- #
# The ragged tail                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_cqt,chunk,dmax,right", GEOMETRIES)
def test_the_ragged_tail_is_dropped_rather_than_padded(
    tmp_path, n_cqt, chunk, dmax, right
):
    """``n_chunks = n_dp // chunk``. A partial final chunk would break the uniform width
    the scan's static shapes -- and its CUDA graph capture -- depend on, so the tail is
    dropped. Pinned because it is a silent *data* loss: nothing reports the frames that
    never reach the objective."""
    track = _track(tmp_path, n_cqt, chunk, dmax, right)
    n_dp = n_cqt // STRIDE
    assert track.n_chunks == n_dp // chunk
    assert track.n_chunks * chunk <= n_dp
    assert (track.n_chunks + 1) * chunk > n_dp


def test_a_track_shorter_than_one_chunk_yields_nothing(tmp_path):
    """Rather than one short window. The stream loop advances to the next track on
    ``chunk_no >= n_chunks``, so zero is the value that makes that skip work."""
    assert _track(tmp_path, 30, chunk=40, dmax=30).n_chunks == 0


# --------------------------------------------------------------------------- #
# The link: gold's local index and the evidence's DP index are the same frame  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_cqt,chunk,dmax,right", GEOMETRIES)
def test_a_gold_index_names_the_dp_frame_the_scan_will_score_it_at(
    tmp_path, n_cqt, chunk, dmax, right
):
    """The last link in the chain, and the only one nothing else covers.

    The dataset rebases gold into window-local **DP** frames. The evidence is built
    separately, by striding the window's **CQT** frame times. Both then index the same
    axis inside the scan, and nothing until now said they agree -- the gold half of the
    NLL and the ``log Z`` half would simply be talking about different frames, at a
    constant offset, for the whole of training.

    Checked by going out to the features and back: DP frame ``k`` of the evidence must
    hold the track frame that gold's local index ``k`` names.
    """
    from stemify.transcription.frames import Frames
    from stemify.semicrf.evidence import build_evidence

    track = _track(tmp_path, n_cqt, chunk, dmax, right)
    for c in range(track.n_chunks):
        features, times, gold, _, _ = track[c]
        origin = c * chunk - dmax
        n_dp_window = features.shape[-1] // STRIDE
        evidence = build_evidence(
            Frames(
                heads={"onset": np.full(times.size, 0.5, dtype=np.float32),
                       "voicing": np.full(times.size, 0.5, dtype=np.float32)},
                frame_times=times,
            ),
            stride=STRIDE,
        )
        assert evidence.lengths == [n_dp_window]
        # The evidence's own clock, frame by frame, against the track's.
        want = (np.arange(origin, origin + n_dp_window) * STRIDE) * HOP_S
        assert np.allclose(evidence.frame_times[0], want)

        for label, start, end in gold:
            # The DP frame gold points at, read back off the features it labels.
            for local in (start, end - 1):
                cqt = features[0, 0, local * STRIDE]
                track_cqt = (origin + local) * STRIDE
                if 0 <= track_cqt < n_cqt:
                    assert cqt == pytest.approx(track_cqt), (
                        f"chunk {c}, segment ({label}, {start}, {end}): local DP frame "
                        f"{local} holds CQT {cqt}, not track CQT {track_cqt}"
                    )
                assert evidence.frame_times[0][local] == pytest.approx(
                    track_cqt * HOP_S
                )
