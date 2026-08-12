"""``build_evidence``: the padding, the striding, and what a span reads.

``Evidence`` is where the DP's numbers come from, and almost everything about it is a
*silent* failure mode -- every field is the right shape and the right dtype whether or
not it holds the right values, so a mistake here is a model trained on a different
signal rather than a crash.

Three things are pinned:

**Flat padding.** Ragged tracks are padded to a common width, and every field but
``onset_logit`` is a prefix sum that a span reads as a *difference*. Zero-padding one
would make every span reaching past a track's end read a discontinuity the size of the
whole track's mass. Repeating the last real value keeps those differences at zero. This
is asserted as the property it is -- "a span past the end is worth nothing" -- rather
than by reading the buffer.

**Row independence.** A batch of ragged tracks must score each row exactly as a batch
of one would. The padded batch and the single-track build share no code path beyond
``_track_tensors``, so agreement is evidence about the scatter.

**The mean.** ``mean_pitch_logprob`` is a mean and not a sum, which is what stops a long
note being penalised for being long -- duration is the duration potential's job.
"""

import numpy as np
import pytest
import torch

from stemify.audio.features import BINS_PER_SEMITONE
from stemify.transcription.frames import Frames
from stemify.semicrf.evidence import Evidence, build_evidence
from stemify.music.contour import Contour

DT = 0.01
BINS = 12
TRUNK = 4


def _frames(n: int, seed: int = 0, pitch: bool = True, hidden: bool = True) -> Frames:
    g = torch.Generator().manual_seed(seed)
    times = np.arange(n) * DT
    heads: dict[str, object] = {
        "onset": torch.rand(n, generator=g),
        "voicing": torch.rand(n, generator=g),
    }
    if pitch:
        heads["pitch"] = torch.randn(n, BINS, generator=g)
    return Frames(
        heads=heads,                                        # type: ignore[arg-type]
        frame_times=times,
        contour=Contour(times=times, freqs=np.full(n, 110.0, dtype=np.float32)),
        hidden=torch.randn(n, TRUNK, generator=g) if hidden else None,
    )


# --------------------------------------------------------------------------- #
# Flat padding                                                                 #
# --------------------------------------------------------------------------- #
def test_a_span_past_a_tracks_end_carries_no_mass():
    """THE padding invariant, stated as what a span reads rather than as a buffer shape.

    The scan clamps candidate starts at 0 and reads the window at its full ``dmax``
    width, so spans reaching past a short track's true length *are* addressed -- they
    pair with a ``NEG`` alpha and cannot win, but they are still evaluated. A zero tail
    would make each one read the negative of the whole track's accumulated mass, which
    is a large finite number, not an obviously wrong one.
    """
    short, long = 8, 20
    evidence = build_evidence([_frames(long), _frames(short, seed=1)], stride=1)
    row = 1
    for start in range(short, long):
        for end in range(start + 1, long + 1):
            voiced = evidence.voiced_c[row, end] - evidence.voiced_c[row, start]
            rest = evidence.rest_c[row, end] - evidence.rest_c[row, start]
            assert float(voiced.abs()) < 1e-6, f"[{start}, {end}) accrued voiced mass"
            assert float(rest.abs()) < 1e-6, f"[{start}, {end}) accrued rest mass"


def test_a_span_straddling_the_end_reads_only_the_real_frames():
    """The harder half: a span that begins inside the track and runs past its end must
    equal the same span truncated at the end. Pure flat padding gives that for free and
    a zero tail does not."""
    short, long = 8, 20
    evidence = build_evidence([_frames(long), _frames(short, seed=1)], stride=1)
    row = 1
    straddling = evidence.voiced_c[row, long] - evidence.voiced_c[row, 5]
    truncated = evidence.voiced_c[row, short] - evidence.voiced_c[row, 5]
    assert torch.allclose(straddling, truncated, atol=1e-6)


def test_the_trunk_is_flat_padded_too_and_for_a_subtler_reason():
    """``hidden`` is raw per-frame activations, not a cumsum, so "pad flat" is not
    forced by a difference. It is forced by what a scorer *does* with it: a learned
    projection has a bias, so zero frames past the end still accrue a non-zero
    ``phi`` into every prefix the moments are read from."""
    short, long = 6, 15
    evidence = build_evidence([_frames(long), _frames(short, seed=1)], stride=1)
    assert evidence.hidden is not None
    tail = evidence.hidden[1, short:]
    last = evidence.hidden[1, short - 1]
    assert torch.allclose(tail, last.expand_as(tail), atol=1e-6), (
        "the trunk's padding is not the last real frame repeated"
    )


def test_the_pitch_cumsum_is_flat_padded():
    short, long = 7, 18
    evidence = build_evidence([_frames(long), _frames(short, seed=1)], stride=1)
    assert evidence.pitch_logprob_c is not None
    row = 1
    for end in range(short + 1, long + 1):
        step = (
            evidence.pitch_logprob_c[row, end] - evidence.pitch_logprob_c[row, end - 1]
        )
        assert float(step.abs().max()) < 1e-6, f"frame {end - 1} past the end accrued"


# --------------------------------------------------------------------------- #
# Row independence                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lengths", [[12], [12, 12], [20, 8], [5, 17, 11]])
def test_each_row_of_a_batch_equals_its_own_single_track_build(lengths):
    """A padded batch must be exactly the tracks it was built from, row by row.

    ``one(row)`` re-cuts a row to its own length, so this compares like with like: the
    scatter into ``(B, Tmax, ...)`` and the single-track path share only
    ``_track_tensors``, and a transposed or mis-offset write shows up nowhere else.
    """
    tracks = [_frames(n, seed=k) for k, n in enumerate(lengths)]
    batch = build_evidence(tracks, stride=1)
    assert batch.n_tracks == len(lengths)
    assert batch.n_frames == max(lengths)
    assert batch.lengths == lengths

    for row, track in enumerate(tracks):
        alone = build_evidence(track, stride=1)
        cut = batch.one(row)
        assert cut.lengths == alone.lengths
        assert torch.allclose(cut.onset_logit, alone.onset_logit, atol=1e-6)
        assert torch.allclose(cut.voiced_c, alone.voiced_c, atol=1e-6)
        assert torch.allclose(cut.rest_c, alone.rest_c, atol=1e-6)
        assert cut.pitch_logprob_c is not None and alone.pitch_logprob_c is not None
        assert torch.allclose(cut.pitch_logprob_c, alone.pitch_logprob_c, atol=1e-6)
        assert cut.hidden is not None and alone.hidden is not None
        assert torch.allclose(cut.hidden, alone.hidden, atol=1e-6)
        assert np.allclose(cut.frame_times[0], alone.frame_times[0])


def test_one_track_and_a_list_of_one_are_the_same_evidence():
    """``build_evidence`` accepts ``Frames | Sequence[Frames]`` and dispatches on
    ``isinstance(cues, Sequence)``. A single track is a batch of one, everywhere."""
    track = _frames(10, seed=4)
    bare, wrapped = build_evidence(track, stride=1), build_evidence([track], stride=1)
    assert bare.lengths == wrapped.lengths
    assert torch.allclose(bare.voiced_c, wrapped.voiced_c, atol=1e-6)


def test_one_does_not_carry_the_precomputed_span_table():
    """``span_scores`` is indexed by row. A table built for the whole batch, carried
    into a single-row view, would answer row 0 for every question asked of it -- with
    every shape still matching."""
    batch = build_evidence([_frames(9), _frames(9, seed=1)], stride=1)
    batch.span_scores = torch.zeros(2, 10, 3, 2)
    assert batch.one(1).span_scores is None


# --------------------------------------------------------------------------- #
# Striding                                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stride", [1, 2, 3, 5])
def test_striding_decimates_every_field_onto_one_grid(stride):
    """The DP runs on a coarser grid than the encoder. Every field has to be decimated
    the *same* way or the cumsums index a different frame than ``frame_times`` names --
    and a decoded note's onset would then be read off the wrong clock."""
    n = 31
    track = _frames(n, seed=6)
    evidence = build_evidence(track, stride=stride)
    strided = len(range(0, n, stride))

    assert evidence.lengths == [strided]
    assert evidence.n_frames == strided
    assert evidence.onset_logit.shape == (1, strided)
    assert evidence.voiced_c.shape == (1, strided + 1)
    assert np.allclose(evidence.frame_times[0], track.frame_times[::stride])
    assert evidence.hidden is not None
    assert evidence.hidden.shape == (1, strided, TRUNK)
    assert evidence.pitch_logprob_c is not None
    assert evidence.pitch_logprob_c.shape == (
        1, strided + 1, BINS // BINS_PER_SEMITONE
    )
    # The onset logit at DP frame k is the encoder's frame k*stride, not an average.
    got = torch.logit(torch.as_tensor(track.onset).float().clamp(1e-6, 1 - 1e-6))
    assert torch.allclose(evidence.onset_logit[0], got[::stride], atol=1e-5)


# --------------------------------------------------------------------------- #
# mean_pitch_logprob                                                           #
# --------------------------------------------------------------------------- #
def test_the_pitch_logprob_is_a_mean_not_a_sum():
    """A long note must not be penalised for being long. If this were a sum, every
    doubling of a span would double a negative number, and the duration potential --
    whose job that is -- would be fighting a term it cannot see."""
    n = 24
    times = np.arange(n) * DT
    logits = torch.full((n, BINS), -6.0)
    logits[:, 4] = 8.0                                   # identical every frame
    track = Frames(
        heads={"onset": torch.rand(n), "voicing": torch.rand(n), "pitch": logits},
        frame_times=times,
    )
    evidence = build_evidence(track, stride=1)
    short = evidence.mean_pitch_logprob(0, 4, 8)
    long = evidence.mean_pitch_logprob(0, 4, 20)
    assert torch.allclose(short, long, atol=1e-5), (
        "the same constant pitch scored differently over 4 frames and 16 -- this is a "
        "sum, not a mean"
    )


def test_the_pitch_logprob_matches_the_per_frame_average():
    """The closed form against the definition: ``(C[end] - C[start]) / (end - start)``
    must equal the plain average of the per-frame semitone log-probabilities."""
    n = 16
    track = _frames(n, seed=8)
    evidence = build_evidence(track, stride=1)
    n_pitches = BINS // BINS_PER_SEMITONE

    logits = torch.as_tensor(track.pitch_logits)
    logprob = torch.log_softmax(logits, dim=-1)
    pad = logprob.new_full((n, 1), float("-inf"))
    blocks = torch.cat([pad, logprob], dim=-1)[:, : n_pitches * BINS_PER_SEMITONE]
    per_frame = torch.logsumexp(
        blocks.reshape(n, n_pitches, BINS_PER_SEMITONE), dim=-1
    )
    for start in range(0, n - 1):
        for end in range(start + 1, n + 1):
            want = per_frame[start:end].mean(dim=0)
            got = evidence.mean_pitch_logprob(0, start, end)
            assert torch.allclose(got, want, atol=1e-5), f"[{start}, {end})"


def test_the_pitch_logprob_broadcasts_like_the_scan_asks():
    """The scan asks with ``(B,1)`` rows against ``(1,dmax)`` starts and a scalar end;
    the gold scorer asks with three ``(S,)`` tensors. One accessor serves both."""
    evidence = build_evidence([_frames(14), _frames(14, seed=2)], stride=1)
    end, dmax = 10, 4
    starts = torch.arange(end - dmax, end)
    rows = torch.arange(2)[:, None]
    grid = evidence.mean_pitch_logprob(rows, starts[None, :], end)
    assert grid.shape == (2, dmax, BINS // BINS_PER_SEMITONE)
    for row in range(2):
        for k, start in enumerate(starts.tolist()):
            assert torch.allclose(
                grid[row, k], evidence.mean_pitch_logprob(row, start, end), atol=1e-6
            )


def test_a_model_without_a_pitch_head_refuses_rather_than_zeroing():
    """A pitch-labelled alphabet cannot fall back to scoring every pitch alike: then
    every relabelling of one segmentation is equally good and ``log Z`` spreads its
    mass over all of them, which trains toward nothing at all."""
    evidence = build_evidence(_frames(10, pitch=False), stride=1)
    assert evidence.pitch_logprob_c is None
    with pytest.raises(ValueError, match="no pitch-valued evidence"):
        evidence.mean_pitch_logprob(0, 0, 5)


# --------------------------------------------------------------------------- #
# The guards                                                                   #
# --------------------------------------------------------------------------- #
def test_a_batch_mixing_pitch_bearing_and_pitchless_tracks_is_refused():
    """Not a crash otherwise -- a wrong number. The padded buffer is allocated off the
    *first* track, so a batch led by a pitch-bearing track would leave the others'
    rows at zero and score their spans against a uniform pitch head."""
    with pytest.raises(ValueError, match="mixes tracks with and without pitch"):
        build_evidence([_frames(8), _frames(8, seed=1, pitch=False)], stride=1)


def test_a_batch_mixing_trunk_bearing_and_trunkless_tracks_is_refused():
    with pytest.raises(ValueError, match="mixes tracks with and without backbone"):
        build_evidence([_frames(8), _frames(8, seed=1, hidden=False)], stride=1)


def test_an_empty_batch_is_refused():
    with pytest.raises(ValueError, match="at least one track"):
        build_evidence([], stride=1)


def test_a_partial_set_of_contours_is_dropped_rather_than_half_filled():
    """A batch of contours only when *every* track has one. A partial list would give
    some decoded notes a contour pitch and others none, silently."""
    with_contour = _frames(9)
    without = Frames(
        heads={"onset": torch.rand(9), "voicing": torch.rand(9),
               "pitch": torch.randn(9, BINS)},
        frame_times=np.arange(9) * DT,
        hidden=torch.randn(9, TRUNK),
    )
    assert build_evidence([with_contour, with_contour], stride=1).contours is not None
    assert build_evidence([with_contour, without], stride=1).contours is None


def test_a_split_device_is_refused_at_construction():
    """The DP adds these term by term. ``Evidence`` asserts one device so the failure
    names the field rather than surfacing as a matmul error inside the scan."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    with pytest.raises(ValueError):
        Evidence(
            onset_logit=torch.zeros(1, 4),
            voiced_c=torch.zeros(1, 5, device="cuda"),
            rest_c=torch.zeros(1, 5),
            lengths=[4],
            frame_times=[np.arange(4) * DT],
            contours=None,
        )


def test_a_voicingless_track_falls_back_to_the_contour_and_then_refuses():
    """Voiced/unvoiced is the one signal the scan cannot proceed without: it is what
    separates a note from a silence. With a contour, fall back to its voiced mask; with
    neither, raise rather than return a plausible number for an unasked question."""
    times = np.arange(10) * DT
    freqs = np.where(np.arange(10) < 5, 110.0, 0.0).astype(np.float32)
    with_contour = Frames(
        heads={"onset": torch.rand(10)},
        frame_times=times,
        contour=Contour(times=times, freqs=freqs),
    )
    evidence = build_evidence(with_contour, stride=1)
    # The voiced half accrues far more mass than the unvoiced half, and vice versa.
    assert float(evidence.voiced_c[0, 5] - evidence.voiced_c[0, 0]) > float(
        evidence.voiced_c[0, 10] - evidence.voiced_c[0, 5]
    )

    bare = Frames(heads={"onset": torch.rand(10)}, frame_times=times)
    with pytest.raises(ValueError, match="no voicing signal"):
        build_evidence(bare, stride=1)
