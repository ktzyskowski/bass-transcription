"""``ChunkGeometry``/``ChunkFrame``: the coordinate conversions, in one place.

Six coordinate systems meet in the chunked semi-CRF and until this module they were
all bare ``int``. A gold segment is ``(label, i, j)`` before it is rebased into a
window's own frame and ``(label, i, j)`` after -- same tuple type, different meaning,
no marker. Both recorded bugs in this area came through that seam.

What is pinned here is the *algebra*: every conversion round-trips, and the two that
were previously written at their call sites agree with the definitions the scan and
the loss actually rely on. These are pure integer functions, so the cases are exact
rather than approximate, and the awkward ones (a window that starts before the track)
are reachable directly instead of only through a dataset.
"""

import pytest

from stemify.semicrf.fit.geometry import ChunkFrame, ChunkGeometry

# The shape a real run uses: left context equal to dmax, a 128-frame chunk, a right
# margin for the encoder's forward receptive field, on the stride-decimated grid.
REAL = ChunkGeometry(context=64, chunk=128, right=8, stride=2)


def test_width_is_the_three_parts():
    assert REAL.width == 64 + 128 + 8


def test_dp_to_cqt_is_the_stride():
    assert REAL.cqt(0) == 0
    assert REAL.cqt(10) == 20


# --------------------------------------------------------------------------- #
# localise / to_track round-trip                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("origin", [-64, 0, 64, 1024])
@pytest.mark.parametrize("track_frame", [0, 1, 500, 4096])
def test_localising_a_track_frame_round_trips(origin, track_frame):
    frame = REAL.at(origin)
    assert frame.to_track(frame.localise(track_frame)) == track_frame


def test_a_windows_own_origin_is_its_local_zero():
    frame = REAL.at(1024)
    assert frame.localise(1024) == 0
    assert frame.to_track(0) == 1024


def test_a_negative_origin_is_a_tracks_first_chunk():
    """The window begins before the track does and is zero-padded there. The origin is
    genuinely negative rather than clamped, because clamping it would shift every gold
    index in the chunk against the features it describes."""
    frame = REAL.at(-REAL.context)
    assert frame.localise(0) == REAL.context      # track frame 0 sits after the context
    assert frame.origin < 0


def test_the_cqt_window_is_the_origin_and_width_in_feature_frames():
    frame = REAL.at(100)
    start, width = frame.cqt_window()
    assert start == 100 * REAL.stride
    assert width == REAL.width * REAL.stride


# --------------------------------------------------------------------------- #
# seed_slot: the formula that used to live at its call site                   #
# --------------------------------------------------------------------------- #
def test_a_seed_at_the_scan_start_lands_on_the_last_slot():
    """"No frames consumed yet" -- a track's first chunk, where ``START`` belongs. The
    scan's first step is ``j = context + 1`` and reads window index ``k`` as position
    ``j - dmax + k``, so ``context`` itself is the newest slot."""
    dmax = 64
    assert REAL.seed_slot(REAL.context, dmax) == dmax - 1


def test_a_seed_one_frame_back_lands_one_slot_earlier():
    dmax = 64
    assert REAL.seed_slot(REAL.context - 1, dmax) == dmax - 2


def test_the_furthest_reachable_seed_is_the_oldest_slot():
    """A gold segment ending inside the chunk starts at most ``dmax`` back, which is
    exactly slot 0. Anything earlier is out of the window, and ``initial_window``
    refuses it."""
    dmax = 64
    assert REAL.seed_slot(REAL.context - dmax + 1, dmax) == 0
    assert REAL.seed_slot(REAL.context - dmax, dmax) == -1     # out of range


def test_seed_slot_does_not_require_context_to_equal_dmax():
    """They are equal today and reach that value by different routes -- the dataset
    takes ``context`` from ``SemiCRFRecipe._geometry``. The conversion is correct for
    any pair, so the two staying separate fields costs nothing and leaves left context
    free to change independently of the widest scoreable span."""
    geo = ChunkGeometry(context=10, chunk=128, right=8, stride=2)
    assert geo.seed_slot(10, 64) == 63     # at the scan start -> newest slot
    assert geo.seed_slot(5, 64) == 58      # five frames back -> five slots back


# --------------------------------------------------------------------------- #
# scored_slice: what the NLL and every aux term must agree on                 #
# --------------------------------------------------------------------------- #
def test_the_scored_slice_is_the_chunks_own_frames():
    """Not the whole window. A term averaging over the context too would divide by a
    different frame count than the NLL does, so the shared "nats per frame" unit that
    makes the head weights interpretable would quietly stop holding between terms."""
    window_cqt = REAL.width * REAL.stride
    lo, hi = REAL.scored_slice(window_cqt)
    assert lo == REAL.context * REAL.stride
    assert hi == window_cqt - REAL.right * REAL.stride
    assert hi - lo == REAL.chunk * REAL.stride


def test_a_geometry_with_no_context_scores_everything_but_the_right_margin():
    geo = ChunkGeometry(context=0, chunk=50, right=0, stride=1)
    assert geo.scored_slice(50) == (0, 50)


# --------------------------------------------------------------------------- #
# The value objects themselves                                                #
# --------------------------------------------------------------------------- #
def test_a_geometry_is_frozen():
    """The batch carries one, and a field changed after the chunks were cut would
    describe a cutting that never happened."""
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        REAL.context = 1  # type: ignore[misc]


def test_at_returns_a_frame_carrying_this_geometry():
    frame = REAL.at(7)
    assert isinstance(frame, ChunkFrame)
    assert frame.geometry is REAL
