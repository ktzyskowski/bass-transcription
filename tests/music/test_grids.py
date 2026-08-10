"""Nearest-neighbour resampling between two time grids.

Pure array arithmetic over hand-written grids, so every expected value below is
countable by eye. The cases worth having are the ones where two candidate frames are
*nearly* equally good -- exact hits, exact midpoints, and queries off either end -- since
those are where an off-by-one picks a neighbouring frame and every shape downstream
still checks out.
"""

import numpy as np

from stemify.music.grids import nearest_frame, resample_mask

GRID = np.array([0.0, 1.0, 2.0, 3.0])


def test_a_query_on_a_frame_picks_that_frame():
    assert list(nearest_frame(GRID, GRID)) == [0, 1, 2, 3]


def test_a_query_between_frames_picks_the_closer_one():
    got = nearest_frame(GRID, np.array([0.4, 0.6, 2.9]))
    assert list(got) == [0, 1, 3]


def test_an_exact_midpoint_resolves_to_the_later_frame():
    """The tie-break, pinned because it is a choice and not a law.

    Nothing depends on *which* way it goes -- a query landing exactly halfway between two
    frames is a measure-zero event on real timestamps -- but it must not drift, because
    it is the difference between an onset target sitting at frame k and at frame k+1.
    """
    assert list(nearest_frame(GRID, np.array([0.5, 1.5, 2.5]))) == [1, 2, 3]


def test_queries_off_either_end_clamp_to_the_edge_frames():
    """Clamped rather than extrapolated: a grid is a finite set of sample times, and the
    nearest member of it to a query outside the range is the end. The pyin contour and
    the CQT routinely differ by a frame at the tail, which is exactly this case."""
    assert list(nearest_frame(GRID, np.array([-10.0, 99.0]))) == [0, 3]


def test_a_mask_resamples_by_taking_each_target_frames_nearest_source_value():
    src_times = np.array([0.0, 0.1, 0.2, 0.3])
    src_mask = np.array([0, 1, 1, 0])
    # 0.02 -> src 0 (0), 0.09 -> src 1 (1), 0.19 -> src 2 (1), 0.31 -> src 3 (0)
    got = resample_mask(src_times, src_mask, np.array([0.02, 0.09, 0.19, 0.31]))
    assert list(got) == [0.0, 1.0, 1.0, 0.0]


def test_resampling_is_lossy_in_the_direction_you_would_expect():
    """A source event shorter than the target hop can vanish entirely.

    Not a defect to be fixed here -- it is what nearest-neighbour means, and the
    alternative (any-overlap) would widen every mask instead. Recorded because it is the
    mechanism behind a whole class of missing target: a 20 ms voiced blip on the 10 ms
    reference grid has no frame of its own on a 100 ms grid to land on.
    """
    src_times = np.arange(10) * 0.01
    src_mask = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])  # one 10 ms blip at t=0.04
    coarse = np.array([0.0, 0.1])
    assert list(resample_mask(src_times, src_mask, coarse)) == [0.0, 0.0]


def test_an_empty_grid_on_either_side_yields_zeros_not_an_error():
    """Both are reachable: a track with no voiced frames at all, and a zero-length clip.
    Returning the correctly shaped zeros keeps the caller free of a length check."""
    assert list(resample_mask(np.array([]), np.array([]), GRID)) == [0.0] * 4
    assert resample_mask(GRID, np.ones(4), np.array([])).shape == (0,)


def test_the_resampled_mask_is_float32_whichever_path_it_took():
    """The targets are compared against float32 model outputs; a float64 mask would
    upcast the whole loss silently. Both branches must agree, and the empty-input branch
    builds its own array rather than passing one through."""
    assert resample_mask(GRID, np.ones(4, dtype=np.int64), GRID).dtype == np.float32
    assert resample_mask(np.array([]), np.array([]), GRID).dtype == np.float32
