"""Shared constants for the semi-CRF decoder: grid decimation, segment labels, and
the duration-bucket edges. Kept in one module so ``emissions``, ``gold``, ``dp`` and
``potentials`` agree on the label integers and the ``log 0`` sentinel without a cycle.
"""

import numpy as np

# Decimate the ~11.6 ms feature grid before the DP: the scan is O(T * dmax), so a
# coarser grid is a direct speed lever, and a ~35 ms boundary resolution is well
# inside the 50 ms onset tolerance the note metric scores at.
STRIDE = 3
# Longest representable segment. Must exceed the longest real bass note, or a held
# note is forced to split; 4 s is comfortably past that.
DMAX_S = 4.0
# A between-note MIDI gap shorter than this is a note-off artifact (the audio rings
# on), not audible silence -- gold closes it into a contiguous note->note transition
# instead of a phantom rest. Longer gaps stay rests. This is the fix that stops the
# CRF learning to avoid note->note and collapsing legato runs.
GAP_REST_S = 0.12
# Note-duration potential buckets, in seconds. A learned scalar per bucket lets the
# model say "notes this short are implausible" -- the over-segmentation lever.
DUR_EDGES_S = np.array([0.06, 0.12, 0.20, 0.35, 0.6, 1.0, 2.0])

# Segment labels. START is only reachable at position 0, which makes the first-
# segment case fall out of the same scan as every other step (no special-casing).
NOTE, REST, START = 0, 1, 2
NEG = -1e9  # log 0, for unreachable states
