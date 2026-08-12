"""Shared constants for the semi-CRF decoder: grid decimation and segment labels.
Kept in one module so ``evidence``, ``gold``, the recurrences and the scorers agree on
the label integers and the ``log 0`` sentinel without a cycle.
"""

# Decimate the ~11.6 ms feature grid before the DP: the scan is O(T * dmax), so a
# coarser grid is a direct speed lever. A ~35 ms grid places an onset within +-17.4 ms,
# comfortably inside the note metric's 50 ms tolerance -- but only because gold rounds
# to the *nearest* frame. It used to round up, making the error one-sided at 0..+35 ms
# and leaving a model 15 ms of late-side budget where it now has 33.
STRIDE = 3
# Longest representable segment. A note outlasting it becomes consecutive spans of the
# same label rather than being cut short, so this is a cost in extra boundaries the
# model must learn to expect, not a correctness bound; 4 s is past any real bass note.
DMAX_S = 4.0
# Segment labels. START is only reachable at position 0, which makes the first-
# segment case fall out of the same scan as every other step (no special-casing).
NOTE, REST, START = 0, 1, 2
NEG = -1e9  # log 0, for unreachable states
