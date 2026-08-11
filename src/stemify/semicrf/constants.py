"""Shared constants for the semi-CRF decoder: grid decimation and segment labels.
Kept in one module so ``evidence``, ``gold``, the recurrences and the scorers agree on
the label integers and the ``log 0`` sentinel without a cycle.
"""

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

# Segment labels. START is only reachable at position 0, which makes the first-
# segment case fall out of the same scan as every other step (no special-casing).
NOTE, REST, START = 0, 1, 2
NEG = -1e9  # log 0, for unreachable states
