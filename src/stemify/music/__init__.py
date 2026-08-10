"""The bottom pillar: what a note is, what a frame grid is, and how to move between them.

The first of the vertical pillars that replace ``core``. ``core`` was defined
*negatively* -- "does not depend on a model" -- and a negative definition cannot refuse
anything, which is how it accumulated 1800 lines of loose modules and two god-objects.
Each pillar is defined by what it *is*, and is held to two properties:

    **detachable**  it imports nothing else from ``stemify``
    **verifiable**  its tests need no fakes, stubs or doubles

``music`` is the extreme case of both: three modules, no ``stemify`` imports at all, and
nothing but ``numpy`` underneath. Everything here is a pure value or a pure function of
arrays, so every test is a literal input and a literal expected output.

    notes   ``NoteEvent`` -- a discrete note: onset, offset, pitch
    grids   nearest-neighbour resampling between two time grids
    labels  per-frame training targets (onset, voicing, pitch) on a given grid

The through-line is that all three are about *time and pitch as data*, independent of
where the audio came from or what consumes it. That is why the pyin ``Contour`` is not
here yet and why ``features`` (the CQT) is not either: both name a specific producer,
which is the ``audio`` pillar's business.

``labels`` belongs here rather than with training because a target is a fact about the
reference, not about the optimiser -- the same builders serve the trainer and the
evaluation harness, and neither owns them.
"""
