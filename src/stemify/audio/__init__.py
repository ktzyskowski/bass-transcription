"""Waveforms and the grid they become: what audio a track has, and how it is heard.

The pillar between the machine and the domain. It answers one question -- *how does a
recording become something a model can consume* -- and it answers it the same way
whatever corpus the recording came from, which is exactly why it is not part of
``corpus``.

    sources     ``Sources`` -- the audio roles one track offers (bass, mix, ...)
    transforms  ``AudioTransform`` -- cached audio-in/audio-out, i.e. separation
    features    the CQT front-end, its constants, and ``FrameFeatures``

The three compose in that order: a corpus item offers ``Sources``, a transform resolves
one role to an actual file, and the CQT turns that file into a ``FrameFeatures``. Each
step is a pure function of the one before plus a named configuration, which is what
lets the whole chain be content-addressed rather than sequenced -- ``cache_path`` and
``SIGNATURE`` exist so a caller can ask *where would this land* without doing the work.

**The constants are the point of the pillar.** ``FMIN``, ``BINS_PER_SEMITONE``,
``HOP_LENGTH``, ``SILENCE_DB``: every one of them encodes a decision whose reasoning
cost more than its code, and every one of them has to be agreed on by the model, the
targets, the metrics and the DP grid. They live here because this is the layer that
*defines* them; anything above reads them rather than restating them, and
``semicrf.labelset`` derives its whole pitch alphabet from ``FMIN`` and
``BINS_PER_SEMITONE`` rather than declaring a range that could disagree.

**No torch, and no corpus.** The audio path is deliberately reachable without loading a
network -- ``tests/test_layers.py`` holds that -- and nothing here knows what a dataset
or a reference is. ``FrameFeatures`` carries features and the frame times together, and
notably *not* a pitch contour: different producer, different grid, different cache.
"""
