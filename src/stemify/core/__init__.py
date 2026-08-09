"""The bottom layer: what every other package is allowed to depend on.

**The rule, in one line:** ``core`` imports nothing from ``models``, ``training``,
``semicrf``, ``evaluation`` or ``serving``. Nothing above it may be imported here, ever.
That is the whole contract, and it is checkable -- an import of an upper layer from
inside this package is a defect, not a judgement call.

The sharpest consequence, and the one worth stating first: **nothing here imports
torch.** Not one module. That began as a consequence of the layering rather than a goal,
but it is the version of the rule a test can hold -- ``tests/test_layers.py`` asserts
that importing this package leaves ``torch`` out of ``sys.modules`` -- so it is now the
tripwire for the whole boundary. Deep learning lives one layer up, in ``models``.

It holds four kinds of thing, and the boundary between them and the layers above is
worth stating because it is not obvious from size:

    protocols     what *inference* composes -- Encoder, Decoder, PitchEstimator,
                  AudioTransform, Dataset/EvalItem, the metric ABCs. The contracts an
                  *extension* implements (Recipe, Loss, Batch) are a different set and
                  live in ``training``: they are about how a model is fitted, which
                  nothing here needs to know.
    data          Contour, Frames, NoteEvent, Sources -- the values that cross every
                  seam, plus the serde and cache that persist them.
    dsp           features (CQT), transforms (separation), grids. The frame grid is
                  defined here because targets, cues and metrics must all agree on it.
    impls         the torch-free implementations of those protocols --
                  ``ClassicalEncoder``, ``PyinPitch``, ``GreedyDecoder``, the metrics,
                  the datasets. Their learned counterparts wear the same contracts from
                  ``models``, which is what the contracts are *for*: ``Cache`` accepts
                  any ``PitchEstimator`` without knowing whether it runs a network.

``labels``, ``conditions`` and ``examples`` moved down from ``training`` for the same
reason each: the evaluation harness varies conditions and builds targets too, so they
were never training-specific -- ``conditions.py`` had been saying so in its own
docstring for some time.
"""
