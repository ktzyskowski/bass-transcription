"""The lower layer, mid-dissolution: what is left of it, and where each piece is going.

**This package is being taken apart.** It was defined *negatively* -- "does not depend on
a model" -- and a negative definition cannot refuse anything, which is how it reached
1800 lines of loose modules and two god-objects (``cache``, fan-out 6; ``examples``,
fan-out 8) whose consumers must stub them to test anything. It is being replaced by
vertical pillars, each defined by what it *is* and each detachable and testable on its
own. ``music`` (notes, grids, labels) is out; ``audio``, ``corpus``, ``persistence`` and
``contracts`` follow. ``tests/test_layers.py`` carries the running order.

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
    data          Contour, Frames, Sources -- the values that cross every seam, plus
                  the serde and cache that persist them.
    dsp           features (CQT), transforms (separation). Both name a specific
                  producer, which is what will make them the ``audio`` pillar; the
                  grid *arithmetic* they share has already gone to ``music.grids``.
    impls         the torch-free implementations of those protocols --
                  ``ClassicalEncoder``, ``PyinPitch``, ``GreedyDecoder``, the metrics,
                  the datasets. Their learned counterparts wear the same contracts from
                  ``models``, which is what the contracts are *for*: ``Cache`` accepts
                  any ``PitchEstimator`` without knowing whether it runs a network.

``conditions`` and ``examples`` moved down from ``training`` because the evaluation
harness varies conditions too, so neither was ever training-specific --
``conditions.py`` had been saying so in its own docstring for some time. ``labels``
came down for the same reason and has since carried on into ``music``, which is the
pattern the rest of this package is following.
"""
