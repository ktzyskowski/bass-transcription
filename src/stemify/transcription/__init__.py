"""The note pipeline: audio in, notes out, and the seam in the middle.

    Sources -> Encoder -> Frames -> Decoder -> [NoteEvent]

One shape, and every piece of it is here: the two contracts, the value that crosses
between them, and the torch-free implementations that make the whole path runnable with
no network at all.

    frames       ``Frames`` -- per-frame evidence, the seam
    encoders     ``Encoder``; the implementation is ``models.LearnedEncoder``
    decoders     ``Decoder``; ``GreedyDecoder`` peak-picks and merges
    transcriber  ``Transcriber`` -- the composition of one of each

**The decoders live beside their contract, the encoder does not.** ``GreedyDecoder``
and ``SemiCRFDecoder`` are both here, substitutable over one ``Frames``, and the
package is testable end to end because of it. ``Encoder`` is a contract with no
implementation in this layer at all -- its one implementation holds a ``FrameNet``, so
it is a layer up. That asymmetry used to be filled by ``ClassicalEncoder``, the pyin
DSP floor; it went with pyin, and what it demonstrated is now asserted by
``tests/test_layers.py`` instead.

**This sits below ``models``, and that is the load-bearing fact.** ``models`` implements
these contracts (``LearnedEncoder``) and reads this ``Frames``, so
nothing here may know that torch exists. That is also why the *fitting* contracts are a
different layer and not this one: ``Recipe.build_model()`` returns a ``FrameNet``, so it
depends on ``models`` where these are depended *on* by it. One package holding both
would import ``models`` while ``models`` imports it -- the cycle, relocated rather than
removed.

Below it: ``music`` for ``NoteEvent``/``Contour``, ``audio`` for ``Sources`` and the
CQT. Nothing else.
"""
