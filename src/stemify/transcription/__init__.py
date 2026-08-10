"""The note pipeline: audio in, notes out, and the seam in the middle.

    Sources -> Encoder -> Frames -> Decoder -> [NoteEvent]

One shape, and every piece of it is here: the two contracts, the value that crosses
between them, and the torch-free implementations that make the whole path runnable with
no network at all.

    frames       ``Frames`` -- per-frame evidence, the seam
    encoders     ``Encoder``; ``ClassicalEncoder`` (spectral flux + pyin) is the floor
    decoders     ``Decoder``; ``GreedyDecoder`` peak-picks and merges
    pitch        ``PitchEstimator``; ``PyinPitch``, and the memoised contour
    transcriber  ``Transcriber`` -- the composition of one of each

**Contracts and implementations together, deliberately.** The pairing is what the
contracts are *for*: ``ClassicalEncoder`` and ``LearnedEncoder`` are substitutable
because they wear one interface, and ``GreedyDecoder`` and ``SemiCRFDecoder`` likewise.
Splitting the interface from its DSP sibling would leave a package of pure abstraction
with nothing to check it against -- and the floor is not scaffolding, it is the baseline
every learned result is quoted against.

**This sits below ``models``, and that is the load-bearing fact.** ``models`` implements
these contracts (``LearnedEncoder``, ``NeuralPitch``) and reads this ``Frames``, so
nothing here may know that torch exists. That is also why the *fitting* contracts are a
different layer and not this one: ``Recipe.build_model()`` returns a ``CueNet``, so it
depends on ``models`` where these are depended *on* by it. One package holding both
would import ``models`` while ``models`` imports it -- the cycle, relocated rather than
removed.

Below it: ``music`` for ``NoteEvent``/``Contour``, ``audio`` for ``Sources`` and the
CQT, ``persistence`` for the contour cache. Nothing else.
"""
