"""The corpora: what recordings exist, and what is true of them.

One pillar, one question. A loader here answers *what is on disk and what is annotated
about it* -- and nothing else. It does not know what a CQT is, what a model is, or how
anything gets cached; it hands over paths and truth, and every consumer decides what to
do with them.

    base           the contracts -- ``AudioItem`` (one scorable unit) and ``Dataset``
    slakh/         synthesised multitrack MIDI: bass stems beside their mixtures
    filobass       transcribed jazz double bass
    idmt_smt_bass  single-note clips, annotated by string and playing technique
    mdb_bass_synth frame-annotated bass, resynthesised
    midi_contour   MIDI note events -> a frame-level ``Contour``, with the decay gate

Four corpora with four directory layouts, four annotation formats and two kinds of truth
(frame-level contours, note-level events) reduced to one interface. That reduction is
the pillar's whole product: the evaluation harness, the trainer and the feature cache
each accept a ``Dataset`` and none of them contains a branch on which one it got.

**References are lazy.** ``reference()`` and ``reference_notes()`` are methods, not
fields, because building one can cost a MIDI parse per item and a run that filters most
items out should not pay for all of them.

**Grouping is the item's business.** An item declares its own ``groups`` -- "string 1",
"plucking FS", "split test" -- and the report machinery slices by whatever keys it
finds. That is how per-string and per-technique breakdowns exist without the evaluation
layer knowing anything about bass guitars.

Below it: ``music`` for the truth values (``NoteEvent``, ``Contour``) and ``audio`` for
``Sources``, which is how an item offers more than one rendering of the same music --
the bass stem and the mixture it sits inside, which is exactly what a two-channel
encoder reads.
"""
