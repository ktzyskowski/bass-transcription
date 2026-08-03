"""Training scaffolding for the learned segmenter heads.

Onset, voicing and (optionally) pitch are the *same* problem -- per-frame
sequence tagging on the shared CQT front-end (``stemify.features``) -- so they
share one encoder and differ only in the head and the target. That is the whole
design:

    features (CQT)            stemify.features.cqt, on the 11.6 ms frame grid
    targets                   labels.py -- one builder per head, same grid
    model                     stemify.encoders -- backbone (TCN) + swappable heads
    data                      dataset.py -- SlakhItem -> (features, targets)
    fit                       trainer.py -- head-agnostic loop

The sequenced plan (see the note-level findings) trains **onset first** -- it is
the biggest lever and its labels are exact MIDI onsets with no gate-fidelity
question -- so ``CueNet`` is built to train a single head, not forced into
joint multi-task. Voicing follows once the reference voicing gate is validated
against MDB's annotations; pitch only if it beats pyin, which today it does not.

Every head is evaluated by the existing note-level harness: a learned head drops
in as a ``Segmenter`` and is scored by ``PRF`` and ``Fragmentation`` against the
same DSP floor the heuristic set.
"""