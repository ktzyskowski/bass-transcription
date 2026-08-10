"""Where the process meets the machine: the filesystem and the accelerator.

The pillar defined by what it *touches* rather than what it knows. Nothing here knows
what a note is, what a CQT is, or what a corpus is -- these are the two places the
project makes a syscall or asks the hardware a question, and they are here so that every
layer can reach them without reaching through a layer that has opinions.

    atomic  ``atomic_write`` -- write to a temp path, rename into place
    device  ``best_device``  -- cuda > mps > cpu, probed once

Both are needed from *both directions*, which is the whole reason this pillar exists.
``audio.transforms`` writes a separated stem atomically and probes for an accelerator to
run Demucs on; ``training.checkpoint`` writes weights atomically and ``training.trainer``
probes for the same accelerator. Putting either in ``audio`` would make
``training.checkpoint`` depend on audio processing; putting them in ``persistence``
would invert the pillar order, since ``audio`` is beneath it.

**The membership test is negative and that is deliberate** -- but it is a negative about
*knowledge*, not about position, which is what makes it able to refuse things. ``serde``
touches the filesystem too and does not belong here, because it knows what a ``Contour``
is. ``cache`` touches it and knows what a corpus is. Both are domain code that happens
to persist; these two are the persistence with no domain in them.

``best_device`` guards its torch import inside the function rather than at module scope.
The light audio path (Demucs, the CQT) reaches this module and must not pull torch in
just by naming it -- and ``tests/test_layers.py`` asserts exactly that.
"""
