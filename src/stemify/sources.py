"""``Sources``: the audio a track offers an encoder, keyed by role.

The input peer of ``Cues``. Different encoders consume different audio -- the DSP
floor needs only the bass stem, a mix-conditioned neural encoder needs the bass
stem *and* the original mixture (to learn that a low transient in the bass which
lines up with a kick in the mix is bleed, not a re-pluck). A single ``audio_path``
cannot express that, so an encoder is handed a ``Sources`` and reads the roles it
needs.

This carries *only audio*. Pitch is not here: producing the pitch contour is the
encoder's own job (pyin today, a learned pitch head later), so it comes out in the
``Cues``, never in as a parameter -- that is what lets a learned-pitch encoder drop
in without changing the interface.

``bass`` is always present -- it is what gets transcribed, whether an isolated bass
DI was uploaded or a full mix was separated. ``mix`` is optional: present when the
input was a mixture, absent for an isolated-stem upload. An encoder that wants the
mix falls back to bass-only when it is ``None`` -- which stem-dropout training makes
robust, so "needs the mix in training, tolerates its absence at inference" is
exactly a backbone channel fed an optional source.

Populating this is free: demucs emits every stem in one pass and we keep the
original mixture, so the caller already has each role on hand. Extend with further
roles (``drums``, ``other``, ...) as experiments need them.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sources:
    """A track's audio by role: ``bass`` always, others as available."""

    bass: str | Path                     # the (separated) bass stem -- transcribed
    mix: str | Path | None = None        # original mixture, when the input was a mix
