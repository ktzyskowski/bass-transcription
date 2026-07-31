"""Pitch sources: audio -> a ``Contour``.

``PitchEstimator`` is the interface, ``PyinPitch`` the DSP implementation, and
``NeuralPitch`` a trained pitch head wearing the same contract.

Only the first two are exported here. ``NeuralPitch`` lives in ``.neural`` and must be
imported explicitly -- it pulls in torch and a trained backbone, so keeping it out of
this ``__init__`` means the light DSP path never transitively imports them. Same guard
as ``encoders``, ``decoders`` and ``models`` use, and the reason this became a package
at all: a module cannot hold both a torch-free and a torch-bearing implementation
without every importer paying for the second.
"""

from stemify.pitch.base import PitchEstimator
from stemify.pitch.pyin import FMAX, FMIN, HOP_LENGTH, PyinPitch

__all__ = ["FMAX", "FMIN", "HOP_LENGTH", "PitchEstimator", "PyinPitch"]
