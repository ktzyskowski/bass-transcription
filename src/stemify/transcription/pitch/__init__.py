"""Pitch sources: audio -> a ``Contour``.

``PitchEstimator`` is the interface and ``PyinPitch`` the DSP implementation. The
learned one, ``NeuralPitch``, is not here: it is a trained pitch head wearing this
contract, so it lives in ``stemify.models`` beside the ``CueNet`` it reads.

The contract staying here while the neural implementation sits a layer up is the point
of both -- ``PitchEstimator`` is what lets ``ensure_contour`` and ``ClassicalEncoder`` accept
either without importing torch to do it. This package used to keep that separation by
convention, exporting only the light pair; it is now a property of the layering, and
``tests/test_layers.py`` is what holds it.
"""

from stemify.transcription.pitch.base import PitchEstimator
from stemify.transcription.pitch.pyin import FMAX, FMIN, HOP_LENGTH, PyinPitch

__all__ = ["FMAX", "FMIN", "HOP_LENGTH", "PitchEstimator", "PyinPitch"]
