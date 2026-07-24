"""Compute-device selection for the trainer -- re-exported from the top-level leaf.

Kept as a training-local name so the trainer and learned segmenter can import it
from here, but the logic lives in ``stemify.device`` so the light audio path
(transforms) shares the exact same probe without a training dependency."""

from stemify.device import best_device

__all__ = ["best_device"]
