"""Pick the compute device -- the one place cuda > mps > cpu is decided.

A leaf of ``core`` so the torch-bearing trainer/encoder and the light audio
transforms path both agree without either importing the other. The torch import
is guarded, not top-level, so pulling this into the audio path never forces torch
to load: Demucs defaults to ``"cpu"`` and never probes for an accelerator, so
anything relying on its default runs on CPU without saying so -- this makes the
probe explicit and shared.
"""


def best_device() -> str:
    """The fastest accelerator torch can reach here (``cuda`` > ``mps`` > ``cpu``)."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
