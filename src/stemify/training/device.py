"""Pick the compute device -- shared by the trainer and the learned segmenter.

Its own tiny module so the segmenter can agree with the trainer on cuda > mps >
cpu without importing the whole trainer just for this."""

import torch


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
