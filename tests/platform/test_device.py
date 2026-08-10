"""``best_device`` -- the accelerator preference, and the import it must not force.

Two properties. One is the preference order, which is only interesting because getting
it wrong is invisible: Demucs and torch both default to CPU quite happily, so a mistake
here costs a large constant factor and reports nothing.

The other is structural, and is the reason this module sits in ``platform`` rather than
beside the trainer -- ``audio.transforms`` names it at module scope, and the audio path
must stay reachable without loading torch.
"""

import subprocess
import sys

import pytest

from stemify.platform import device


def test_it_returns_something_torch_can_actually_use():
    """No mocking: whatever this box has, the answer has to be one of the three names
    torch accepts, on the machine the suite is running on."""
    assert device.best_device() in {"cuda", "mps", "cpu"}


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"),
     (False, False, "cpu")],
)
def test_cuda_wins_over_mps_wins_over_cpu(monkeypatch, cuda, mps, expected):
    """The preference order, including the case both are available -- which no single
    machine can exercise, and which is exactly where a reordering would go unnoticed."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert device.best_device() == expected


def test_naming_this_module_does_not_import_torch():
    """The structural property, in a subprocess because the suite has torch loaded.

    ``audio.transforms`` imports ``best_device`` at module scope so it can run Demucs on
    an accelerator, and the audio path -- separation, the CQT -- is deliberately usable
    without a network. A top-level ``import torch`` here would put a multi-second import
    behind every CLI that merely touches a transform, and would break the torch-free
    guarantee ``tests/test_layers.py`` holds over ``platform`` and ``audio``.
    """
    probe = (
        "import sys; import stemify.platform.device; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", probe]).returncode == 0
