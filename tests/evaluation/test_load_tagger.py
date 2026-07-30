"""Rebuilding a trained model from a checkpoint alone.

The architecture has to come out of the file, not out of a default. ``in_channels``
is the case that made this matter: a mix-conditioned model has a wider first conv,
so a hardcoded ``TCN()`` cannot load one -- and the deploy path is exactly where
that would be discovered, long after training.
"""

import numpy as np
import torch

from stemify.contour import Contour
from stemify.encoders.neural import NeuralEncoder
from stemify.evaluation.cli import load_tagger
from stemify.features import FrameFeatures
from stemify.training.recipes import HEADS, FrameRecipe


def _save(tmp_path, recipe, **extra):
    path = tmp_path / "model.pt"
    torch.save(
        {"state_dict": recipe.build_model().state_dict(), "heads": HEADS,
         "encoder": "tcn", **extra},
        path,
    )
    return path


def _encode(model, n_channels, n=24):
    """Drive the rebuilt model the way deployment does, to prove it actually runs."""
    frame_times = np.arange(n) * 0.0116
    shape = (216, n) if n_channels == 1 else (n_channels, 216, n)
    return NeuralEncoder(model, device="cpu").encode_features(
        Contour(frame_times, np.full(n, 110.0)),
        FrameFeatures(np.random.rand(*shape).astype(np.float32), frame_times),
    )


def test_a_single_channel_checkpoint_round_trips(tmp_path):
    recipe = FrameRecipe(["stem-solo"])
    model = load_tagger(_save(tmp_path, recipe, in_channels=1))
    assert model.in_channels == 1
    assert _encode(model, 1).onset.shape == (24,)


def test_a_bichannel_checkpoint_round_trips(tmp_path):
    # Without reading in_channels back this raises on the first conv's weights.
    recipe = FrameRecipe(["demucs-mix"])
    model = load_tagger(_save(tmp_path, recipe, in_channels=2, conditions=["demucs-mix"]))
    assert model.in_channels == 2
    assert _encode(model, 2).onset.shape == (24,)


def test_a_checkpoint_predating_the_field_loads_as_single_channel(tmp_path):
    # Every checkpoint written before mix-conditioning existed was one channel, so
    # the absent field has an unambiguous meaning -- earlier models stay loadable.
    model = load_tagger(_save(tmp_path, FrameRecipe(["stem"])))
    assert model.in_channels == 1
