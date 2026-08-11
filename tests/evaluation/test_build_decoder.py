"""Building the *scoring* decoder from a checkpoint.

A ``SemiCRFDecoder``'s potentials are the model. A freshly built one has a zeroed
duration embedding and a zeroed label readout -- no preference at all, which is the
mechanism the structured objective exists to learn -- and it still emits notes. So
scoring a checkpoint through one produces a plausible note F1 for a model nobody
trained, which is the worst shape of failure available in an experiment loop.

The eval CLI wrote ``loss_state`` at train time and never read it back; these pin that
it does now, and that the cases where it cannot are errors rather than defaults.
"""

import argparse

import pytest
import torch

from stemify.transcription.greedy import GreedyDecoder
from stemify.evaluation.cli import build_decoder
from stemify.semicrf.fit.recipe import SemiCRFRecipe
from stemify.training.checkpoint import Checkpoint


def _args(**kw) -> argparse.Namespace:
    base = {"decoder": "semicrf", "encoder": "learned", "checkpoint": None}
    return argparse.Namespace(**{**base, **kw})


def _trained_checkpoint() -> Checkpoint:
    """A checkpoint whose potentials are visibly not the constructor's defaults."""
    recipe = SemiCRFRecipe(["clean"])
    with torch.no_grad():
        recipe.decoder.scorer.label_bias.fill_(0.31)
        recipe.decoder.scorer.rest_logprob.fill_(-0.77)
        dur = recipe.decoder.scorer.dur.weight
        dur.copy_(torch.linspace(-1.0, 1.0, dur.numel()).reshape(dur.shape))
    return Checkpoint(
        recipe=recipe,
        model_state=recipe.build_model().state_dict(),
        objective_state={
            k: v.detach().clone() for k, v in recipe.decoder.state_dict().items()
        },
    )


def test_semicrf_decoder_is_restored_from_the_checkpoint(tmp_path):
    checkpoint = _trained_checkpoint()
    path = tmp_path / "semicrf.pt"
    checkpoint.save(path)

    built = build_decoder(_args(checkpoint=path)).state_dict()
    for key, want in checkpoint.objective_state.items():
        assert torch.allclose(built[key], want), f"{key} came back at its default"
    # The duration prior specifically: all-zeros is "no preference", and that is what
    # a fresh decoder has. If this passes with zeros the restore did nothing.
    zeros = torch.zeros_like(built["scorer.dur.weight"])
    assert not torch.allclose(built["scorer.dur.weight"], zeros)


def test_semicrf_without_a_checkpoint_is_refused():
    with pytest.raises(ValueError, match="needs --checkpoint"):
        build_decoder(_args(checkpoint=None))


def test_a_frame_checkpoint_scored_as_semicrf_is_refused(tmp_path):
    # A --objective frame checkpoint has no potentials at all. Silently supplying
    # untrained ones would report the semi-CRF's numbers for a CRF that never trained.
    path = tmp_path / "frame.pt"
    torch.save({"state_dict": {}, "heads": ["onset", "voicing"]}, path)
    with pytest.raises(ValueError, match="no objective parameters"):
        build_decoder(_args(checkpoint=path))


def test_a_version_1_semicrf_checkpoint_is_refused_by_name(tmp_path):
    # Every v1 semi-CRF file was trained against the retired log-linear scorer, and v1
    # recorded no scorer configuration to rebuild anything else from. Loading one at
    # today's defaults would put weights fitted for other features into this scorer.
    path = tmp_path / "v1.pt"
    torch.save(
        {"state_dict": {}, "heads": ["onset"], "loss_state": {"scorer.w_on": 0.3}},
        path,
    )
    with pytest.raises(ValueError, match="version-1 semi-CRF checkpoint"):
        build_decoder(_args(checkpoint=path))


def test_heuristic_still_needs_no_checkpoint():
    assert isinstance(build_decoder(_args(decoder="heuristic")), GreedyDecoder)
