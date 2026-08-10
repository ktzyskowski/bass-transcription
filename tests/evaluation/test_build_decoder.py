"""Building the *scoring* decoder from a checkpoint.

A ``SemiCRFDecoder``'s potentials are the model. A default-constructed one has
``w_on=1`` and ``dur=zeros`` -- no duration preference at all, which is the mechanism
the structured objective exists to learn -- and it still emits notes. So scoring a
checkpoint through one produces a plausible note F1 for a model nobody trained, which
is the worst shape of failure available in an experiment loop.

The eval CLI wrote ``loss_state`` at train time and never read it back; these pin that
it does now, and that the cases where it cannot are errors rather than defaults.
"""

import argparse

import pytest
import torch

from stemify.transcription.decoders.greedy import GreedyDecoder
from stemify.semicrf import SemiCRFDecoder
from stemify.evaluation.cli import build_decoder


def _args(**kw) -> argparse.Namespace:
    base = {"decoder": "semicrf", "encoder": "learned", "checkpoint": None}
    return argparse.Namespace(**{**base, **kw})


def _trained_state() -> dict:
    """Potentials that are visibly not the constructor's defaults."""
    decoder = SemiCRFDecoder()
    with torch.no_grad():
        decoder.scorer.w_on.fill_(0.31)
        decoder.scorer.b_note.fill_(-0.77)
        decoder.scorer.dur.copy_(torch.linspace(-1.0, 1.0, decoder.scorer.dur.numel()))
    return {k: v.detach().clone() for k, v in decoder.state_dict().items()}


def test_semicrf_decoder_is_restored_from_the_checkpoint(tmp_path):
    state = _trained_state()
    path = tmp_path / "semicrf.pt"
    torch.save({"state_dict": {}, "heads": ["onset"], "loss_state": state}, path)

    built = build_decoder(_args(checkpoint=path)).state_dict()
    for key, want in state.items():
        assert torch.allclose(built[key], want), f"{key} came back at its default"
    # The duration prior specifically: all-zeros is "no preference", and that is what
    # a fresh decoder has. If this passes with zeros the restore did nothing.
    assert not torch.allclose(built["scorer.dur"], torch.zeros_like(built["scorer.dur"]))


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


def test_heuristic_still_needs_no_checkpoint():
    assert isinstance(build_decoder(_args(decoder="heuristic")), GreedyDecoder)
