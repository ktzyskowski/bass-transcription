"""crf_loss: the whole-track model -> TensorCues -> NLL training step.

A fake tagger emits fixed onset/voicing logit curves scaled by one learnable gain,
so a gradient step through crf_loss must move a *model* parameter -- proof the
structured loss trains the encoder end to end, no audio and no real backbone.
"""

import numpy as np
import torch
from torch import nn

from stemify.contour import Contour
from stemify.decoders.semicrf import SemiCRFDecoder
from stemify.notes import NoteEvent
from stemify.training.semicrf import crf_loss

DT = 0.0116
T = 240
TIMES = np.arange(T) * DT


class _ScaledHeads(nn.Module):
    """Fixed onset/voicing logit curves times one learnable gain -- a stand-in tagger
    whose single parameter must receive gradient for the loss to train."""

    def __init__(self, onset_logit: np.ndarray, voicing_logit: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("onset", torch.from_numpy(onset_logit).float())
        self.register_buffer("voicing", torch.from_numpy(voicing_logit).float())
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):  # x: (1, 1, n_bins, T)
        t = x.shape[-1]
        return {
            "onset": (self.gain * self.onset).view(1, t, 1),
            "voicing": (self.gain * self.voicing).view(1, t, 1),
        }


def _track(spans, pitches_hz):
    """Onset logit spiking at each note start, voicing logit high inside notes, a
    contour holding each pitch, and the reference notes -- the whole-track inputs."""
    onset = np.full(T, -6.0, dtype=np.float32)
    voicing = np.full(T, -6.0, dtype=np.float32)
    freqs = np.zeros(T, dtype=np.float32)
    for (a, b), hz in zip(spans, pitches_hz):
        onset[round(a / DT)] = 6.0
        voicing[(TIMES >= a) & (TIMES < b)] = 6.0
        freqs[(TIMES >= a) & (TIMES < b)] = hz
    contour = Contour(times=TIMES, freqs=freqs)
    notes = [NoteEvent(onset=a, offset=b, midi_pitch=45.0) for a, b in spans]
    features = np.zeros((8, T), dtype=np.float32)
    return onset, voicing, contour, notes, features


def test_crf_loss_is_finite_and_differentiable():
    onset, voicing, contour, notes, features = _track([(0.2, 0.9), (1.2, 1.9)], [110, 146])
    model = _ScaledHeads(onset, voicing)
    loss = crf_loss(model, SemiCRFDecoder(), features, TIMES, contour, notes)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_crf_loss_step_trains_a_model_parameter():
    onset, voicing, contour, notes, features = _track([(0.2, 0.9), (1.2, 1.9)], [110, 146])
    model = _ScaledHeads(onset, voicing)
    decoder = SemiCRFDecoder()
    opt = torch.optim.Adam([*model.parameters(), *decoder.parameters()], lr=0.1)

    first = crf_loss(model, decoder, features, TIMES, contour, notes).item()
    for _ in range(40):
        opt.zero_grad()
        loss = crf_loss(model, decoder, features, TIMES, contour, notes)
        loss.backward()
        assert model.gain.grad is not None and model.gain.grad.abs().item() > 0
        opt.step()
    assert crf_loss(model, decoder, features, TIMES, contour, notes).item() < first


def test_crf_loss_without_voicing_head():
    # An onset-only tagger: the decoder falls back to the contour's voicing.
    onset, _, contour, notes, features = _track([(0.2, 0.9), (1.2, 1.9)], [110, 146])

    class _OnsetOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("onset", torch.from_numpy(onset).float())
            self.gain = nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            t = x.shape[-1]
            return {"onset": (self.gain * self.onset).view(1, t, 1)}

    loss = crf_loss(_OnsetOnly(), SemiCRFDecoder(), features, TIMES, contour, notes)
    assert torch.isfinite(loss) and loss.requires_grad
