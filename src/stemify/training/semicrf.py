"""End-to-end semi-CRF training: the forward that ties the encoder to the decoder.

The frame-head ``Trainer`` (``trainer.py``) fits per-frame BCE over fixed windows.
The semi-CRF trains a *different* objective over **whole tracks**: the decoder's
segmental negative log-likelihood, with the gradient flowing back through the frame
heads so the encoder learns to emit onsets/voicing the structured decoder can parse.

``crf_loss`` is the seam -- one track in, one differentiable loss out. It is kept a
plain function (no optimiser, no dataset) so the model->TensorCues->NLL path is
unit-testable with a fake tagger and a synthetic track, no audio. The training loop
composes it: sum ``crf_loss`` over a minibatch of tracks, ``backward``, step.

The tagger is fully convolutional over time, so a whole track goes through in one
pass -- there is no windowing here, because the DP needs the entire sequence at once
(a note can be longer than any window). That is the structural reason this cannot
reuse ``FrameWindowDataset`` and gets its own whole-track path.
"""

import numpy as np
import torch

from stemify.contour import Contour
from stemify.decoders.semicrf import SemiCRFDecoder, TensorCues
from stemify.encoders.tagger import FrameTagger
from stemify.notes import NoteEvent


def crf_loss(
    model: FrameTagger,
    decoder: SemiCRFDecoder,
    features: np.ndarray,
    frame_times: np.ndarray,
    contour: Contour,
    notes: list[NoteEvent],
    device: torch.device | None = None,
) -> torch.Tensor:
    """The semi-CRF NLL for one whole track, differentiable through ``model``.

    Runs the tagger over the track's CQT in one pass, wraps the onset (and voicing,
    when present) head logits in a ``TensorCues`` -- grad-bearing, so the loss reaches
    the heads -- and scores the gold segmentation built from ``notes``. An optimiser
    step over ``model`` and ``decoder`` parameters trains the encoder and the CRF
    potentials jointly.

    ``contour`` (pyin) and ``frame_times`` are constants and stay on the CPU; the
    tagger and its logits move to ``device``. The decoder builds its own emission
    constants (the pitch-change cumsum) on the CPU, so a GPU loop must keep the whole
    step on one device -- reconciling that placement is the loop's job, deferred.
    """
    device = device or next(model.parameters()).device
    x = torch.as_tensor(np.ascontiguousarray(features)).float().to(device)
    out = model(x[None, None])  # (1, 1, n_bins, T) -> {head: (1, T, 1)}
    onset_logit = out["onset"].squeeze(0).squeeze(-1)  # (T,)
    voicing_logit = (
        out["voicing"].squeeze(0).squeeze(-1) if "voicing" in out else None
    )
    cues = TensorCues.from_logits(contour, onset_logit, frame_times, voicing_logit)
    return decoder.nll(cues, notes)
