"""``SemiCRFLoss``: the semi-CRF's structured NLL, over ordered chunks of a track.

``log Z - score(gold)`` over a whole track, paid for one chunk at a time, so a track
never has to fit in memory at once. That is what buys the gradient steps: whole tracks
force B=2 and 17 steps/epoch on this corpus, of which ~40% of the compute is padding to
the longest track in the batch; fixed chunks fit B=8+ with no padding at all and give
~13x the steps for the same DP work. There used to be a separate whole-track ``Loss``
and dataset for the padded form; ``detach_carry`` is what remains of it.

**The telescoping identity** (see ``forward.chunk_log_partition``) is what makes this
exact rather than an approximation. With ``A_0 = 0`` and the carry detached between
chunks::

    sum_k [ (A_k - A_{k-1}) - S_k(gold) ]  ==  log Z - score(gold)

so the per-chunk losses sum to the true whole-track NLL, while each term's gradient
reaches only its own chunk's encoder activations. Detaching the carry is precisely
what makes it *truncated*: passing the window through undetached reproduces whole-track
backprop, and gives back the memory saving with it. That is the one axis on which the
deleted path differed, so it is a flag here rather than a second module.

**Why the carry is keyed by stream and not by batch row.** With ``num_workers > 1``
consecutive batches come from different workers, so row *b* of one batch is not the
continuation of row *b* of the last. ``ChunkDataset`` stamps a globally unique
``stream`` on every chunk for exactly this reason. Entries are reset by the chunk's
own ``reset`` flag, which the dataset raises on each track's first chunk --
so a fresh pass over the loader (a new epoch, or the val loader) rebuilds every carry
before reading it, and no epoch resumes from the last one's state.
"""

from collections.abc import Iterable

import torch
from torch import nn

from stemify.decoders.semicrf import SemiCRFDecoder, TensorCues
from stemify.decoders.semicrf.window import initial_window
from stemify.training.base import Loss
from stemify.training.semicrf.dataset import ChunkBatch


class SemiCRFLoss(Loss[ChunkBatch]):
    """The semi-CRF NLL over ordered chunks, carrying alpha between them."""

    def __init__(self, decoder: SemiCRFDecoder, detach_carry: bool = True) -> None:
        """``detach_carry=False`` stops truncating: the alpha window is passed to the
        next chunk with its graph intact, which reproduces whole-track backprop and
        whole-track activation memory. Only sensible for short tracks -- it is why the
        padded whole-track path was removed rather than kept as a second module."""
        self.decoder = decoder
        self.detach_carry = detach_carry
        # stream id -> (alpha window, cumulative A) at that stream's current position.
        # Resumption state; a gradient path only when detach_carry is False.
        self._carry: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def parameters(self) -> Iterable[nn.Parameter]:
        return self.decoder.parameters()

    def state_dict(self) -> dict[str, torch.Tensor]:
        # The ~15 potentials ARE the decoder. Without them a checkpoint restores an
        # encoder paired with an untrained CRF -- w_on=1, dur=zeros, i.e. no duration
        # preference at all, which is precisely the mechanism this objective exists
        # to learn.
        return {k: v.detach().cpu().clone() for k, v in self.decoder.state_dict().items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.decoder.load_state_dict(state)

    def to(self, device: torch.device) -> "SemiCRFLoss":
        self.decoder.to(device)
        return self

    def _resume(
        self, batch: ChunkBatch, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """The carry to resume from, and the previous cumulative ``A``, per row.

        A row flagged ``reset`` starts a fresh track: no window (the scan builds its
        own ``initial_window``) and ``A_{-1} = 0``. Rows are stacked into one batched
        window, so a batch mixing fresh and continuing streams is the normal case --
        which is why every chunk is the same width and shares one context.
        """
        if all(batch.resets):
            return None, torch.zeros(len(batch), device=device)
        missing = [
            s for s, r in zip(batch.streams, batch.resets, strict=True)
            if not r and s not in self._carry
        ]
        if missing:
            raise KeyError(
                f"streams {missing} continue a track but have no carry; the loader "
                "must emit a stream's chunks in order, with reset on the first"
            )
        # A reset row needs a window shaped like the others but carrying no history.
        # `initial_window` is that: NEG everywhere except START, which sits in the last
        # slot -- exactly the position "no frames consumed yet" occupies once the scan
        # starts at `context`. The chunk's leading zero-padding is then read against a
        # NEG alpha and contributes exp(NEG - max) == 0.
        template = next(iter(self._carry.values()))[0]   # (dmax, 3)
        fresh = initial_window(template, 1, template.shape[0])[0]
        windows, prev = [], []
        for stream, reset in zip(batch.streams, batch.resets, strict=True):
            if reset:
                windows.append(fresh)
                prev.append(torch.zeros((), device=device))
            else:
                window, a = self._carry[stream]
                windows.append(window)
                prev.append(a)
        return torch.stack(windows).to(device), torch.stack(prev).to(device)

    def compute(
        self, model: nn.Module, batch: ChunkBatch, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        out = model(batch.features.to(device))       # {head: (B, window, 1)}
        onset = out["onset"].squeeze(-1)
        voicing = out["voicing"].squeeze(-1) if "voicing" in out else None
        # (B, T, K) -- the CRF's path/spread features read these.
        pitch = out["pitch"] if "pitch" in out else None

        # No contour: nothing the scan reads needs one, and every recipe builds a
        # voicing head, so the fallback that used to want it is unreachable here.
        cues_list = [
            TensorCues.from_logits(
                None, onset[b], batch.frame_times[b],
                voicing[b] if voicing is not None else None,
                pitch[b] if pitch is not None else None,
            )
            for b in range(len(batch))
        ]
        carry, prev_A = self._resume(batch, device)
        A, gold, window = self.decoder.nll_chunk(
            cues_list, batch.golds, carry, batch.context, batch.right
        )
        # The telescoping term. Mean over rows so the loss is per-track rather than
        # per-batch, and the learning rate carries over from the frame path.
        nll = ((A - prev_A) - gold).mean()

        # Detaching here is the truncation itself: it severs chunk k+1's gradient from
        # chunk k's encoder activations, which is what bounds memory to one chunk.
        carried_A = A.detach() if self.detach_carry else A
        carried_window = window.detach() if self.detach_carry else window
        for b, stream in enumerate(batch.streams):
            self._carry[stream] = (carried_window[b], carried_A[b])

        scalar = float(nll.item())
        return nll, {"nll": scalar, "total": scalar}

