"""``ChunkedSemiCRFLoss``: the semi-CRF NLL under truncated BPTT.

The same objective as ``SemiCRFLoss`` -- ``log Z - score(gold)`` over a whole track --
but paid for one chunk at a time, so a track never has to fit in memory at once. That
is what buys the gradient steps: whole tracks force B=2 and 17 steps/epoch on this
corpus, of which ~40% of the compute is padding to the longest track in the batch;
fixed chunks fit B=8+ with no padding at all and give ~13x the steps for the same DP
work.

**Why this is a separate ``Loss`` and not a flag on ``SemiCRFLoss``.** It is stateful.
The alpha window from chunk *k* is what chunk *k+1* resumes from, so ``compute`` is no
longer a pure function of its batch -- it carries state between calls, keyed by stream.
Bolting that onto the whole-track loss would make the stateless path pay for a
concept it does not have, and the two change for different reasons.

**The telescoping identity** (see ``forward.chunk_log_partition``) is what makes this
exact rather than an approximation. With ``A_0 = 0`` and the carry detached between
chunks::

    sum_k [ (A_k - A_{k-1}) - S_k(gold) ]  ==  log Z - score(gold)

so the per-chunk losses sum to the true whole-track NLL, while each term's gradient
reaches only its own chunk's encoder activations. Detaching the carry is precisely
what makes it *truncated*: passing the window through undetached reproduces
whole-track backprop and the memory saving with it.

**Why the carry is keyed by stream and not by batch row.** With ``num_workers > 1``
consecutive batches come from different workers, so row *b* of one batch is not the
continuation of row *b* of the last. ``ChunkedSequenceDataset`` stamps a globally
unique ``stream`` on every chunk for exactly this reason. Entries are reset by the
chunk's own ``reset`` flag, which the dataset raises on each track's first chunk --
so a fresh pass over the loader (a new epoch, or the val loader) rebuilds every carry
before reading it, and no epoch resumes from the last one's state.
"""

import torch
from torch import nn

from stemify.decoders.semicrf import SemiCRFDecoder, TensorCues
from stemify.decoders.semicrf.window import initial_window
from stemify.training.semicrf.tbptt.dataset import ChunkBatch
from stemify.training.base import Loss


class ChunkedSemiCRFLoss(Loss):
    """The semi-CRF NLL over ordered chunks, carrying alpha between them."""

    def __init__(self, decoder: SemiCRFDecoder) -> None:
        self.decoder = decoder
        # stream id -> (alpha window, cumulative A) at that stream's current position.
        # Detached: this is resumption state, never a gradient path.
        self._carry: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def parameters(self):
        return self.decoder.parameters()

    def state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.decoder.state_dict().items()}

    def load_state_dict(self, state):
        self.decoder.load_state_dict(state)

    def to(self, device: torch.device) -> "ChunkedSemiCRFLoss":
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

        cues_list = [
            TensorCues.from_logits(
                batch.contours[b], onset[b], batch.frame_times[b],
                voicing[b] if voicing is not None else None,
            )
            for b in range(len(batch))
        ]
        carry, prev_A = self._resume(batch, device)
        A, gold, window = self.decoder.nll_chunk(
            cues_list, batch.golds, carry, batch.context, batch.right
        )
        # The telescoping term. Mean over rows so the scale matches the whole-track
        # loss and the learning rate carries over between the two paths.
        nll = ((A - prev_A) - gold).mean()

        detached_A, detached_window = A.detach(), window.detach()
        for b, stream in enumerate(batch.streams):
            self._carry[stream] = (detached_window[b], detached_A[b])

        scalar = float(nll.item())
        return nll, {"nll": scalar, "total": scalar}

    def batch_size(self, batch: ChunkBatch) -> int:
        return len(batch)
