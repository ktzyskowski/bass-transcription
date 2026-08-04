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

from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import nn

from stemify.decoders.semicrf import SemiCRFDecoder, TensorCues
from stemify.decoders.semicrf.window import initial_window
from stemify.models.heads import frame_bce, pitch_ce
from stemify.training.base import Loss
from stemify.training.semicrf.dataset import ChunkBatch

# Which per-frame loss each auxiliary head is scored with. A table rather than an
# if-chain so that "is there a term for this head?" and "which one?" are the same
# lookup -- the constructor validates against exactly the keys `_aux_term` dispatches
# on, so a head can never be declared, accepted, and then silently not scored.
AUX_TERMS = {"pitch": "categorical", "onset": "binary", "voicing": "binary"}


class SemiCRFLoss(Loss[ChunkBatch]):
    """The semi-CRF NLL over ordered chunks, carrying alpha between them."""

    def __init__(
        self,
        decoder: SemiCRFDecoder,
        detach_carry: bool = True,
        aux_heads: Sequence[str] = (),
        aux_weights: Mapping[str, float] | None = None,
        pos_weight: Mapping[str, float] | None = None,
    ) -> None:
        """``detach_carry=False`` stops truncating: the alpha window is passed to the
        next chunk with its graph intact, which reproduces whole-track backprop and
        whole-track activation memory. Only sensible for short tracks -- it is why the
        padded whole-track path was removed rather than kept as a second module.

        ``aux_heads`` are heads the NLL supervises too weakly, each needing a
        per-frame term. Two reasons a head lands here, and they are different:

        ``pitch`` -- the NLL *cannot* supervise it. The span features read how far the
        distribution moves across a span and how much its frames disagree, both
        invariant to what pitch it actually names, so nothing constrains the head's
        values and without this term it trains on no signal at all.

        ``voicing`` (and ``onset``, available but not declared by default) -- the NLL
        *can* supervise it, but only competitively. The objective is discriminative
        over segmentations, so a feature earns gradient in proportion to how much it
        helps rank gold above the alternatives, and ``LinearSpanScorer`` weights every
        feature with a learned scalar. Voicing is largely redundant given onset
        evidence, the duration prior and the transition table -- its unique
        contribution is spotting rests, which are rare -- so ``w_v``/``w_u`` can shrink
        toward zero and the head is left orphaned. Measured: voicing AUROC **0.516**
        (chance) under the structured objective alone, against **0.984** for the same
        head under frame BCE. Nothing here forces voicing to *be* voicing; the
        auxiliary term does.

        ``pos_weight`` is the binary terms' class weight, exactly as on the frame path:
        an auxiliary onset BCE is the *same term* there and here, so the imbalance it
        corrects (onsets are ~7% of frames after +-1 widening) is the same imbalance.
        This objective refused the flag outright until the binary terms existed, on the
        true-at-the-time grounds that a structured NLL takes no class weights.

        ``aux_weights`` is per head because the terms are not interchangeable: pitch is
        a categorical CE over 216 bins scored on voiced frames, the binary heads are
        BCE over every frame. One shared coefficient would mean whichever head was
        tuned last.

        Declared rather than inferred from what the batch happens to carry. "Score it
        if a pitch target turns up" is the same shape as the bug where ``Cues`` lacked
        ``pitch_logits`` and the span features silently read zero: the failure is a
        term quietly not applied, which no loss curve shows.
        """
        self.decoder = decoder
        self.detach_carry = detach_carry
        self.aux_heads = tuple(aux_heads)
        self.pos_weight = dict(pos_weight or {})
        weights = dict(aux_weights or {})
        self.aux_weights = {h: float(weights.get(h, 1.0)) for h in self.aux_heads}
        unsupported = [h for h in self.aux_heads if h not in AUX_TERMS]
        if unsupported:
            raise NotImplementedError(
                f"no auxiliary term for head(s) {unsupported}; have {sorted(AUX_TERMS)}"
            )
        stray = sorted(set(weights) - set(self.aux_heads))
        if stray:
            raise ValueError(
                f"aux_weights names head(s) {stray} that are not in aux_heads "
                f"{sorted(self.aux_heads)}. A weight for a term that is never computed "
                "is a setting that silently does nothing -- the failure this file's "
                "`aux_heads` docstring exists to refuse."
            )
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
        # per-batch, then divided by the frames it scored so it is a *rate*.
        #
        # The division is what makes this composable. `log Z - score(gold)` is the
        # surprisal of one structured object -- a whole chunk's segmentation -- while
        # every per-frame head reports a mean. Adding a total to an average is
        # dimensionally incoherent, and no choice of weight repairs it: the weight just
        # absorbs the frame count, which is why the measured gradient ratio into the
        # shared trunk was ~1200:1 and the pitch term contributed under 0.1% of it.
        # Per scored feature frame, both sides are nats about the same 11.6 ms of
        # audio, so a weight is a ratio of information rates and `1.0` is a claim
        # someone can argue with.
        #
        # Feature frames rather than DP frames on purpose: the heads emit on the
        # feature grid, so normalising the NLL to the DP grid would leave a factor of
        # `stride` hidden inside every head weight.
        scored = self.scored_frames(batch, batch.features.shape[-1])
        nll = ((A - prev_A) - gold).mean() / scored

        # Detaching here is the truncation itself: it severs chunk k+1's gradient from
        # chunk k's encoder activations, which is what bounds memory to one chunk.
        carried_A = A.detach() if self.detach_carry else A
        carried_window = window.detach() if self.detach_carry else window
        for b, stream in enumerate(batch.streams):
            self._carry[stream] = (carried_window[b], carried_A[b])

        total = nll
        metrics = {"nll": float(nll.item())}
        for head in self.aux_heads:
            term = self._aux_term(head, out, batch, device)
            total = total + self.aux_weights[head] * term
            metrics[head] = float(term.item())
        metrics["total"] = float(total.item())
        return total, metrics

    def _aux_term(
        self,
        head: str,
        out: Mapping[str, torch.Tensor],
        batch: ChunkBatch,
        device: torch.device,
    ) -> torch.Tensor:
        """The per-frame auxiliary loss for one head, over the chunk's *own* frames.

        **Why the slice.** The window is ``dmax + chunk + right`` DP frames wide: the
        leading ``context`` is the previous chunk's audio, present so a span reaching
        back across the boundary can read its cues, and the trailing ``right`` is the
        next chunk's, present so the encoder's receptive field does not see zero-padding
        at the edge. Both are scored again as part of *their own* chunk. Scoring the
        whole window would therefore train roughly two thirds of every frame twice per
        epoch, at a weight that depends on where a track's chunk boundaries happened to
        fall -- which is a data-dependent reweighting, not a regulariser.

        NaN padding past a track's end needs no handling here: ``target_windows`` writes
        NaN there and both terms mask on it -- ``pitch_ce`` because unvoiced frames have
        no pitch to learn, ``frame_bce`` because unmasked NaN makes the whole term NaN.
        """
        if head not in out:
            raise ValueError(
                f"auxiliary head {head!r} was requested but the model has no such "
                f"branch (has {sorted(out)}), so there are no logits to score. The "
                "recipe builds the branches from the same flags that declare the "
                "auxiliary terms -- a checkpoint or recipe config where the two "
                "disagree is the cause."
            )
        if head not in batch.y:
            raise KeyError(
                f"auxiliary head {head!r} was requested but the batch carries no such "
                f"target (has {sorted(batch.y)}). `SemiCRFRecipe.aux_heads` drives both "
                "what the dataset caches and what this scores; they have drifted."
            )
        logits = out[head]
        lo, hi = self.scored_slice(batch, logits.shape[1])
        target = batch.y[head][:, lo:hi].to(device)
        if AUX_TERMS[head] == "categorical":
            return pitch_ce(logits[:, lo:hi], target)
        return frame_bce(logits[:, lo:hi], target, self.pos_weight.get(head))

    def scored_slice(self, batch: ChunkBatch, frames: int) -> tuple[int, int]:
        """The feature-frame half-open range the NLL actually scores.

        One place, because every auxiliary term must use *this* range and nothing else.
        A term that averaged over the whole window would divide by a different frame
        count than the NLL does, so the shared "nats per frame" unit -- the only thing
        that makes the head weights interpretable -- would silently stop holding
        between the terms. It would also supervise the margins twice per epoch, since
        one chunk's context is the previous chunk's scored region.

        ``stride`` converts the DP grid (where ``context``/``right`` are counted) to
        the feature grid the heads emit on.
        """
        stride = self.decoder.stride
        return batch.context * stride, frames - batch.right * stride

    def scored_frames(self, batch: ChunkBatch, frames: int) -> int:
        """How many feature frames ``scored_slice`` covers -- the denominator that puts
        the structured NLL on the same per-frame footing as the per-frame heads."""
        lo, hi = self.scored_slice(batch, frames)
        return max(hi - lo, 1)

