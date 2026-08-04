"""The training *loop's* knobs -- what every strategy needs, and nothing else.

Feature settings are not here: they live in ``stemify.features`` because they are
shared with the pyin baseline and the eval harness, and a model that trained on a
different CQT than it is evaluated on would be silently broken.

Strategy settings are not here either. Window length, window hop and per-head
``pos_weight`` are read by the frame recipe and by nothing else; the semi-CRF sets its
chunk length from ``CHUNK_FRAMES`` and its objective takes no class weights. A knob that
only one strategy reads belongs on that strategy's ``Recipe``, where handing it to a
different one raises -- while they sat here, ``--pos-weight onset=5`` on a semi-CRF run
was accepted and silently ignored, the failure mode this project keeps paying for.

The bar for adding a field: every strategy reads it. Anything else has a home that can
reject it.
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Optimisation.
    batch_size: int = 16
    learning_rate: float = 1e-3
    max_epochs: int = 50
    patience: int = 8  # early-stop after this many epochs without val gain

    # Which number picks the best checkpoint and drives early stopping.
    #
    # ``loss/<head>/val`` names an inline validation loss; anything else names a metric
    # from the note-level harness (``note/separated/f1``, ``cues/clean/onset_ap``, ...),
    # which the loop then has to score *synchronously*, because a checkpoint cannot be
    # chosen on a number that does not exist yet.
    #
    # The default is not the right choice, only the cheap one, and it has been caught
    # being wrong: a semi-CRF run scored `note/clean/f1` 0.8175 at epoch 0 and 0.4086 at
    # epoch 1, while `loss/total/val` went 20.04 -> 16.25 and called epoch 1 *better*.
    # A finished run would have restored the worse model and reported success. The
    # deployment number is `note/separated/f1`; monitoring it costs one harness pass per
    # epoch and removes that whole failure mode.
    monitor: str = "loss/total/val"
    monitor_mode: str = "min"  # "min" for losses, "max" for F1/AP/AUROC

    # Gradient-norm clip applied over the model *and* the objective's parameters,
    # i.e. everything the one optimiser moves. A safety net rather than a tuned
    # regulariser: a single unclipped outlier step sends the weights to inf
    # permanently, taking every head with it because they share a trunk, and a
    # structured NLL is the objective that produces those -- the log-partition over a
    # chunk whose evidence is unusual.
    #
    # The threshold has to sit *above* the run's ordinary norm, and that scale is a
    # property of the objective, not of this loop: a per-frame BCE and a per-chunk
    # structured NLL differ by orders of magnitude in loss and so in gradient. A
    # threshold below the ordinary norm does not protect anything -- it clips every
    # step to the same length, which discards exactly the magnitude difference that
    # distinguishes a divergent step from a normal one.
    #
    # The default is *not* yet inert for the semi-CRF: measured at ~1500 on the full
    # corpus, so this clips most steps. Kept anyway because a step-matched comparison
    # showed it changing the loss trajectory not at all (docs/experiments.md), and
    # raising it is a second variable in a run that is still diverging for another
    # reason. Tune from the ``grad_norm`` metric the loop logs, and beware sizing it
    # off a small smoke run -- 12 tracks put the norm 3x lower than the full corpus
    # does. ``None`` disables clipping while still measuring.
    max_grad_norm: float | None = 1000.0

    # Micro-batches accumulated into one optimiser step. Exists for a *correctness*
    # reason, not a memory one -- the usual motive for accumulation is fitting a large
    # effective batch on a small card, and that is not what this is for.
    #
    # A recurrent objective carries state between micro-batches (the semi-CRF's alpha
    # window). That state is computed by whatever the weights were at the time, so when
    # the weights move every step the state a chunk consumes is stale -- the same
    # representational drift R2D2 describes for stored recurrent states. Here it is
    # unusually bad: the carry resets only at a track's first chunk, and a track is a
    # median of 53 chunks, so by a track's end the alpha state is ~53 updates old. And
    # unlike a GRU, alpha is an unnormalised accumulator rather than a contractive
    # state, so it never forgets its way back -- there is no cheap burn-in that heals
    # it, and recomputing it honestly is O(N^2) in chunks per track.
    #
    # Holding the weights still for N micro-batches makes the state *exact* within the
    # window and one update stale across a boundary, at no compute cost at all. The
    # price is N-fold fewer optimiser steps, which is why this is a dial rather than a
    # constant. It also makes the telescoping identity the loss is built on
    # (``SemiCRFLoss``) hold within a window, where today it holds only under frozen
    # weights -- which is why the reported train loss is inflated relative to val.
    #
    # 1 reproduces the un-accumulated loop exactly. Harmless for a stateless objective
    # like frame BCE, where it is just a larger effective batch.
    accumulate_steps: int = 1

    # DataLoader worker subprocesses. 0 loads batches on the main thread, which
    # can starve a fast model's GPU step; >0 prefetches batches across CPU cores
    # so the device stays fed. A few below the physical core count leaves headroom
    # for the main process and the GPU driver.
    num_workers: int = 0

    device: str | None = None  # None -> best available (cuda > mps > cpu)
