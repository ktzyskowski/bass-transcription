"""The training *loop's* knobs -- what every strategy needs, and nothing else.

Feature settings are not here: they live in ``stemify.features`` because they are
shared with the pyin baseline and the eval harness, and a model that trained on a
different CQT than it is evaluated on would be silently broken.

Strategy settings are not here either, which is newer. Window length, window hop and
per-head ``pos_weight`` are read by the frame recipe and by nothing else; the semi-CRF
sets its chunk length from ``CHUNK_FRAMES`` and its objective takes no class weights.
While they sat here, ``--pos-weight onset=5`` on a semi-CRF run was accepted and
silently ignored -- the failure mode this project keeps paying for. They now live on
``FrameRecipe``, where passing one to another strategy is an error instead.

Four fields were deleted outright when this was audited, all with zero readers:
``heads`` (duplicated ``Recipe.heads``, the actual source of truth), ``train_sources``
(superseded by ``Recipe.conditions`` -- and its default still named the pre-2026-07-30
condition tags, so it would have raised if anything had read it), ``seed`` (nothing was
ever seeded from it) and ``onset_tolerance_frames`` (``examples.py`` has its own
parameter with the same default, never wired to this one).
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Optimisation.
    batch_size: int = 16
    learning_rate: float = 1e-3
    max_epochs: int = 50
    patience: int = 8  # early-stop after this many epochs without val gain

    # DataLoader worker subprocesses. 0 loads batches on the main thread, which
    # can starve a fast model's GPU step; >0 prefetches batches across CPU cores
    # so the device stays fed. A few below the physical core count leaves headroom
    # for the main process and the GPU driver.
    num_workers: int = 0

    device: str | None = None  # None -> best available (cuda > mps > cpu)
