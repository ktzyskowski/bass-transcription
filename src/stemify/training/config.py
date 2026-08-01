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

    # DataLoader worker subprocesses. 0 loads batches on the main thread, which
    # can starve a fast model's GPU step; >0 prefetches batches across CPU cores
    # so the device stays fed. A few below the physical core count leaves headroom
    # for the main process and the GPU driver.
    num_workers: int = 0

    device: str | None = None  # None -> best available (cuda > mps > cpu)
