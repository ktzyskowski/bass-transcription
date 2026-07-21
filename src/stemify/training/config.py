"""One place for the training knobs, so a run is reproducible from a dataclass.

Feature settings are *not* here -- they live in ``stemify.features`` because they
are shared with the pyin baseline and the eval harness, and a model that trained
on a different CQT than it is evaluated on would be silently broken. This holds
only the choices training owns: windowing, optimisation, and which heads to fit.
"""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # Which heads to train. One entry is the onset-first plan; more is joint
    # multi-task over the shared encoder.
    heads: tuple[str, ...] = ("onset",)

    # A window is one training example: enough frames for the GRU to have
    # context, short enough to batch. 256 frames ~= 3 s at the 11.6 ms hop.
    window_frames: int = 256
    hop_frames: int = 128  # window stride; 50% overlap so onsets near an edge
    #                        of one window sit mid-window in the next.

    # Onset targets are widened by this many frames each side (see labels.py).
    onset_tolerance_frames: int = 1

    # Optimisation.
    batch_size: int = 16
    learning_rate: float = 1e-3
    max_epochs: int = 50
    patience: int = 8  # early-stop after this many epochs without val gain

    # Onsets and (on sparse basslines) voicing are rare positives, so their BCE
    # is class-weighted. Per-head positive weight; None lets the trainer estimate
    # it from the data.
    pos_weight: dict[str, float | None] = field(default_factory=dict)

    # Input domain randomisation: train on these source conditions with labels
    # always from the clean stem, so the head is robust to the deployment
    # (separated) condition. See the training-consistency discussion.
    train_sources: tuple[str, ...] = ("stem", "mix", "demucs")

    device: str | None = None  # None -> best available (cuda > mps > cpu)
    seed: int = 0