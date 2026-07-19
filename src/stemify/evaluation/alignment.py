"""Put a reference and an estimate on the same time grid, then pool across items.

Every baseline emits its own grid: pyin hops 512 samples at 22.05 kHz (~23 ms),
PESTO defaults to 10 ms, MDB-bass-synth annotates every 128 samples at 44.1 kHz
(~2.9 ms). Nothing can be compared until they agree.

The reference grid wins. That is the mir_eval convention and it is the right one
-- the reference decides where the question "what pitch is sounding?" is asked,
and the estimate answers wherever it is asked. Resampling the reference down to
a coarse estimate grid would instead let a baseline improve its own score by
predicting less often.

Resampling is voicing-aware (``mir_eval.melody.resample_melody_series``):
interpolating Hz straight through an unvoiced gap would invent a glide between
two unrelated notes, so voicing is carried separately and reapplied after.
"""

from dataclasses import dataclass, field

import mir_eval
import numpy as np

from stemify.contour import Contour


def align(reference: Contour, estimate: Contour) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``estimate`` onto the reference grid.

    Returns ``(true, pred)``: two equal-length Hz arrays with 0.0 for unvoiced,
    which is exactly the contract ``Metric.calculate`` expects.
    """
    if len(estimate) == 0:
        return reference.freqs, np.zeros_like(reference.freqs)

    est_freqs, est_voicing = mir_eval.melody.resample_melody_series(
        times=estimate.times,
        frequencies=np.abs(estimate.freqs),
        voicing=estimate.voiced.astype(float),
        times_new=reference.times,
    )
    # resample_melody_series returns a soft voicing weight; anything that is not
    # confidently voiced after interpolation is treated as a gap.
    pred = np.where(est_voicing > 0.5, est_freqs, 0.0)
    return reference.freqs, pred


@dataclass(frozen=True)
class ItemRef:
    """Just enough about an item to slice its frames back out of the pool."""

    name: str
    groups: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameSet:
    """Frames pooled across many items, with enough labelling to slice them.

    Pooling is what makes aggregation correct for free: a metric called on the
    whole set is frame-weighted, so a six-minute track counts for more than a
    thirty-second one. Averaging per-item scores instead would silently weight
    every item equally, which is the classic way to get an RPA that no
    individual item achieves.

    ``item`` holds an index into ``items`` for each frame, so any breakdown --
    per item, per string, per plucking style -- is a boolean mask away and needs
    no separate aggregation path.
    """

    true: np.ndarray
    pred: np.ndarray
    item: np.ndarray  # int index into items, one per frame
    items: list[ItemRef]

    def __len__(self) -> int:
        return int(self.true.size)

    @classmethod
    def from_items(
        cls, parts: list[tuple[ItemRef, np.ndarray, np.ndarray]]
    ) -> "FrameSet":
        """Build from ``(item, true, pred)`` triples, one per evaluated item."""
        if not parts:
            empty = np.array([], dtype=float)
            return cls(empty, empty, np.array([], dtype=int), [])
        return cls(
            true=np.concatenate([t for _, t, _ in parts]),
            pred=np.concatenate([p for _, _, p in parts]),
            item=np.concatenate(
                [np.full(t.size, i, dtype=int) for i, (_, t, _) in enumerate(parts)]
            ),
            items=[ref for ref, _, _ in parts],
        )

    def where(self, mask: np.ndarray) -> "FrameSet":
        """A view holding only the frames selected by ``mask``."""
        return FrameSet(
            true=self.true[mask],
            pred=self.pred[mask],
            item=self.item[mask],
            items=self.items,
        )

    def for_item(self, name: str) -> "FrameSet":
        return self.where(self.item == [i.name for i in self.items].index(name))

    def group_keys(self) -> list[str]:
        """Every grouping key any item declares, e.g. ``["string", "plucking"]``."""
        seen: dict[str, None] = {}
        for ref in self.items:
            seen.update(dict.fromkeys(ref.groups))
        return list(seen)

    def by(self, key: str) -> dict[str, "FrameSet"]:
        """Split into one FrameSet per distinct value of ``key``.

        Items that do not declare the key are omitted rather than bucketed under
        a placeholder -- a dataset where only some items carry a label should
        report on the ones that do, not invent a category for the rest.
        """
        labels = np.array(
            [ref.groups.get(key, "") for ref in self.items], dtype=object
        )
        per_frame = labels[self.item]
        return {
            str(value): self.where(per_frame == value)
            for value in sorted({v for v in labels if v})
        }

    @property
    def voiced(self) -> np.ndarray:
        """Mask of reference-voiced frames -- the ones that carry a pitch question."""
        return self.true > 0
