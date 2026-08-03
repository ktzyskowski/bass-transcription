"""SlakhDataset: the filtered collection of bass stems a run trains or scores on.

Split out of slakh.py. The *filtering policy* -- which split, which source, the
monophonic premise, the limit -- is what changes here, independently of how the tree is
walked or what one stem holds.
"""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from stemify.data.base import Dataset
from stemify.data.slakh.index import SPLITS, index
from stemify.data.slakh.item import BassStem


class SlakhDataset(Dataset):
    """Slakh2100 bass stems, filtered to those the monophonic premise holds for.

    Defaults to the ``test`` split. Slakh is large enough to train on, so
    evaluating on ``train`` reports a number that stops meaning anything the
    moment a model exists.

    Stems whose MIDI is substantially polyphonic are dropped: a doubled-octave
    transcription has no single right answer, and scoring against one would
    manufacture octave errors that are the transcription's fault rather than the
    model's -- especially misleading in a project whose central metric is the
    octave-error rate.

    The polyphony filter reads every candidate's MIDI, so construction is the
    expensive part and iteration is cheap. Pass ``limit`` to cut that cost while
    iterating.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        source: str = "stem",
        limit: int | None = None,
        max_polyphony: float = 0.05,
        unambiguous_only: bool | None = None,
    ) -> None:
        if source not in ("stem", "mix"):
            raise ValueError(f"source must be 'stem' or 'mix', got {source!r}")
        self.split = split
        self.source = source
        # Mixes *require* single-bass tracks -- with two basses sounding, one
        # reference contour cannot describe the mix. Stems do not, so the two
        # sources otherwise yield different track sets (147 vs 124 on test),
        # which would silently make a stem-vs-mix comparison uncontrolled.
        # Pass unambiguous_only=True explicitly to hold the set fixed across
        # conditions.
        if unambiguous_only is None:
            unambiguous_only = source == "mix"
        self.unambiguous_only = unambiguous_only
        stems = index(root, splits=(split,), unambiguous_only=unambiguous_only)
        if limit:
            stems = stems[:limit]

        # Stem audio is only required when it is what gets listened to.
        self.skipped_missing_audio = 0
        if source == "stem":
            present = [s for s in stems if s.stem_path.exists()]
            self.skipped_missing_audio = len(stems) - len(present)
            stems = present

        self.skipped_polyphonic = 0
        kept = []
        for stem in stems:
            if stem.polyphony_rate() > max_polyphony:
                self.skipped_polyphonic += 1
                continue
            kept.append(replace(stem, source=source))
        self._stems = kept

    @property
    def name(self) -> str:
        return f"slakh2100 {self.split} ({self.source})"

    def __iter__(self) -> Iterator[BassStem]:
        return iter(self._stems)

    def __len__(self) -> int:
        return len(self._stems)