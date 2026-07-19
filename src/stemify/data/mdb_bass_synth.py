"""Loader for MDB-bass-synth: 71 real songs with perfect bass f0 annotations.

This is the bridge off isolated notes. IDMT-SMT-Bass answers "can we name a
pitch given one clean sustained note"; this answers "can we track a pitch
through a real bassline in a real recording", which is the actual task -- and it
does so without yet requiring note segmentation, because the annotations are
frame-level contours rather than note events.

How the labels can be perfect: the bass stem was analysed, then **resynthesized**
from that analysis and mixed back in. The annotation is the synthesis control
signal, so it describes the audio exactly rather than approximating it. That
also means the bass you hear is synthetic even though the surrounding song is a
real recording -- worth remembering before reading too much into absolute
numbers.

Each track ships three files:

  - the full mix (stereo), with the resynthesized bass in place of the original
  - the isolated bass stem (mono)
  - a CSV annotation, ``timestamp,hz`` at a 128/44100 s hop (~2.9 ms), first
    frame zero-centred, **0 Hz meaning silence** -- already the mir_eval
    convention this codebase uses throughout

Evaluating on the stem measures pitch tracking; evaluating on the mix measures
pitch tracking *plus* whether the bass survives a full arrangement. Same labels
for both, which makes that a clean controlled comparison -- see ``Track.audio``.

Download: https://synthdatasets.weebly.com/mdb-bass-synth.html (behind an email
form, so there is no scriptable fetch). License is **CC BY-NC 4.0** -- non-
commercial, but unlike IDMT-SMT-Bass there is no ND clause, so trained weights
are not themselves a licensing problem. See docs/datasets.md.

Layout note: the directory names below are inferred from the dataset
description, not from an unpacked copy. ``index()`` discovers files by pattern
and raises with a listing of what it actually found, so a mismatch surfaces
immediately rather than silently yielding zero tracks.
"""

import csv
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import librosa
import numpy as np

from stemify.contour import Contour
from stemify.data.base import Dataset, EvalItem

# Annotation hop, per the dataset documentation. Not used for parsing -- the
# CSV carries explicit timestamps -- but recorded because it sets the reference
# grid every baseline gets resampled onto, and it is far finer than any
# baseline's native rate (pyin ~23 ms, PESTO 10 ms).
ANNOTATION_HOP_S = 128 / 44100


@dataclass(frozen=True)
class Track(EvalItem):
    """One song: a bass stem, the full mix it came from, and their shared labels.

    ``source`` is fixed per item rather than passed to ``audio()``, so a dataset
    configured for mixes cannot accidentally be scored against stems partway
    through a run.
    """

    track_name: str
    stem_path: Path
    annotation_path: Path
    mix_path: Path | None = None
    source: str = "stem"

    @property
    def name(self) -> str:
        return self.track_name

    @property
    def groups(self) -> dict[str, str]:
        return {"source": self.source}

    def audio(self) -> Path:
        """``stem`` isolates pitch tracking; ``mix`` additionally tests whether
        the bass is findable inside an arrangement."""
        if self.source == "stem":
            return self.stem_path
        if self.mix_path is None:
            raise FileNotFoundError(f"{self.name}: no mix file found")
        return self.mix_path

    def reference(self) -> Contour:
        """Load the annotation as ``(times, freqs)``, 0 Hz where silent.

        This replaces the RMS-gated synthetic reference the IDMT harness had to
        construct: here voicing is annotated rather than inferred, so silence
        between notes is ground truth instead of a threshold guess.
        """
        times: list[float] = []
        freqs: list[float] = []
        with self.annotation_path.open(newline="") as fh:
            for row in csv.reader(fh):
                if not row or row[0].lstrip().startswith("#"):
                    continue
                try:
                    t, f = float(row[0]), float(row[1])
                except (ValueError, IndexError):
                    continue  # tolerate a header line
                times.append(t)
                freqs.append(f)
        return Contour(np.asarray(times), np.asarray(freqs))

    def duration(self) -> float:
        return librosa.get_duration(path=str(self.stem_path))


def _find_audio(root: Path, name: str) -> tuple[Path | None, Path | None]:
    """Locate the stem and mix wavs belonging to ``name``.

    Matches on the annotation's basename rather than a fixed directory layout,
    since the published archive's folder names are unverified.
    """
    candidates = [p for p in root.rglob(f"{name}*.wav")]
    mix = next((p for p in candidates if "MIX" in p.stem.upper()), None)
    stem = next((p for p in candidates if "MIX" not in p.stem.upper()), None)
    return stem, mix


def index(root: str | Path) -> list[Track]:
    """Index every annotated track under ``root``."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} does not exist. MDB-bass-synth is downloaded manually from "
            "https://synthdatasets.weebly.com/mdb-bass-synth.html -- see docs/datasets.md"
        )

    tracks: list[Track] = []
    orphans: list[str] = []
    for annotation in sorted(root.rglob("*.csv")):
        name = annotation.stem
        stem, mix = _find_audio(root, name)
        if stem is None:
            orphans.append(name)
            continue
        tracks.append(
            Track(
                track_name=name,
                stem_path=stem,
                annotation_path=annotation,
                mix_path=mix,
            )
        )

    if not tracks:
        found = sorted({p.suffix for p in root.rglob("*") if p.is_file()})
        raise FileNotFoundError(
            f"No annotated tracks under {root}. File types present: {found or 'none'}. "
            "The loader pairs each *.csv with a *.wav sharing its basename; if the "
            "published archive names them differently, fix _find_audio()."
        )
    if orphans:
        print(f"warning: {len(orphans)} annotations without audio, e.g. {orphans[:3]}")
    return tracks


class MdbDataset(Dataset):
    """MDB-bass-synth, evaluated on either the isolated stem or the full mix.

    The same annotations describe both, which is what makes stem-vs-mix a
    controlled comparison rather than two unrelated numbers.
    """

    def __init__(
        self,
        root: str | Path,
        source: str = "stem",
        limit: int | None = None,
    ) -> None:
        if source not in ("stem", "mix"):
            raise ValueError(f"source must be 'stem' or 'mix', got {source!r}")
        self.source = source
        tracks = index(root)
        if limit:
            tracks = tracks[:limit]
        # Source is bound into each item at construction, so it cannot drift.
        self._tracks = [replace(t, source=source) for t in tracks]

    @property
    def name(self) -> str:
        return f"mdb-bass-synth ({self.source})"

    def __iter__(self) -> Iterator[Track]:
        return iter(self._tracks)

    def __len__(self) -> int:
        return len(self._tracks)
