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
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

# Annotation hop, per the dataset documentation. Not used for parsing -- the
# CSV carries explicit timestamps -- but recorded because it sets the reference
# grid every baseline gets resampled onto, and it is far finer than any
# baseline's native rate (pyin ~23 ms, PESTO 10 ms).
ANNOTATION_HOP_S = 128 / 44100


@dataclass(frozen=True)
class Track:
    """One song: a bass stem, the full mix it came from, and their shared labels."""

    name: str
    stem_path: Path
    annotation_path: Path
    mix_path: Path | None = None

    def audio(self, source: str = "stem") -> Path:
        """Path to evaluate against. ``stem`` isolates pitch tracking; ``mix``
        additionally tests whether the bass is findable inside an arrangement."""
        if source == "stem":
            return self.stem_path
        if source == "mix":
            if self.mix_path is None:
                raise FileNotFoundError(f"{self.name}: no mix file found")
            return self.mix_path
        raise ValueError(f"source must be 'stem' or 'mix', got {source!r}")

    def reference(self) -> tuple[np.ndarray, np.ndarray]:
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
        return np.asarray(times), np.asarray(freqs)

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
            Track(name=name, stem_path=stem, annotation_path=annotation, mix_path=mix)
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
