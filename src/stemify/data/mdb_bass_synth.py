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

Layout, verified against an unpacked copy and the archive's README::

    MDB-bass-synth/
        annotation_bass/<artist>_<song>_STEM_<id>.RESYN.csv
        audio_bass/<artist>_<song>_STEM_<id>.RESYN.wav      mono
        audio_mix/<artist>_<song>_MIX_basssynth.wav         stereo

Flat folders by file type, not a directory per track. Note that the three files
for one track do **not** share a common prefix: the mix drops the
``_STEM_<id>.RESYN`` component the other two carry, so they are paired by
stripping it rather than by globbing a shared stem.
"""

import csv
import re
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

# The archive's own layout, per its README -- flat folders by file type rather
# than a directory per track.
ANNOTATION_DIR = "annotation_bass"
BASS_DIR = "audio_bass"
MIX_DIR = "audio_mix"


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

        This replaces the RMS-gated synthetic reference IDMT has to
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


# The annotation and bass filenames carry a stem id; the mix filename does not.
# Stripping it yields the <artist>_<songtitle> key the three files share.
_ANNOTATION = re.compile(r"^(?P<song>.+)_STEM_(?P<stem_id>\d+)\.RESYN$")


def index(root: str | Path) -> list[Track]:
    """Index every annotated track under ``root``.

    Driven by the naming conventions in the archive's own README rather than by
    pattern-matching, since the three files for one track do *not* share a
    common prefix -- the mix drops the ``_STEM_<id>.RESYN`` part that the bass
    and annotation both carry.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} does not exist. MDB-bass-synth is requested via the form at "
            "https://synthdatasets.weebly.com/mdb-bass-synth-download.html -- "
            "see docs/datasets.md"
        )

    annotations = sorted((root / ANNOTATION_DIR).glob("*.csv"))
    if not annotations:
        present = sorted(p.name for p in root.iterdir()) if root.is_dir() else []
        raise FileNotFoundError(
            f"No annotations in {root / ANNOTATION_DIR}. Contents of {root}: "
            f"{present or 'empty'}. Expected the archive's own layout: "
            f"{ANNOTATION_DIR}/, {BASS_DIR}/, {MIX_DIR}/."
        )

    tracks: list[Track] = []
    orphans: list[str] = []
    for annotation in annotations:
        match = _ANNOTATION.match(annotation.stem)
        if match is None:
            orphans.append(annotation.name)
            continue
        song = match["song"]
        stem_path = root / BASS_DIR / f"{annotation.stem}.wav"
        mix_path = root / MIX_DIR / f"{song}_MIX_basssynth.wav"
        if not stem_path.exists():
            orphans.append(annotation.name)
            continue
        tracks.append(
            Track(
                track_name=song,
                stem_path=stem_path,
                annotation_path=annotation,
                mix_path=mix_path if mix_path.exists() else None,
            )
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
