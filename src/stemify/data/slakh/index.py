"""Walking a Slakh tree: which tracks exist, which stems are bass, where the root is.

Split out of slakh.py. This is *corpus layout* knowledge -- the archive's extra
directory level, the metadata.yaml stem table, the omitted split -- which
changes when the dataset's on-disk shape does, not when an item or a filter changes.
"""

from pathlib import Path
from typing import Any

import yaml

from stemify.data.slakh.item import BassStem

# Slakh's instrument taxonomy labels bass stems with this inst_class.
BASS_CLASS = "Bass"

SPLITS = ("train", "validation", "test")


# The tarball unpacks into a directory of its own, so extracting it into
# data/raw/slakh2100 yields data/raw/slakh2100/slakh2100_flac_redux/<split>/...
# Everyone gets this wrong once; descend automatically rather than making the
# configured root depend on how the archive happened to be unpacked.
ARCHIVE_DIR = "slakh2100_flac_redux"


def resolve_root(root: str | Path) -> Path:
    """The directory that actually holds the split folders."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} does not exist. Slakh2100 is a single 104 GB tarball from "
            "https://zenodo.org/records/4599666 -- see docs/datasets.md."
        )
    nested = root / ARCHIVE_DIR
    if nested.is_dir() and not any((root / s).is_dir() for s in SPLITS):
        return nested
    return root


def read_metadata(track_dir: Path) -> dict[str, Any]:
    """A track's ``metadata.yaml`` stem table, or ``{}`` if unreadable.

    Returning empty on failure is deliberate: callers treat "no metadata" as
    "this track has no identifiable bass", which is the safe reading for both
    indexing (skip it) and pruning (delete nothing).
    """
    meta_path = track_dir / "metadata.yaml"
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open() as fh:
            meta = yaml.safe_load(fh) or {}
    except yaml.YAMLError:
        return {}
    return meta.get("stems") or {}


def bass_stem_ids(track_dir: Path, require_rendered: bool = True) -> set[str]:
    """Stem ids in ``track_dir`` that hold bass.

    The single definition of "which stems are bass", used both by ``index`` and
    by the pruning tool. These previously disagreed -- pruning matched on
    ``inst_class`` alone while indexing also required the render flags -- which
    meant a change to one silently failed to reach the other.

    ``require_rendered`` is what separates the two callers. Indexing wants only
    stems with audio on disk. Pruning wants every stem the dataset *calls* bass,
    including ones that were never rendered, so that a metadata quirk can never
    cause real bass audio to be deleted.
    """
    stems = read_metadata(track_dir)
    ids = set()
    for stem_id, info in stems.items():
        if info.get("inst_class") != BASS_CLASS or info.get("is_drum"):
            continue
        # Not every stem in the metadata was actually rendered or saved; the
        # flags are the dataset's own record of that, and trusting them avoids
        # a pile of missing-file errors deep in a training run.
        if require_rendered and (
            info.get("audio_rendered") is False or info.get("midi_saved") is False
        ):
            continue
        ids.add(stem_id)
    return ids


def _bass_stems(track_dir: Path, split: str) -> list[BassStem]:
    stems = read_metadata(track_dir)
    mix = track_dir / "mix.flac"
    found = []
    for stem_id in sorted(bass_stem_ids(track_dir)):
        stem_path = track_dir / "stems" / f"{stem_id}.flac"
        midi_path = track_dir / "MIDI" / f"{stem_id}.mid"
        # Only the MIDI is universally required -- it is the reference, so
        # without it there is nothing to score against. The stem audio is
        # needed for source="stem" and never read for source="mix", so
        # requiring it here would discard tracks that are perfectly usable for
        # mix evaluation. SlakhDataset applies the source-specific check.
        if not midi_path.exists():
            continue
        program = stems.get(stem_id, {}).get("midi_program_name", "")
        found.append((stem_id, stem_path, midi_path, program))

    return [
        BassStem(
            track=track_dir.name,
            split=split,
            stem_id=stem_id,
            stem_path=stem_path,
            midi_path=midi_path,
            mix_path=mix,
            program_name=program,
            n_bass_stems=len(found),
        )
        for stem_id, stem_path, midi_path, program in found
    ]


def index(
    root: str | Path,
    splits: tuple[str, ...] = SPLITS,
    unambiguous_only: bool = False,
) -> list[BassStem]:
    """Index every rendered bass stem under ``root``.

    A track with two bass stems yields two entries. Pass ``unambiguous_only`` to
    keep only single-bass tracks, which is what mix-based evaluation needs.
    """
    root = resolve_root(root)

    stems: list[BassStem] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for track_dir in sorted(split_dir.iterdir()):
            if track_dir.is_dir():
                stems.extend(_bass_stems(track_dir, split))

    if not stems:
        seen = sorted({p.name for p in root.iterdir()}) if root.is_dir() else []
        raise FileNotFoundError(
            f"No bass stems under {root}. Top-level entries: {seen or 'none'}. "
            f"Expected <root>/<split>/TrackXXXXX/metadata.yaml with splits {SPLITS}."
        )
    if unambiguous_only:
        stems = [s for s in stems if s.unambiguous]
    return stems

