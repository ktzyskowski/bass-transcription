"""Drop the Slakh stems this project will never read, before paying to store them.

    uv run python -m stemify.data.prune_slakh /data/slakh2100 --dry-run
    uv run python -m stemify.data.prune_slakh /data/slakh2100

A Slakh track carries ~10 rendered stems. The bass pipeline reads exactly one of
them plus the mix, so uploading the rest to S3 is paying rent on audio nothing
opens. Pruning takes the corpus from ~100 GB to roughly a third of that.

Kept: ``mix.flac``, every bass ``stems/SNN.flac``, all of ``MIDI/``, and
``metadata.yaml``. MIDI is kept whole because it is negligible in size and
describes instruments the bass model may later be conditioned on.

This is destructive and one-way -- re-obtaining a pruned stem means
re-downloading the 104 GB tarball. It defaults to a dry run for that reason, and
refuses to touch a tree it cannot parse as Slakh.
"""

import argparse
import sys
from pathlib import Path

import yaml

from stemify.data.slakh import BASS_CLASS


def prunable(track_dir: Path) -> list[Path]:
    """Stem audio files in ``track_dir`` that are not bass.

    Returns an empty list -- rather than every stem -- when the metadata is
    missing or unreadable, so a malformed track is skipped instead of stripped.
    """
    meta_path = track_dir / "metadata.yaml"
    if not meta_path.exists():
        return []
    try:
        with meta_path.open() as fh:
            meta = yaml.safe_load(fh) or {}
    except yaml.YAMLError:
        return []

    stems = meta.get("stems")
    if not stems:
        return []

    keep = {sid for sid, info in stems.items() if info.get("inst_class") == BASS_CLASS}
    return [
        path
        for path in sorted((track_dir / "stems").glob("*.flac"))
        if path.stem not in keep
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="unpacked Slakh root")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be deleted without deleting it",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="delete without the confirmation prompt (for unattended EC2 runs)",
    )
    args = ap.parse_args()

    tracks = sorted(p for p in args.root.rglob("metadata.yaml"))
    if not tracks:
        raise SystemExit(
            f"No metadata.yaml under {args.root} -- this does not look like an "
            "unpacked Slakh tree, refusing to delete anything."
        )

    doomed: list[Path] = []
    kept = 0
    for meta in tracks:
        victims = prunable(meta.parent)
        doomed.extend(victims)
        kept += len(list((meta.parent / "stems").glob("*.flac"))) - len(victims)

    freed = sum(p.stat().st_size for p in doomed if p.exists())
    print(f"tracks           {len(tracks):>8,}")
    print(f"bass stems kept  {kept:>8,}")
    print(f"stems to delete  {len(doomed):>8,}")
    print(f"space freed      {freed / 1e9:>8.1f} GB")

    if args.dry_run:
        print("\ndry run, nothing deleted")
        return
    if not doomed:
        return
    if not args.force:
        # One-way and expensive to undo: a 104 GB re-download.
        reply = input(f"\ndelete {len(doomed):,} files? [y/N] ")
        if reply.strip().lower() != "y":
            print("aborted", file=sys.stderr)
            raise SystemExit(1)

    for path in doomed:
        path.unlink(missing_ok=True)
    print(f"deleted {len(doomed):,} files, freed {freed / 1e9:.1f} GB")


if __name__ == "__main__":
    main()