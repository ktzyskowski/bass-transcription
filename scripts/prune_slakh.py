"""Drop the Slakh stems this project will never read, before paying to store them.

    uv run python scripts/prune_slakh.py /data/slakh2100 --dry-run
    uv run python scripts/prune_slakh.py /data/slakh2100

A Slakh track carries ~10 rendered stems. The bass pipeline reads exactly one of
them plus the mix, so uploading the rest to S3 is paying rent on audio nothing
opens. Pruning takes the corpus from ~100 GB to roughly a third of that.

Kept: ``mix.flac``, every bass ``stems/SNN.flac``, all of ``MIDI/``, and
``metadata.yaml``. MIDI is kept whole because it is negligible in size and
describes instruments the bass model may later be conditioned on.

This is destructive and one-way -- re-obtaining a pruned stem means
re-downloading the 104 GB tarball. It defaults to a dry run for that reason, and
refuses to touch a tree it cannot parse as Slakh.

Lives in ``scripts/`` rather than in ``stemify.data``: it is corpus preparation, run
once per copy of the dataset, and nothing imports it -- it is a leaf *consumer* of
``stemify.data.slakh``, not part of what the package does. The two CLIs that stayed in
the package (``stemify-train``, ``stemify-eval``) did so because they carry library
code other modules import; this carries none.
"""

import argparse
import sys
from pathlib import Path

from stemify.data.slakh import bass_stem_ids, read_metadata


def prunable(track_dir: Path) -> list[Path]:
    """Stem audio files in ``track_dir`` that are not bass.

    Returns an empty list -- rather than every stem -- when the metadata is
    missing or unreadable, so a malformed track is skipped instead of stripped.

    ``require_rendered=False`` deliberately keeps every stem the dataset calls
    bass, including any whose render flags say it was never produced. Indexing
    is stricter, but the asymmetry is the safe direction: the worst case here is
    keeping a file nothing reads, where the worst case the other way is deleting
    bass audio that a metadata quirk mislabelled.
    """
    if not read_metadata(track_dir):
        return []
    keep = bass_stem_ids(track_dir, require_rendered=False)
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
