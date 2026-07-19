"""Run a baseline over a dataset and score it.

    uv run python -m stemify.evaluation.harness --baseline pyin --dataset mdb
    uv run python -m stemify.evaluation.harness --baseline pesto --dataset mdb --source mix
    uv run python -m stemify.evaluation.harness --baseline pyin --dataset idmt --limit 200

The pipeline is fixed and each stage is somebody else's job:

    dataset -> reference contour     (this module's adapters)
    baseline -> estimate contour     (stemify.baselines)
    align onto the reference grid    (stemify.evaluation.alignment)
    pool frames across tracks        (FrameSet)
    score                            (stemify.evaluation.metrics)

Pooling before scoring is what makes aggregation frame-weighted for free: a
six-minute track contributes more frames than a thirty-second one, and a metric
called on a masked subset gives a per-track breakdown with no separate
aggregation path to keep consistent.

Datasets differ only in how a reference contour is obtained, which is the entire
content of an adapter below. MDB-bass-synth ships annotated contours; IDMT-SMT-
Bass does not, so its reference is synthesised from the filename's pitch label
gated by signal energy -- a much weaker reference, since voicing is inferred by
threshold rather than annotated.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from stemify.baselines.base import PitchBaseline
from stemify.baselines.pesto import PestoBaseline
from stemify.baselines.pyin import PyinBaseline
from stemify.data import mdb_bass_synth, slakh
from stemify.data.idmt_smt_bass import index as idmt_index
from stemify.evaluation.alignment import FrameSet, align
from stemify.evaluation.metrics import Metric, PitchAccuracy, PitchConfusion

BASELINES: dict[str, type[PitchBaseline]] = {
    "pyin": PyinBaseline,
    "pesto": PestoBaseline,
}

METRICS: list[Metric] = [PitchAccuracy(), PitchConfusion()]

IDMT_REF_HOP_S = 0.01
IDMT_REF_FLOOR_DB = -40.0  # frames this far below the clip peak count as silent


@dataclass(frozen=True)
class EvalItem:
    """One unit of evaluation: audio plus the reference contour to score against."""

    name: str
    audio: Path
    ref_times: np.ndarray
    ref_freqs: np.ndarray  # Hz, 0.0 where unvoiced


def mdb_items(root: str, source: str, limit: int | None) -> list[EvalItem]:
    """MDB-bass-synth: annotated contours, so the reference is read not inferred."""
    tracks = mdb_bass_synth.index(root)
    if limit:
        tracks = tracks[:limit]
    items = []
    for track in tracks:
        times, freqs = track.reference()
        items.append(EvalItem(track.name, track.audio(source), times, freqs))
    return items


def slakh_items(
    root: str,
    source: str,
    limit: int | None,
    splits: tuple[str, ...] = ("test",),
    max_polyphony: float = 0.05,
) -> list[EvalItem]:
    """Slakh2100: the reference is derived from the MIDI that rendered the audio.

    Defaults to the ``test`` split, because the whole point of Slakh is that it
    is large enough to train on -- evaluating on ``train`` would quietly report
    a number that means nothing once a model exists.

    Bass stems whose MIDI is substantially polyphonic are dropped: this project
    assumes one bass note at a time, and a doubled-octave transcription has no
    single right answer to score against. ``max_polyphony`` is the tolerance for
    incidental note overlap at releases.
    """
    stems = slakh.index(root, splits=splits, unambiguous_only=(source == "mix"))
    if limit:
        stems = stems[:limit]
    items, skipped = [], 0
    for stem in stems:
        if stem.polyphony_rate() > max_polyphony:
            skipped += 1
            continue
        times, freqs = stem.reference()
        if freqs.size == 0 or not (freqs > 0).any():
            continue
        items.append(EvalItem(stem.name, stem.audio(source), times, freqs))
    if skipped:
        print(f"skipped {skipped} bass stems above {max_polyphony:.0%} polyphony",
              file=sys.stderr)
    return items


def idmt_items(root: str, limit: int | None) -> list[EvalItem]:
    """IDMT-SMT-Bass: one sustained note per file, so the reference is that
    note's labelled pitch wherever the clip has energy.

    The RMS gate is the weak link -- it decides voicing by threshold, so a note's
    decay tail is cut off at an arbitrary level. Fine for isolated notes where
    the answer is constant throughout; it is exactly what MDB-bass-synth's
    annotated voicing replaces.
    """
    notes = [n for n in idmt_index(root) if n.midi_pitch is not None]
    if limit:
        notes = notes[:limit]
    items = []
    for note in notes:
        audio, sr = librosa.load(note.path, sr=None, mono=True)
        hop = max(1, round(sr * IDMT_REF_HOP_S))
        rms = librosa.feature.rms(y=audio, frame_length=hop * 4, hop_length=hop)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=hop)
        voiced = librosa.amplitude_to_db(rms, ref=np.max) > IDMT_REF_FLOOR_DB
        freqs = np.where(voiced, librosa.midi_to_hz(note.midi_pitch), 0.0)
        items.append(EvalItem(str(note.path.name), note.path, times, freqs))
    return items


def collect(baseline: PitchBaseline, items: list[EvalItem]) -> FrameSet:
    """Predict and align every item, pooling the result into one FrameSet."""
    parts: list[tuple[str, np.ndarray, np.ndarray]] = []
    for i, item in enumerate(items, 1):
        est = baseline.predict_f0(item.audio)
        true, pred = align(item.ref_times, item.ref_freqs, est)
        parts.append((item.name, true, pred))
        print(f"  {i}/{len(items)} {item.name[:40]:<40}", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    return FrameSet.from_tracks(parts)


def report(baseline_name: str, dataset_name: str, frames: FrameSet) -> None:
    # Refuse to print a table for an empty run. Every rate would come out 0.0%,
    # which reads as "the baseline scored zero" rather than "nothing was
    # evaluated" -- the two are indistinguishable in the output and only one of
    # them is a result.
    if not frames.track_names:
        raise SystemExit(
            f"No items to evaluate for {dataset_name}. Check the dataset root, "
            "and whether filters (polyphony, split, limit) excluded everything."
        )
    if not frames.voiced.any():
        raise SystemExit(
            f"{dataset_name}: {len(frames.track_names)} items but no "
            "reference-voiced frames. The reference contours are all silent."
        )

    print(f"\n=== {baseline_name} on {dataset_name} "
          f"({len(frames.track_names)} items, {len(frames):,} frames) ===")
    for metric in METRICS:
        result = metric.calculate(frames.true, frames.pred)
        print(f"\n{result.name}")
        print(result.render())

    # Per-item RPA, worst first. With the errors pooled it is easy to miss that
    # a good average hides a handful of tracks the baseline fails outright --
    # which is a different problem from being uniformly mediocre.
    accuracy = PitchAccuracy()
    scored = []
    for i, name in enumerate(frames.track_names):
        subset = frames.where(frames.track == i)
        if not subset.voiced.any():
            continue
        r = accuracy.calculate(subset.true, subset.pred)
        scored.append((r.scalars["rpa"], r.scalars["octave_gap"], name))
    if len(scored) > 1:
        scored.sort()
        print(f"\nworst items by RPA")
        print(f"  {'RPA':>7} {'oct-gap':>8}  item")
        for rpa, gap, name in scored[:10]:
            print(f"  {rpa:>7.1%} {gap:>8.1%}  {name[:50]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", choices=BASELINES, required=True)
    ap.add_argument("--dataset", choices=["mdb", "slakh", "idmt"], default="mdb")
    ap.add_argument("--root", help="dataset root (defaults per dataset)")
    ap.add_argument(
        "--source",
        choices=["stem", "mix"],
        default="stem",
        help="mdb/slakh only: isolated bass stem, or the full mix it sits in",
    )
    ap.add_argument(
        "--split",
        default="test",
        help="slakh only: train/validation/test. Defaults to test -- slakh is a "
        "training set, so evaluating on train reports a meaningless number.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        help="score only the first N items, in sorted order -- a smoke test, NOT a "
        "sample. On IDMT the sort groups by expression, so a small limit lands "
        "entirely inside one technique and the numbers do not generalise.",
    )
    args = ap.parse_args()

    if args.dataset == "mdb":
        root = args.root or "data/raw/mdb_bass_synth"
        items = mdb_items(root, args.source, args.limit)
        label = f"mdb-bass-synth ({args.source})"
    elif args.dataset == "slakh":
        root = args.root or "data/raw/slakh2100"
        items = slakh_items(root, args.source, args.limit, splits=(args.split,))
        label = f"slakh2100 {args.split} ({args.source})"
    else:
        root = args.root or "data/raw/idmt_smt_bass"
        items = idmt_items(root, args.limit)
        label = "idmt-smt-bass"

    baseline = BASELINES[args.baseline]()
    report(baseline.name, label, collect(baseline, items))


if __name__ == "__main__":
    main()
