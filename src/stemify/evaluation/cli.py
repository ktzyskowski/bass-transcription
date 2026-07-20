"""Command-line entry point for evaluating a baseline on a dataset.

    uv run python -m stemify.evaluation.cli --baseline pyin --dataset idmt
    uv run python -m stemify.evaluation.cli --baseline pyin --dataset slakh --source mix
    uv run python -m stemify.evaluation.cli --baseline pesto --dataset mdb --json out.json

This module is only argument parsing. The work is:

    Evaluation(dataset, baseline).run() -> Report

Adding a dataset means writing a ``Dataset`` subclass and adding one line to
``DATASETS`` -- the runner, the metrics and the report need no changes, because
a dataset owns how its own references are built.

Replaces the old evaluation/pitch.py, whose note-level accuracy on isolated
notes was RPA/RCA at a different granularity and whose per-string and
per-plucking tables are now ``Report`` breakdowns. Baselines that only answer
with a single pitch per clip (basic-pitch) still work: see ``Evaluation._estimate``.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from stemify.baselines.base import PitchBaseline
from stemify.baselines.basic_pitch import BasicPitchBaseline
from stemify.baselines.pesto import PestoBaseline
from stemify.baselines.pyin import PyinBaseline
from stemify.data.base import Dataset
from stemify.data.idmt_smt_bass import IdmtDataset
from stemify.data.mdb_bass_synth import MdbDataset
from stemify.data.slakh import SlakhDataset
from stemify.evaluation import Evaluation
from stemify.transforms import AudioTransform, DemucsSeparator, Identity

BASELINES: dict[str, type[PitchBaseline]] = {
    "pyin": PyinBaseline,
    "pesto": PestoBaseline,
    "basic-pitch": BasicPitchBaseline,
}

# Each entry is (default root, factory). Adding a dataset is one line here plus
# the Dataset subclass itself -- the runner, metrics and report are untouched,
# because a dataset owns how its own references are built. Factories take the
# parsed args so a dataset can read whichever flags it cares about and ignore
# the rest.
DATASETS: dict[str, tuple[str, Callable[[str, argparse.Namespace], Dataset]]] = {
    "idmt": (
        "data/raw/idmt_smt_bass",
        lambda root, a: IdmtDataset(root, limit=a.limit),
    ),
    "mdb": (
        "data/raw/mdb_bass_synth",
        lambda root, a: MdbDataset(root, source=a.source, limit=a.limit),
    ),
    "slakh": (
        "data/raw/slakh2100",
        lambda root, a: SlakhDataset(
            root, split=a.split, source=a.source, limit=a.limit
        ),
    ),
}


def build_dataset(args: argparse.Namespace) -> Dataset:
    default_root, factory = DATASETS[args.dataset]
    return factory(args.root or default_root, args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", choices=BASELINES, required=True)
    ap.add_argument("--dataset", choices=DATASETS, default="mdb")
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
        help="evaluate only the first N items, in sorted order -- a smoke test, "
        "NOT a sample. On IDMT the sort groups by expression, so a small limit "
        "lands entirely inside one technique and the numbers do not generalise.",
    )
    ap.add_argument(
        "--group-by",
        nargs="*",
        help="breakdown keys (default: whatever the dataset's items declare)",
    )
    ap.add_argument(
        "--separate",
        choices=["none", "demucs"],
        default="none",
        help="preprocess audio through a source separator before the baseline "
        "hears it. 'demucs' is the deployment condition; 'none' is the clean "
        "upper bound. Results are cached, so repeat runs are free.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="process-pool width for scoring. Near-linear speedup on CPU-bound "
        "conditions (pyin). Keep at 1 for --separate demucs on a cold cache.",
    )
    ap.add_argument("--json", type=Path, help="also write the report here")
    args = ap.parse_args()

    # Both failure modes here are the user's situation, not a bug: a dataset
    # that was never downloaded, or filters that excluded everything. A
    # traceback would bury the one sentence that says which.
    transform: AudioTransform = (
        DemucsSeparator() if args.separate == "demucs" else Identity()
    )
    try:
        evaluation = Evaluation(
            dataset=build_dataset(args),
            baseline=BASELINES[args.baseline](),
            transform=transform,
            group_by=args.group_by,
            workers=args.workers,
        )
        report = evaluation.run()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(report.render())
    if args.json:
        report.write_json(args.json)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
