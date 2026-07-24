"""Command-line entry point for evaluating a baseline on a dataset.

    uv run python -m stemify.evaluation.cli --baseline pyin --dataset idmt
    uv run python -m stemify.evaluation.cli --baseline pyin --dataset slakh --source mix
    uv run python -m stemify.evaluation.cli --baseline pyin --dataset slakh --level note

This module is only argument parsing. The work is:

    FrameEvaluation(dataset, baseline).run() -> FrameReport   (--level frame)
    NoteEvaluation(dataset, baseline).run()  -> NoteReport    (--level note)

Adding a dataset means writing a ``Dataset`` subclass and adding one line to
``DATASETS`` -- the runners, the metrics and the reports need no changes, because
a dataset owns how its own references are built.
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
from stemify.decoders.heuristic import HeuristicDecoder
from stemify.decoders.semicrf import SemiCRFDecoder
from stemify.encoders.dsp import DSPEncoder
from stemify.evaluation import FrameEvaluation, NoteEvaluation
from stemify.segmenter import Segmenter
from stemify.transforms import AudioTransform, DemucsSeparator, Identity

BASELINES: dict[str, type[PitchBaseline]] = {
    "pyin": PyinBaseline,
    "pesto": PestoBaseline,
    "basic-pitch": BasicPitchBaseline,
}

# Decoders available to the note harness. The heuristic is the DSP floor; semicrf is
# the learned global decoder (untrained potentials unless a checkpoint carries them).
DECODERS = {"heuristic": HeuristicDecoder, "semicrf": SemiCRFDecoder}


def load_tagger(checkpoint: Path):
    """Rebuild a ``FrameTagger`` from a saved checkpoint (state_dict + head list)."""
    import torch

    from stemify.encoders.heads import onset_head, voicing_head
    from stemify.encoders.networks import TCN
    from stemify.encoders.tagger import FrameTagger

    ckpt = torch.load(checkpoint, map_location="cpu")
    backbone = TCN()
    builders = {"onset": onset_head, "voicing": voicing_head}
    heads = {h: builders[h](backbone.hidden) for h in ckpt["heads"]}
    model = FrameTagger(backbone, heads)
    model.load_state_dict(ckpt["state_dict"])
    return model


def build_note_transcriber(args: argparse.Namespace) -> Segmenter:
    """Compose the ``Segmenter`` the note harness scores: a DSP or learned encoder
    (pyin contour + onset cues, or a trained ``FrameTagger``) paired with a decoder.

    ``--baseline pyin`` is the DSP floor; ``--baseline learned --checkpoint X`` runs a
    trained model. The frame-only baselines (pesto, basic-pitch) are not note-native
    -- they produce a contour, not notes -- so they are rejected here.
    """
    if args.baseline == "pyin":
        encoder = DSPEncoder()
    elif args.baseline == "learned":
        if not args.checkpoint:
            raise SystemExit("--baseline learned needs --checkpoint <model.pt>")
        from stemify.encoders.neural import NeuralEncoder

        encoder = NeuralEncoder(load_tagger(args.checkpoint), device="cpu")
    else:
        raise SystemExit(
            f"note-level eval needs --baseline pyin or learned, not {args.baseline!r}"
        )
    return Segmenter(encoder, DECODERS[args.decoder]())

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
    ap.add_argument(
        "--baseline", choices=[*BASELINES, "learned"], required=True,
        help="frame: pyin/pesto/basic-pitch. note: pyin (DSP floor) or learned "
        "(a trained FrameTagger, with --checkpoint).",
    )
    ap.add_argument(
        "--decoder", choices=DECODERS, default="heuristic",
        help="note only: how cues become notes -- heuristic (greedy) or semicrf.",
    )
    ap.add_argument(
        "--checkpoint", type=Path,
        help="note --baseline learned: the trained model .pt to load.",
    )
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
        "conditions (pyin). Frame-level only. Keep at 1 for --separate demucs "
        "on a cold cache.",
    )
    ap.add_argument(
        "--level",
        choices=["frame", "note"],
        default="frame",
        help="frame: pitch per instant (RPA/RCA/Overall). note: discrete note "
        "events (precision/recall/F1), which needs a note-native dataset "
        "(slakh, idmt). Frame-level separation flags do not apply to note.",
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
        dataset = build_dataset(args)
        if args.level == "note":
            transcriber = build_note_transcriber(args)
            report = NoteEvaluation(dataset, transcriber, transform=transform).run()
        else:
            if args.baseline not in BASELINES:
                raise SystemExit(f"frame-level eval needs a pitch baseline, not {args.baseline!r}")
            report = FrameEvaluation(
                dataset=dataset,
                baseline=BASELINES[args.baseline](),
                transform=transform,
                group_by=args.group_by,
                workers=args.workers,
            ).run()
    except (FileNotFoundError, ValueError, NotImplementedError) as exc:
        raise SystemExit(str(exc)) from exc

    print(report.render())
    if args.json:
        report.write_json(args.json)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
