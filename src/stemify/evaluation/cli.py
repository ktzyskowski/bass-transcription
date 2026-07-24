"""Command-line entry point for evaluating a model on a dataset.

    uv run python -m stemify.evaluation.cli --encoder pyin --dataset slakh --level cues
    uv run python -m stemify.evaluation.cli --encoder pyin --dataset slakh --level note
    uv run python -m stemify.evaluation.cli --encoder learned --checkpoint m.pt \
        --dataset slakh --level note --decoder semicrf

This module is only argument parsing. The work is:

    CuesEvaluation(dataset, encoder).run() -> CuesReport    (--level cues)
    NoteEvaluation(dataset, Segmenter(encoder, decoder)).run() -> NoteReport  (--level note)

Both levels score the same encoder -- ``pyin`` is the DSP floor (``DSPEncoder``),
``learned`` a trained ``FrameTagger`` (``NeuralEncoder``, needs ``--checkpoint``).
Cues-level scores that encoder's per-frame evidence; note-level pairs it with a
decoder and scores the notes. Adding a dataset means writing a ``Dataset`` subclass
and one line in ``DATASETS``; the runners, metrics and reports need no changes.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from stemify.data.base import Dataset
from stemify.data.idmt_smt_bass import IdmtDataset
from stemify.data.mdb_bass_synth import MdbDataset
from stemify.data.slakh import SlakhDataset
from stemify.decoders.heuristic import HeuristicDecoder
from stemify.decoders.semicrf import SemiCRFDecoder
from stemify.encoders.base import Encoder
from stemify.encoders.dsp import DSPEncoder
from stemify.evaluation import CuesEvaluation, NoteEvaluation
from stemify.segmenter import Segmenter
from stemify.transforms import AudioTransform, DemucsSeparator, Identity

ENCODERS = ("pyin", "learned")

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


def build_encoder(args: argparse.Namespace) -> Encoder:
    """The encoder both levels score: the pyin DSP floor, or a trained model.

    ``--encoder pyin`` is ``DSPEncoder`` (spectral-flux onset + pyin pitch), the floor.
    ``--encoder learned --checkpoint X`` loads a trained ``FrameTagger`` into a
    ``NeuralEncoder``. The learned encoder is CPU-pinned here -- eval is not the hot path.
    """
    if args.encoder == "pyin":
        return DSPEncoder()
    if not args.checkpoint:
        raise SystemExit("--encoder learned needs --checkpoint <model.pt>")
    from stemify.encoders.neural import NeuralEncoder

    return NeuralEncoder(load_tagger(args.checkpoint), device="cpu")


# Each entry is (default root, factory). Adding a dataset is one line here plus the
# Dataset subclass itself -- the runner, metrics and report are untouched, because a
# dataset owns how its own references are built. Factories take the parsed args so a
# dataset can read whichever flags it cares about and ignore the rest.
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
        "--encoder", choices=ENCODERS, required=True,
        help="pyin (DSP floor) or learned (a trained FrameTagger, with --checkpoint).",
    )
    ap.add_argument(
        "--decoder", choices=DECODERS, default="heuristic",
        help="note only: how cues become notes -- heuristic (greedy) or semicrf.",
    )
    ap.add_argument(
        "--checkpoint", type=Path,
        help="--encoder learned: the trained model .pt to load.",
    )
    ap.add_argument("--dataset", choices=DATASETS, default="slakh")
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
        "--separate",
        choices=["none", "demucs"],
        default="none",
        help="preprocess audio through a source separator before the encoder "
        "hears it. 'demucs' is the deployment condition; 'none' is the clean "
        "upper bound. Results are cached, so repeat runs are free.",
    )
    ap.add_argument(
        "--level",
        choices=["cues", "note"],
        default="note",
        help="cues: the encoder's per-frame evidence (onset/voicing/pitch quality). "
        "note: discrete note events (precision/recall/F1), which needs a note-native "
        "dataset (slakh, idmt).",
    )
    ap.add_argument("--json", type=Path, help="also write the report here")
    args = ap.parse_args()

    # Both failure modes here are the user's situation, not a bug: a dataset that was
    # never downloaded, or filters that excluded everything. A traceback would bury
    # the one sentence that says which.
    transform: AudioTransform = (
        DemucsSeparator() if args.separate == "demucs" else Identity()
    )
    try:
        dataset = build_dataset(args)
        encoder = build_encoder(args)
        if args.level == "note":
            transcriber = Segmenter(encoder, DECODERS[args.decoder]())
            report = NoteEvaluation(dataset, transcriber, transform=transform).run()
        else:
            report = CuesEvaluation(dataset, encoder, transform=transform).run()
    except (FileNotFoundError, ValueError, NotImplementedError) as exc:
        raise SystemExit(str(exc)) from exc

    print(report.render())
    if args.json:
        report.write_json(args.json)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
