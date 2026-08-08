"""Command-line entry point for evaluating a model on a dataset.

    uv run stemify-eval --encoder pyin --dataset slakh --level cues
    uv run stemify-eval --encoder pyin --dataset slakh --level note
    uv run stemify-eval --encoder learned --checkpoint m.pt \
        --dataset slakh --level note --decoder semicrf

(``python -m stemify.evaluation.cli`` still works; the console script is declared in
``pyproject.toml`` so the entry point is discoverable without reading this file.)

This module is only argument parsing. The work is:

    CuesEvaluation(dataset, encoder).run() -> CuesReport    (--level cues)
    NoteEvaluation(dataset, Transcriber(encoder, decoder)).run() -> NoteReport  (--level note)

Both levels score the same encoder -- ``pyin`` is the DSP floor (``ClassicalEncoder``),
``learned`` a trained ``CueNet`` (``LearnedEncoder``, needs ``--checkpoint``).
Cues-level scores that encoder's per-frame evidence; note-level pairs it with a
decoder and scores the notes. Adding a dataset means writing a ``Dataset`` subclass
and one line in ``DATASETS``; the runners, metrics and reports need no changes.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from stemify.core.data.base import Dataset
from stemify.core.data.filobass import FiloBassDataset
from stemify.core.data.idmt_smt_bass import IdmtDataset
from stemify.core.data.mdb_bass_synth import MdbDataset
from stemify.core.data.slakh import SlakhDataset
from stemify.core.decoders.greedy import DSP_PITCH_SPLIT_SEMITONES, GreedyDecoder
from stemify.core.decoders.base import Decoder
from stemify.semicrf import SemiCRFDecoder
from stemify.core.encoders.base import Encoder
from stemify.core.encoders.classical import ClassicalEncoder
from stemify.evaluation import CuesEvaluation, NoteEvaluation
from stemify.evaluation.report import Report
from stemify.core.transcriber import Transcriber
from stemify.training.checkpoint import Checkpoint, load_cuenet
from stemify.core.transforms import AudioTransform, DemucsSeparator, Identity

ENCODERS = ("pyin", "learned")

# Decoders available to the note harness. The heuristic is the DSP floor; semicrf is
# the learned global decoder, whose potentials must be restored from the checkpoint --
# see ``build_decoder``.
DECODERS = {"heuristic": GreedyDecoder, "semicrf": SemiCRFDecoder}


def build_encoder(args: argparse.Namespace) -> Encoder:
    """The encoder both levels score: the pyin DSP floor, or a trained model.

    ``--encoder pyin`` is ``ClassicalEncoder`` (spectral-flux onset + pyin pitch), the floor.
    ``--encoder learned --checkpoint X`` loads a trained ``CueNet`` into a
    ``LearnedEncoder``. The learned encoder is CPU-pinned here -- eval is not the hot path.
    """
    if args.encoder == "pyin":
        return ClassicalEncoder()
    if not args.checkpoint:
        raise SystemExit("--encoder learned needs --checkpoint <model.pt>")
    from stemify.models.learned_encoder import LearnedEncoder

    return LearnedEncoder(load_cuenet(args.checkpoint), device="cpu")


def build_decoder(args: argparse.Namespace) -> Decoder:
    """The decoder, with the one setting that depends on which encoder feeds it.

    A pitch change places notes only for an encoder that has no trained onset head.
    ``ClassicalEncoder`` emits a spectral-flux spike train and no voicing curve at all, so
    for it the contour moving really is the only evidence a new note began -- and the
    DSP floor has to stay the floor it always was, or every comparison against it
    silently improves. For a ``LearnedEncoder`` the same rule overrides a ~7 ms-accurate
    onset with a ~58 ms-late pitch transition, which is why it defaults off; see
    ``greedy.PITCH_SPLIT_SEMITONES``.

    The semicrf path does not have that setting, but it has a harder requirement: its
    potentials *are* the model. A default-constructed ``SemiCRFDecoder`` has ``w_on=1``
    and ``dur=zeros`` -- no duration preference at all, which is the mechanism the
    objective exists to learn -- and it still emits notes, so scoring through one
    produces plausible numbers for a model nobody trained. They are restored from the
    checkpoint here, and their absence is an error rather than a silent default.
    """
    if args.decoder == "heuristic":
        if args.encoder == "pyin":
            return GreedyDecoder(pitch_split_semitones=DSP_PITCH_SPLIT_SEMITONES)
        return GreedyDecoder()

    if args.checkpoint is None:
        raise ValueError(
            "--decoder semicrf needs --checkpoint: its potentials are trained "
            "parameters, and a decoder built without them is not a model that was fit."
        )
    checkpoint = Checkpoint.load(args.checkpoint)
    if not checkpoint.objective_state:
        raise ValueError(
            f"{args.checkpoint} carries no objective parameters, so it was not trained "
            "with a structured objective -- most likely --objective frame. Scoring it "
            "through the semi-CRF would report an untrained CRF over its cues; use "
            "--decoder heuristic, or a checkpoint from --objective semicrf."
        )
    return checkpoint.decoder()


# Each entry is (default root, factory). Adding a dataset is one line here plus the
# Dataset subclass itself -- the runner, metrics and report are untouched, because a
# dataset owns how its own references are built. Factories take the parsed args so a
# dataset can read whichever flags it cares about and ignore the rest.
DATASETS: dict[str, tuple[str, Callable[[str, argparse.Namespace], Dataset]]] = {
    # Stems only, so it ignores --source: FiloBass ships no mixture and a bichannel
    # checkpoint cannot be scored on it. See core/data/filobass.py.
    "filobass": (
        "data/raw/filobass",
        lambda root, a: FiloBassDataset(root, limit=a.limit),
    ),
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
        help="pyin (DSP floor) or learned (a trained CueNet, with --checkpoint).",
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
        # Typed at the base: the two levels produce different reports, and everything
        # below here asks only for what `Report` promises (`render`, `write_json`).
        report: Report
        if args.level == "note":
            transcriber = Transcriber(encoder, build_decoder(args))
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
