"""Train the onset head on Slakh, scored by note-level F1.

    uv run python -m stemify.training.train --root data/raw/slakh2100 \
        --cache-dir data/cache/onset --tracking-uri file:./mlruns \
        --sources stem,mix,demucs --limit 400

Domain randomisation: the input is drawn from several source conditions
(``--sources``) while the target is always the clean-stem MIDI, so the head is
robust to the condition it will actually see. stem+mix need no separation; adding
``demucs`` separates the train set first (cached; ~8s/track on MPS).

Validation runs on every deployment condition (``--val-sources``, default
stem,mix,demucs), not just clean stems: cheap BCE val loss on the stem set drives
early stopping inline, while ``AsyncValidationCallback`` scores each checkpoint
through the real ``NeuralEncoder`` + ``HeuristicDecoder`` pair -- one
``OnsetValidator`` per condition, each caching its pyin contours so a score is a
model forward, not a pyin re-run. From that one forward it logs both altitudes to
MLflow -- ``note/<condition>/f1`` (decoder) and ``cues/<condition>/{onset_ap,
voicing_auroc}`` (encoder) -- and attaches a per-epoch ``validation/`` JSON artifact
with the full breakdown. The gap between ``note/stem/f1`` and ``note/demucs/f1`` is
the verdict on how well the learned onsets survive separation (the DSP stem floor is
~80% on val / 73% on test).
"""

import argparse
import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from stemify.cache import Cache
from stemify.data.slakh import SlakhDataset
from stemify.decoders.heuristic import HeuristicDecoder
from stemify.encoders.heads import onset_head, voicing_head
from stemify.encoders.networks import TCN
from stemify.encoders.neural import NeuralEncoder
from stemify.encoders.tagger import FrameTagger
from stemify.training.callbacks import (
    AsyncValidationCallback,
    ConsoleCallback,
    MLflowCallback,
)
from stemify.training.config import TrainConfig
from stemify.training.datasets.frame import FrameWindowDataset, collate
from stemify.training.examples import prepare_examples
from stemify.training.losses import FrameBCELoss
from stemify.training.trainer import Trainer
from stemify.training.validation import OnsetValidator
from stemify.transforms import AudioTransform, DemucsSeparator, Identity

logger = logging.getLogger(__name__)

HEADS = ["onset", "voicing"]

# score_examples' metric keys -> the short leaf name under note/<condition>/, so
# MLflow groups them: note/stem/f1, note/demucs/recall, ... one section per metric.
_NOTE_KEY = {
    "note_f1": "f1", "note_precision": "precision",
    "note_recall": "recall", "fragmentation": "fragmentation",
}

# tag -> (Slakh source, input transform). Target is always the clean stem MIDI,
# regardless of which source the *features* come from.
SOURCES: dict[str, tuple[str, AudioTransform]] = {
    "stem": ("stem", Identity()),
    "mix": ("mix", Identity()),
    "demucs": ("mix", DemucsSeparator()),
}


def build_model() -> FrameTagger:
    backbone = TCN()
    return FrameTagger(
        backbone,
        {"onset": onset_head(backbone.hidden), "voicing": voicing_head(backbone.hidden)},
    )


def build_validators(
    root: str,
    val_limit: int,
    conditions: list[str],
    cache: Cache | None = None,
) -> dict[str, OnsetValidator]:
    """One ``OnsetValidator`` per condition, keyed by tag. A condition is just a
    ``(source, transform)`` pair -- exactly what a validator wraps -- so the same
    ``SOURCES`` table that defines the *training* conditions defines the validation
    ones.

    Every condition shares the one ``cache``: contours and CQTs are keyed on each
    condition's *resolved* audio path, so they cannot collide and no per-tag
    subdirectory is needed. pyin then runs once per file ever rather than at every
    launch -- what makes a large val set affordable. The demucs condition separates
    its val audio through the (cached) DemucsSeparator at construction.

    The per-track loop is serial, and whether that costs anything **depends on the
    numeric backend**. On macOS/Accelerate pyin's FFTs already spread across cores and
    a process pool only oversubscribes; on Linux/OpenBLAS (scipy's pocketfft is
    single-threaded unless handed ``workers``) it pins one core and a pool measures
    ~4.9x. Serial is the right default because the cache means this is paid once per
    file ever -- but see ``docs/refactor-plan.md`` before concluding a pool is
    pointless on a given machine."""
    return {
        tag: OnsetValidator(
            SlakhDataset(
                root, split="validation", source=SOURCES[tag][0], limit=val_limit
            ),
            SOURCES[tag][1],
            cache=cache,
        )
        for tag in conditions
    }


def _eval_fn(
    validators: dict[str, OnsetValidator],
    report_dir: Path | None = None,
    log_artifact: Callable[[Path, int], None] | None = None,
):
    """An async ``eval_fn``: rebuild the model on CPU from the handed-in weights and
    score it through each cached validator (off the training device). Every metric is
    namespaced ``<level>/<condition>/<key>`` so MLflow charts one series per condition
    and the robustness gap between stem and demucs is visible.

    Each checkpoint is scored at both altitudes from the *same* model pass per track:
    note-level (the decoder over the encoder's cues) and cues-level (the encoder's
    onset/voicing head quality, via ``metrics.cues.score_cues`` -- the same core the
    offline ``CuesEvaluation`` uses). If ``report_dir``/``log_artifact`` are given, the
    full breakdown is also written as a per-epoch JSON and attached to the run, so a
    checkpoint's scores survive as a diffable artifact, not only as metric points."""

    def eval_fn(state_dict, epoch):
        model = build_model()
        model.load_state_dict(state_dict)
        # Wrap the checkpoint once (CPU, off the training device) as the encoder/decoder
        # pair deployment composes into a Segmenter. The decoder scores notes over the
        # encoder's cues; the encoder alone scores its own cues -- one forward, both.
        encoder = NeuralEncoder(model, device="cpu")
        decoder = HeuristicDecoder()

        def segment(contour, features):
            return decoder.decode(encoder.encode_features(contour, features))

        metrics: dict[str, float] = {}
        report: dict = {"epoch": epoch, "conditions": {}}
        for tag, validator in validators.items():
            note = validator.score(segment)
            for key, value in note.items():
                metrics[f"note/{tag}/{_NOTE_KEY.get(key, key)}"] = value
            # Cues-level (encoder) quality: onset-curve AP and voicing-head AUROC,
            # threshold-free. Empty voicing for an onset-only model.
            cues = {r.name: r.scalars for r in validator.score_cues(encoder.encode_features)}
            for scalars in cues.values():
                for key, value in scalars.items():
                    metrics[f"cues/{tag}/{key}"] = value
            report["conditions"][tag] = {"note": note, "cues": cues}

        if report_dir is not None and log_artifact is not None:
            report_dir.mkdir(parents=True, exist_ok=True)
            path = report_dir / f"validation_epoch{epoch:03d}.json"
            path.write_text(json.dumps(report, indent=2))
            log_artifact(path, epoch)
        return metrics

    return eval_fn


def _windows(paths, config: TrainConfig) -> FrameWindowDataset:
    return FrameWindowDataset(paths, HEADS, config.window_frames, config.hop_frames)


def train(
    root: str,
    cache_dir: Path,
    config: TrainConfig,
    sources: list[str] | None = None,
    val_conditions: list[str] | None = None,
    val_limit: int = 40,
    val_every: int = 5,
    tracking_uri: str | None = None,
    experiment: str = "onset-detection",
    run_name: str = "tcn-v2",
    batch_log_every: int = 100,
    limit: int | None = None,
    out: Path | None = None,
) -> None:
    sources = sources or ["stem", "mix"]
    # Validate on every deployment condition (not just clean stems), so the note
    # F1 gap between stem and demucs is visible during training rather than only
    # at final test. Independent of train sources: monitor the full breakdown.
    val_conditions = val_conditions or ["stem", "mix", "demucs"]
    cache = Cache(cache_dir)
    out = out or cache_dir / "onset_model.pt"

    # One cache for every condition and both splits. Features are content-addressed on
    # the resolved audio path, so a stem, a mix and a separated mix of the same track
    # land on distinct keys without any tag/split subdirectory to keep them apart --
    # and the val conditions transparently reuse whatever this loop already built.
    train_paths = []
    for tag in sources:
        source, transform = SOURCES[tag]
        ds = SlakhDataset(root, split="train", source=source, limit=limit)
        train_paths += prepare_examples(ds, HEADS, cache, transform)

    val_ds = SlakhDataset(root, split="validation", source="stem", limit=val_limit)
    val_paths = prepare_examples(val_ds, HEADS, cache)

    # persistent_workers keeps the worker pool alive between epochs -- without it
    # the fork cost is paid every epoch and cancels the benefit on short epochs.
    # It is only valid when there are workers, so gate it.
    loader_kw: dict = {
        "batch_size": config.batch_size,
        "collate_fn": collate,
        "num_workers": config.num_workers,
    }
    if config.num_workers > 0:
        loader_kw["persistent_workers"] = True
    train_loader = DataLoader(_windows(train_paths, config), shuffle=True, **loader_kw)
    val_loader = DataLoader(_windows(val_paths, config), **loader_kw)

    trainer = Trainer(build_model(), FrameBCELoss(HEADS, config.pos_weight), config)

    callbacks: list = [ConsoleCallback()]
    mlflow = None
    if tracking_uri:
        # Run names are labels, not keys -- MLflow mints a fresh run_id per launch,
        # so two runs can share a name. Suffix a UTC timestamp so relaunching with
        # the same --run-name yields distinct, self-dating runs ("tcn-v1" stays the
        # groupable prefix) rather than ambiguous duplicates.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        mlflow = MLflowCallback(
            run_name=f"{run_name}-{stamp}", experiment=experiment, tracking_uri=tracking_uri,
            batch_log_every=batch_log_every,
            params={"encoder": "tcn", "sources": ",".join(sources),
                    "val_conditions": ",".join(val_conditions),
                    "lr": config.learning_rate, "window": config.window_frames,
                    "train_tracks": len(train_paths)},
        )
        # Precompute pyin (and, for demucs, the separation) over each val condition
        # once; async scores are then just model forwards. Slowest at construction
        # is demucs -- it separates its val audio before pyin.
        logger.info(
            "caching val contours for %d tracks x conditions %s...",
            val_limit, val_conditions,
        )
        validators = build_validators(root, val_limit, val_conditions, cache=cache)
        # Per-epoch validation reports (all conditions x note+cues) are written here
        # and attached to the run under "validation/", so each scored checkpoint leaves
        # a diffable JSON artifact, not only metric points.
        eval_fn = _eval_fn(
            validators,
            report_dir=cache_dir / "val_reports",
            log_artifact=lambda p, _e: mlflow.log_artifact(p, artifact_path="validation"),
        )
        callbacks += [
            mlflow,
            AsyncValidationCallback(eval_fn, mlflow.sink, every=val_every),
        ]

    trainer.fit(train_loader, val_loader, callbacks=callbacks)

    # fit() restores the best-val checkpoint into trainer.model; persist it so the
    # trained heads survive the process (a NeuralEncoder loads them) rather than
    # living only as MLflow metrics. Save the state_dict plus the metadata needed to
    # rebuild the architecture before loading weights.
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": trainer.model.state_dict(), "heads": HEADS, "encoder": "tcn"},
        out,
    )
    logger.info("saved onset model -> %s", out)
    if mlflow is not None:
        mlflow.log_artifact(out)


def main() -> None:
    # Configure logging once, here at the entry point -- library modules only ever
    # getLogger, never configure, so importing them stays side-effect free.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/raw/slakh2100")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache/onset"))
    ap.add_argument("--sources", default="stem,mix",
                    help="comma-separated train input conditions from " + ",".join(SOURCES))
    ap.add_argument("--val-sources", default="stem,mix,demucs",
                    help="comma-separated val conditions, scored+logged separately")
    ap.add_argument("--tracking-uri", help="MLflow tracking URI, e.g. file:./mlruns")
    ap.add_argument("--experiment", default="onset-detection",
                    help="MLflow experiment name")
    ap.add_argument("--run-name", default="tcn-v2", help="MLflow run name")
    ap.add_argument("--limit", type=int, help="cap train tracks per source")
    ap.add_argument("--val-limit", type=int, default=40)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--batch-log-every", type=int, default=100,
                    help="log a smoothed train loss to MLflow every N steps (0 = off)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader worker subprocesses (0 = main thread)")
    ap.add_argument("--out", type=Path,
                    help="checkpoint path (default <cache-dir>/onset_model.pt)")
    args = ap.parse_args()

    config = TrainConfig(
        heads=tuple(HEADS), max_epochs=args.epochs, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train(
        args.root, args.cache_dir, config, sources=args.sources.split(","),
        val_conditions=args.val_sources.split(","),
        val_limit=args.val_limit, val_every=args.val_every,
        experiment=args.experiment, run_name=args.run_name,
        batch_log_every=args.batch_log_every,
        tracking_uri=args.tracking_uri, limit=args.limit, out=args.out,
    )


if __name__ == "__main__":
    main()