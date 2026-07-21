"""Train the onset head on Slakh, scored by note-level F1.

    uv run python -m stemify.training.train_onset --root data/raw/slakh2100 \
        --cache-dir data/cache/onset --tracking-uri file:./mlruns \
        --sources stem,mix,demucs --limit 400

Domain randomisation: the input is drawn from several source conditions
(``--sources``) while the target is always the clean-stem MIDI, so the head is
robust to the condition it will actually see. stem+mix need no separation; adding
``demucs`` separates the train set first (cached; ~8s/track on MPS).

Validation runs on every deployment condition (``--val-sources``, default
stem,mix,demucs), not just clean stems: cheap BCE val loss on the stem set drives
early stopping inline, while ``AsyncValidationCallback`` scores each checkpoint as
a real ``LearnedSegmenter`` -- one ``OnsetValidator`` per condition, each caching
its pyin contours so a score is a model forward, not a pyin re-run -- and logs
``note_f1_<condition>`` to MLflow. The gap between ``note_f1_stem`` and
``note_f1_demucs`` is the verdict on how well the learned onsets survive separation
(the DSP stem floor is ~80% on val / 73% on test).
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from stemify.data.slakh import SlakhDataset
from stemify.training.callbacks import (
    AsyncValidationCallback,
    ConsoleCallback,
    MLflowCallback,
)
from stemify.training.config import TrainConfig
from stemify.training.dataset import FrameWindowDataset, collate, prepare_examples
from stemify.training.model import FrameTagger, TCNEncoder, onset_head
from stemify.training.trainer import Trainer
from stemify.training.validation import OnsetValidator
from stemify.transforms import AudioTransform, DemucsSeparator, Identity

HEADS = ["onset"]

# tag -> (Slakh source, input transform). Target is always the clean stem MIDI,
# regardless of which source the *features* come from.
SOURCES: dict[str, tuple[str, AudioTransform]] = {
    "stem": ("stem", Identity()),
    "mix": ("mix", Identity()),
    "demucs": ("mix", DemucsSeparator()),
}


def build_model() -> FrameTagger:
    encoder = TCNEncoder()
    return FrameTagger(encoder, {"onset": onset_head(encoder.hidden)})


def build_validators(
    root: str, val_limit: int, conditions: list[str]
) -> dict[str, OnsetValidator]:
    """One ``OnsetValidator`` per condition, keyed by tag. A condition is just a
    ``(source, transform)`` pair -- exactly what a validator wraps -- so the same
    ``SOURCES`` table that defines the *training* conditions defines the validation
    ones. Each caches its own pyin contours; the demucs condition separates its val
    audio through the (cached) DemucsSeparator at construction."""
    validators: dict[str, OnsetValidator] = {}
    for tag in conditions:
        source, transform = SOURCES[tag]
        ds = SlakhDataset(root, split="validation", source=source, limit=val_limit)
        validators[tag] = OnsetValidator(ds, transform)
    return validators


def _eval_fn(validators: dict[str, OnsetValidator]):
    """An async ``eval_fn``: rebuild the model on CPU from the handed-in weights and
    score it through each cached validator (off the training device). Metric names
    are suffixed with the condition tag (``note_f1_stem``, ``note_f1_demucs``, ...)
    so MLflow charts one series per condition and the robustness gap is visible."""

    def eval_fn(state_dict, epoch):
        model = build_model()
        model.load_state_dict(state_dict)
        metrics: dict[str, float] = {}
        for tag, validator in validators.items():
            for key, value in validator.score(model, device="cpu").items():
                metrics[f"{key}_{tag}"] = value
        return metrics

    return eval_fn


def _windows(paths, config: TrainConfig) -> FrameWindowDataset:
    return FrameWindowDataset(paths, HEADS, config.window_frames, config.hop_frames)


def train_onset(
    root: str,
    cache_dir: Path,
    config: TrainConfig,
    sources: list[str] | None = None,
    val_conditions: list[str] | None = None,
    val_limit: int = 40,
    val_every: int = 5,
    tracking_uri: str | None = None,
    experiment: str = "onset-detection",
    run_name: str = "tcn-v1",
    limit: int | None = None,
    out: Path | None = None,
) -> None:
    sources = sources or ["stem", "mix"]
    # Validate on every deployment condition (not just clean stems), so the note
    # F1 gap between stem and demucs is visible during training rather than only
    # at final test. Independent of train sources: monitor the full breakdown.
    val_conditions = val_conditions or ["stem", "mix", "demucs"]
    out = out or cache_dir / "onset_model.pt"

    # One feature cache per source (separate subdir, so stem and mix -- both the
    # identity transform -- do not collide), unioned for domain randomisation.
    train_paths = []
    for tag in sources:
        source, transform = SOURCES[tag]
        ds = SlakhDataset(root, split="train", source=source, limit=limit)
        train_paths += prepare_examples(ds, HEADS, cache_dir / "train" / tag, transform)

    val_ds = SlakhDataset(root, split="validation", source="stem", limit=val_limit)
    val_paths = prepare_examples(val_ds, HEADS, cache_dir / "val" / "stem")

    train_loader = DataLoader(
        _windows(train_paths, config), batch_size=config.batch_size,
        shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        _windows(val_paths, config), batch_size=config.batch_size, collate_fn=collate,
    )

    trainer = Trainer(build_model(), config)

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
            params={"encoder": "tcn", "sources": ",".join(sources),
                    "val_conditions": ",".join(val_conditions),
                    "lr": config.learning_rate, "window": config.window_frames,
                    "train_tracks": len(train_paths)},
        )
        # Precompute pyin (and, for demucs, the separation) over each val condition
        # once; async scores are then just model forwards. Slowest at construction
        # is demucs -- it separates its val audio before pyin.
        print(
            f"caching val contours for {val_limit} tracks x conditions "
            f"{val_conditions}...", flush=True,
        )
        validators = build_validators(root, val_limit, val_conditions)
        callbacks += [
            mlflow,
            AsyncValidationCallback(_eval_fn(validators), mlflow.sink, every=val_every),
        ]

    trainer.fit(train_loader, val_loader, callbacks=callbacks)

    # fit() restores the best-val checkpoint into trainer.model; persist it so the
    # trained onset head survives the process (LearnedSegmenter loads it) rather
    # than living only as MLflow metrics. Save the state_dict plus the metadata
    # LearnedSegmenter needs to rebuild the architecture before loading weights.
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": trainer.model.state_dict(), "heads": HEADS, "encoder": "tcn"},
        out,
    )
    print(f"saved onset model -> {out}", flush=True)
    if mlflow is not None:
        mlflow.log_artifact(out)


def main() -> None:
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
    ap.add_argument("--run-name", default="tcn-v1", help="MLflow run name")
    ap.add_argument("--limit", type=int, help="cap train tracks per source")
    ap.add_argument("--val-limit", type=int, default=40)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path,
                    help="checkpoint path (default <cache-dir>/onset_model.pt)")
    args = ap.parse_args()

    config = TrainConfig(
        heads=("onset",), max_epochs=args.epochs, batch_size=args.batch_size,
    )
    train_onset(
        args.root, args.cache_dir, config, sources=args.sources.split(","),
        val_conditions=args.val_sources.split(","),
        val_limit=args.val_limit, val_every=args.val_every,
        tracking_uri=args.tracking_uri, limit=args.limit, out=args.out,
    )


if __name__ == "__main__":
    main()