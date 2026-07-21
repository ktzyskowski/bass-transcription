"""Train the onset head on Slakh, scored by note-level F1.

    uv run python -m stemify.training.train_onset --root data/raw/slakh2100 \
        --cache-dir data/cache/onset --tracking-uri file:./mlruns --limit 50

The wiring that matters: the cheap BCE val loss drives early stopping inline,
while ``AsyncValidationCallback`` scores each checkpoint the way it will actually
be judged -- as a ``LearnedSegmenter`` through ``NoteEvaluation`` -- off the loop,
on CPU, and logs note-level F1 to MLflow. That F1 is the number the DSP floor
(64.7% on demucs, the deployment condition) has to be beaten by.

First-run shape: onset head only, stem source. Domain randomisation over
stem/mix/demucs is added by preparing more per-source caches (the cache key
already includes the transform) and concatenating them; voicing/pitch heads are
added to ``heads`` once their targets are trusted.
"""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from stemify.baselines.pyin import PyinBaseline
from stemify.data.slakh import SlakhDataset
from stemify.evaluation import NoteEvaluation
from stemify.training.callbacks import (
    AsyncValidationCallback,
    ConsoleCallback,
    MLflowCallback,
)
from stemify.training.config import TrainConfig
from stemify.training.dataset import FrameWindowDataset, collate, prepare_examples
from stemify.training.model import FrameTagger, TCNEncoder, onset_head
from stemify.training.segmenter import LearnedSegmenter
from stemify.training.trainer import Trainer

HEADS = ["onset"]


def build_model() -> FrameTagger:
    encoder = TCNEncoder()
    return FrameTagger(encoder, {"onset": onset_head(encoder.hidden)})


def note_f1_eval_fn(val_dataset: SlakhDataset, device: str = "cpu"):
    """An ``eval_fn`` for the async validator: score a checkpoint as a real
    ``LearnedSegmenter`` and return its note-level metrics.

    Rebuilds the model on ``device`` (CPU, to stay off the training accelerator)
    from the handed-in weights. Runs pyin for pitch each call; caching the val
    contours would remove that cost, since only the onset model changes between
    checkpoints -- the first optimisation if validation dominates.
    """
    baseline = PyinBaseline()

    def eval_fn(state_dict, epoch):
        model = build_model()
        model.load_state_dict(state_dict)
        segmenter = LearnedSegmenter(model, device=device)
        report = NoteEvaluation(
            val_dataset, baseline, segmenter=segmenter, progress=False
        ).run()
        return {
            "note_f1": report.scalar("note-prf", "f1"),
            "note_precision": report.scalar("note-prf", "precision"),
            "note_recall": report.scalar("note-prf", "recall"),
        }

    return eval_fn


def _windows(paths, config: TrainConfig) -> FrameWindowDataset:
    return FrameWindowDataset(paths, HEADS, config.window_frames, config.hop_frames)


def train_onset(
    root: str,
    cache_dir: Path,
    config: TrainConfig,
    tracking_uri: str | None = None,
    limit: int | None = None,
) -> None:
    train_ds = SlakhDataset(root, split="train", source="stem", limit=limit)
    val_ds = SlakhDataset(root, split="validation", source="stem", limit=limit)

    # Cache features + onset targets once (keyed by transform); the CQT is the
    # cost and must not recur per epoch.
    train_paths = prepare_examples(train_ds, HEADS, cache_dir / "train")
    val_paths = prepare_examples(val_ds, HEADS, cache_dir / "val")

    train_loader = DataLoader(
        _windows(train_paths, config), batch_size=config.batch_size,
        shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        _windows(val_paths, config), batch_size=config.batch_size, collate_fn=collate,
    )

    trainer = Trainer(build_model(), config)

    callbacks: list = [ConsoleCallback()]
    if tracking_uri:
        mlflow = MLflowCallback(
            run_name="onset-tcn", tracking_uri=tracking_uri,
            params={"encoder": "tcn", "heads": ",".join(config.heads),
                    "lr": config.learning_rate, "window": config.window_frames},
        )
        callbacks.append(mlflow)
        callbacks.append(
            AsyncValidationCallback(note_f1_eval_fn(val_ds), mlflow.sink, every=5)
        )

    trainer.fit(train_loader, val_loader, callbacks=callbacks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/raw/slakh2100")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache/onset"))
    ap.add_argument("--tracking-uri", help="MLflow tracking URI, e.g. file:./mlruns")
    ap.add_argument("--limit", type=int, help="cap tracks per split (smoke run)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    config = TrainConfig(
        heads=("onset",), max_epochs=args.epochs, batch_size=args.batch_size,
    )
    train_onset(args.root, args.cache_dir, config, args.tracking_uri, args.limit)


if __name__ == "__main__":
    main()