"""Train the onset head on Slakh, scored by note-level F1.

    uv run python -m stemify.training.train_onset --root data/raw/slakh2100 \
        --cache-dir data/cache/onset --tracking-uri file:./mlruns \
        --sources stem,mix --limit 400

Domain randomisation: the input is drawn from several source conditions
(``--sources``) while the target is always the clean-stem MIDI, so the head is
robust to the condition it will actually see. stem+mix need no separation; adding
``demucs`` requires separating the train set first (cached, but hours), so it is
opt-in.

Validation: cheap BCE val loss drives early stopping inline, while
``AsyncValidationCallback`` scores each checkpoint as a real ``LearnedSegmenter``
-- via ``OnsetValidator``, which caches pyin contours so each score is a model
forward, not a pyin re-run -- and logs note-level F1 to MLflow. That F1 versus the
DSP floor (73.1% stem) is the verdict on the learned onsets.
"""

import argparse
from pathlib import Path

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


def _eval_fn(validator: OnsetValidator):
    """An async ``eval_fn``: rebuild the model on CPU from the handed-in weights
    and score it through the cached validator (off the training device)."""

    def eval_fn(state_dict, epoch):
        model = build_model()
        model.load_state_dict(state_dict)
        return validator.score(model, device="cpu")

    return eval_fn


def _windows(paths, config: TrainConfig) -> FrameWindowDataset:
    return FrameWindowDataset(paths, HEADS, config.window_frames, config.hop_frames)


def train_onset(
    root: str,
    cache_dir: Path,
    config: TrainConfig,
    sources: list[str] | None = None,
    val_limit: int = 40,
    val_every: int = 5,
    tracking_uri: str | None = None,
    limit: int | None = None,
) -> None:
    sources = sources or ["stem", "mix"]

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
    if tracking_uri:
        mlflow = MLflowCallback(
            run_name="onset-tcn", tracking_uri=tracking_uri,
            params={"encoder": "tcn", "sources": ",".join(sources),
                    "lr": config.learning_rate, "window": config.window_frames,
                    "train_tracks": len(train_paths)},
        )
        # Precompute pyin over the val subset once; async scores are then cheap.
        print(f"caching pyin contours for {val_limit} val tracks...", flush=True)
        validator = OnsetValidator(
            SlakhDataset(root, split="validation", source="stem", limit=val_limit)
        )
        callbacks += [
            mlflow,
            AsyncValidationCallback(_eval_fn(validator), mlflow.sink, every=val_every),
        ]

    trainer.fit(train_loader, val_loader, callbacks=callbacks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/raw/slakh2100")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache/onset"))
    ap.add_argument("--sources", default="stem,mix",
                    help="comma-separated input conditions from " + ",".join(SOURCES))
    ap.add_argument("--tracking-uri", help="MLflow tracking URI, e.g. file:./mlruns")
    ap.add_argument("--limit", type=int, help="cap train tracks per source")
    ap.add_argument("--val-limit", type=int, default=40)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    config = TrainConfig(
        heads=("onset",), max_epochs=args.epochs, batch_size=args.batch_size,
    )
    train_onset(
        args.root, args.cache_dir, config, sources=args.sources.split(","),
        val_limit=args.val_limit, val_every=args.val_every,
        tracking_uri=args.tracking_uri, limit=args.limit,
    )


if __name__ == "__main__":
    main()