"""Cached validation for the async validator: note-level and cues-level.

``CheckpointValidator`` builds the CQT and reference notes for the val tracks *once* at
construction and holds them, so each subsequent score is a model forward and the
scoring -- seconds, not minutes. That is what makes async validation every few epochs
cheap enough to leave on.

**There is no DSP contour any more.** ``ValItem`` used to hold a deferred pyin source
that a checkpoint without a pitch head would reach through -- a lazy property, a
raising branch, a ``needs_contour`` question the encoder answered, and a second disk
cache behind it. All of it existed for checkpoints predating the pitch head. The
contour is the model's now, produced in the same forward pass as onset and voicing, so
a val example is features and labels and nothing else.

From that one forward per track it scores both altitudes: ``score_examples`` runs the
decoder for note-level metrics, ``score_cues_examples`` scores the encoder's cues
(onset-curve and voicing-head quality). Both are pure cores (given precomputed
examples) that delegate to the shared ``metrics`` scorers -- ``score_notes`` and
``score_cues`` -- so the in-training numbers cannot drift from the offline harness,
and both are tested with synthetic inputs and a fake model, no audio.
"""

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import numpy as np

from stemify.persistence.store import Store
from stemify.transcription.decoder import Decoder
from stemify.transcription.greedy import GreedyDecoder
from stemify.models.learned_encoder import LearnedEncoder
from stemify.models.framenet import FrameNet
from stemify.transcription.frames import Frames
from stemify.corpus.base import Dataset, NoteAnnotated
from stemify.audio.features import FrameFeatures, extract_features, stack_channels
from stemify.metrics import (
    AlignedPairs,
    MetricResult,
    OnsetPRResult,
    VoicingAUROCResult,
    score_cues,
    score_notes,
)
from stemify.music.notes import NoteEvent
from stemify.persistence.serde import features_npy
from stemify.conditions import CONDITIONS, Condition
from stemify.examples import prepare_examples
from stemify.music.labels import onset_targets

@dataclass
class ValItem:
    """One val track, everything fixed across checkpoints precomputed.

    FrameFeatures are held either in memory (``features``, e.g. tests or the no-store
    path) or lazily on disk (``feature_paths``, the cached paths -- one per model
    input channel). Holding all val features in memory for a full-scale val set is
    ~21 GB across three conditions, so when they are cached to disk we keep only the
    paths and load the tensor transiently per score -- see ``load_features``. Only
    ``frame_times`` (small) is kept resident, since scoring needs it for onset
    timing."""

    frame_times: np.ndarray       # (T,)
    notes: list[NoteEvent]        # reference notes
    features: np.ndarray | None = None      # (n_bins, T) or (C, n_bins, T), if resident
    feature_paths: tuple[Path, ...] = ()    # else load from disk on demand, per channel
    voicing_target: np.ndarray | None = None  # (T,) gt voiced mask, for voicing AUROC

    def load_features(self) -> FrameFeatures:
        """The track's CQT paired with its frame grid, from memory or disk. The disk
        read is transient -- freed after each score -- so memory stays flat over the
        run. Each ``feature_paths`` entry is a CachedItem npz; the CQT is its .npy
        sidecar. Several are stacked into the model's channel axis, matching how the
        training loader assembles a window."""
        if self.features is not None:
            return FrameFeatures(self.features, self.frame_times)
        channels = [
            FrameFeatures(np.load(features_npy(p)), self.frame_times)
            for p in self.feature_paths
        ]
        return stack_channels(channels)


# A segmentation step: one val example -> notes. Takes the whole ``ValItem`` rather
# than its features so the caller stays free to read whatever an example carries
# without the validator having to know what a given decoder needs.
SegmentFn = Callable[[ValItem], list[NoteEvent]]
# The encode step: one val example -> the encoder's Frames. Used for cues-level scoring
# (onset + voicing head quality) -- a checkpoint's LearnedEncoder.encode_features,
# so the validator stays model-agnostic.
EncodeFn = Callable[[ValItem], Frames]


def score_examples(
    segment: SegmentFn,
    examples: Sequence[ValItem],
) -> dict[str, float]:
    """Score a segmentation function over precomputed val examples: for each, hand
    its cached CQT to ``segment`` and compare the notes it returns to the
    references.

    The validator owns no segmentation logic -- it just calls the function it was
    given (typically a decoder over a checkpoint's ``LearnedEncoder.encode_features``,
    the same encoder/decoder pair deployment composes into a ``Transcriber``), so how
    notes are made lives entirely there and eval can't drift from deploy. Pure given
    ``examples`` and ``segment``."""
    refs: list[list[NoteEvent]] = []
    ests: list[list[NoteEvent]] = []
    for ex in examples:
        ests.append(segment(ex))
        refs.append(ex.notes)
    # Score through the shared note core, so the val numbers use the same metrics and
    # tolerances the offline NoteEvaluation reports.
    #
    # Every number both metrics report, under the metrics' own names. This used to
    # rename them on the way out -- `f1` became `note_f1` here and a `_NOTE_KEY` table
    # renamed it back to `f1` at the log site, a round trip through two modules that
    # left no single place the logged names could be read off. The one projection that
    # is real (a subset, three of them renamed) now lives at that log site, in
    # ``NOTE_LOG_KEYS``.
    #
    # All four false-positive counts travel, not just fragmentation. They are computed
    # together anyway, and precision alone cannot say *which* error a change fixed: a
    # "rest" FP is a note invented where nothing sounds, a "fragmentation" FP is one
    # real note reported as several, and an "interior_offpitch" FP lands inside a real
    # note at the wrong pitch. Those have different causes and different fixes, so
    # collapsing them into one precision number throws the diagnosis away.
    scalars: dict[str, float] = {}
    for result in score_notes(refs, ests):
        scalars |= result.scalars
    return scalars


def score_cues_examples(
    encode: EncodeFn,
    examples: Sequence[ValItem],
) -> list[MetricResult]:
    """Cues-level (encoder) quality over precomputed val examples: onset-curve and
    voicing-head, both threshold-free, in one model pass per track.

    For each example, encode its cached features into ``Frames`` and turn that into
    the aligned per-channel pairs ``score_cues`` pools: onset (curve vs an impulse
    target built from the reference note onsets on the CQT grid) and voicing (P(voiced)
    vs the cached voiced label). Pitch is *not* scored here -- the offline
    ``CuesEvaluation`` owns frame-level pitch, where a reference contour is available.
    Voicing drops out for an onset-only model. Like
    ``score_examples`` this takes a callable, not a model, so the validator stays
    model-agnostic; the shared ``metrics.frame.scoring.score_cues`` keeps it from drifting from
    the offline harness."""
    tracks: list[AlignedPairs] = []
    for ex in examples:
        frames = encode(ex)
        onset_true = onset_targets(
            np.array([n.onset for n in ex.notes], dtype=float), frames.frame_times
        )
        voicing = None
        # `asarray`, because `Frames` holds tensors under training and numpy at decode
        # and the metrics are numpy. It is a view on a CPU tensor and raises on a CUDA
        # one -- correct either way: a validator scoring accelerator-resident evidence
        # would silently pin the graph it came from.
        if frames.voicing is not None and ex.voicing_target is not None:
            voicing = (np.asarray(frames.voicing), ex.voicing_target)
        tracks.append(
            AlignedPairs(onset=(np.asarray(frames.onset), onset_true), voicing=voicing)
        )
    return score_cues(tracks)


class CheckpointValidator:
    """Precompute CQT + reference notes once; score models cheaply."""

    def __init__(
        self,
        dataset: Dataset[NoteAnnotated],
        condition: Condition | None = None,
        store: Store | None = None,
    ) -> None:
        # A bare stem condition is the neutral default -- one channel, no transform
        # -- so a caller scoring clean audio need not name one.
        condition = condition or CONDITIONS["clean"]

        # With a store the CQT is disk-backed; without one, features are held in memory
        # (the test/small-set path).
        #
        # The CQT is the same artifact the training loader caches. Since features are
        # keyed on the resolved audio path, *every* condition now reuses whatever the
        # loader already built -- not just the stem tag, which was previously a
        # special case worth a comment. Prepare the voicing target too (a cheap
        # backfill, no CQT recompute), so the per-frame voiced label the AUROC path
        # needs is present for every condition.
        if store is not None:
            prepare_examples(dataset, ["onset", "voicing"], store, condition.primary.transform)
            # Every channel's CQT, keyed on its own resolved audio -- so a second
            # channel is typically a pure store hit from whichever run already built
            # that condition. Ordered per track, channel 0 first.
            per_track = condition.channel_paths(dataset, store)

        self.examples: list[ValItem] = []
        for i, item in enumerate(dataset):
            # A transform needs somewhere to write even when features are not being
            # cached: `store=None` means "hold the CQTs resident", not "separate to
            # nowhere". Identity ignores it; a separator would not.
            into = store or Store()
            if store is not None:
                # Keep only the paths + small frame_times/label resident; the feature
                # tensors are loaded transiently per score (load_features), so the
                # whole val set's CQT never sits in memory at once.
                fpaths = per_track[i]
                cached = np.load(fpaths[0])
                frame_times = cached["frame_times"]
                voicing_target = (
                    cached["target_voicing"] if "target_voicing" in cached.files else None
                )
                features = None
            else:
                # No store: featurise each channel now and hold them resident. Only
                # the small-set/test path takes this branch.
                feats = stack_channels(
                    [extract_features(c.audio(item, into)) for c in condition.channels]
                )
                frame_times = feats.frame_times
                features = feats.values
                voicing_target = None
                fpaths = ()
            self.examples.append(
                ValItem(
                    frame_times=frame_times,
                    notes=item.reference_notes(),
                    features=features,
                    feature_paths=fpaths,
                    voicing_target=voicing_target,
                )
            )

    @torch.no_grad()
    def score(self, segment: SegmentFn) -> dict[str, float]:
        """Note-level metrics. **Under ``no_grad``**, which is not merely an economy.

        Everything here is inference, but nothing on this path used to say so: the
        encoder's own forward is guarded, and the *decoder's* was not. That cost nothing
        while every span potential was a handful of adds over cumsums -- and it stopped
        being free the moment a scorer became a network. ``NeuralSpanScorer.precompute``
        materialises one MLP evaluation per span, so a six-minute track retains ~5 GB of
        graph that no backward will ever consume, on top of the training step's own.
        """
        return score_examples(segment, self.examples)

    @torch.no_grad()
    def score_cues(self, encode: EncodeFn) -> list[MetricResult]:
        return score_cues_examples(encode, self.examples)


# --------------------------------------------------------------------------- #
# Wiring a run's validation: the validators, and the eval_fn the callback runs.
#
# These were in `training/train.py`, which is where they grew -- but they are this
# module's concern, not the entry point's: they construct the CheckpointValidators
# above and compose them into the per-epoch scoring function
# `AsyncValidationCallback` calls. Keeping them beside the machinery they drive
# means a change to what validation measures lands in one file.
# --------------------------------------------------------------------------- #

NOTE_LOG_KEYS: dict[str, str] = {
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "fragmentation": "fragmentation",
    "fp": "fp",
    "rest": "fp_rest",
    "interior_offpitch": "fp_interior_offpitch",
}
"""``score_examples`` key -> the leaf name under ``note/<condition>/``.

The one projection between what the note metrics report and what a run logs: seven of
the ten numbers, three of them renamed (``rest`` is ``fp_rest`` on a chart, because
``fp`` alone is ambiguous sitting next to PRF's). ``tp`` and ``fn`` are omitted --
they are recoverable from the rates and the counts that are logged.

It used to be two tables in two modules that cancelled each other out. Exhaustive
rather than a fallback lookup, so a metric key that disappears raises here instead of
quietly logging its own internal name.

**These names are load-bearing.** Every run in docs/experiments.md is recorded under
them, and ``--monitor`` names one on the command line, so a rename starts a new series
rather than extending the old one and can leave checkpoint selection reading a key
that no longer exists. ``tests/metrics/test_logged_schema.py`` pins them.
"""

# The cues-level results the async validator can produce. Pitch is not among them:
# `score_cues_examples` supplies no pitch pair -- frame-level pitch needs a reference
# contour, which is the offline harness's business. Named as *types*, so the metric
# names below are known without scoring anything.
CUES_RESULT_TYPES: tuple[type[MetricResult], ...] = (OnsetPRResult, VoicingAUROCResult)


def harness_metric_names(conditions: Iterable[str]) -> list[str]:
    """Every ``note/*`` and ``cues/*`` name this validator will log, before it runs.

    Knowable statically because a result's record is a ``ClassVar`` on its type rather
    than whatever keys a ``dict`` ended up with -- which is what lets ``--monitor`` be
    checked against reality at startup instead of at the first epoch boundary, an hour
    into a run, on the way to deciding which checkpoint survives.

    **An upper bound, not an exact set.** ``score_cues`` drops a channel no track can
    supply, so an onset-only checkpoint logs no ``voicing_*`` series even though they
    are listed here. That is the one case a ``--monitor`` check against this list still
    passes and the run still fails -- narrower than the everything-was-unchecked it
    replaces, and it needs the model to answer, which this function has no access to.
    """
    names: list[str] = []
    for tag in conditions:
        names += [f"note/{tag}/{leaf}" for leaf in NOTE_LOG_KEYS.values()]
        names += [
            f"cues/{tag}/{key}" for t in CUES_RESULT_TYPES for key in t.SCALARS
        ]
    return names


def build_validators(
    root: str,
    val_limit: int,
    conditions: list[str],
    store: Store | None = None,
    split: str = "validation",
) -> dict[str, CheckpointValidator]:
    """One ``CheckpointValidator`` per condition, keyed by tag. A condition names the
    channels the model reads -- exactly what a validator has to reproduce -- so the
    same ``CONDITIONS`` table that defines the *training* conditions defines the
    validation ones.

    Every condition shares the one ``store``: CQTs are keyed on each condition's
    *resolved* audio path, so they cannot collide and no per-tag subdirectory is
    needed. The demucs condition separates its val audio through the (cached)
    DemucsSeparator at construction.

    The per-track loop is serial. It was worth arguing about when a pyin pass ran here;
    now the only per-track cost is the CQT, which the training loader has usually
    already cached."""
    return {
        tag: CheckpointValidator(
            CONDITIONS[tag].dataset(root, split, val_limit),
            CONDITIONS[tag],
            store=store,
        )
        for tag in conditions
    }


def build_eval_fn(
    validators: dict[str, CheckpointValidator],
    report_dir: Path | None = None,
    log_artifact: Callable[[Path, int], None] | None = None,
    build_decoder: Callable[[Mapping[str, Any]], Decoder] | None = None,
    *,
    build_model: Callable[[], FrameNet],
) -> Callable[[Mapping[str, Any], int], Mapping[str, float]]:
    """An async ``eval_fn``: rebuild the model on CPU from the handed-in weights and
    score it through each cached validator (off the training device). Every metric is
    namespaced ``<level>/<condition>/<key>`` so MLflow charts one series per condition
    and the robustness gap between stem and demucs is visible.

    Each checkpoint is scored at both altitudes from the *same* model pass per track:
    note-level (the decoder over the encoder's cues) and cues-level (the encoder's
    onset/voicing head quality, via ``metrics.frame.scoring.score_cues`` -- the same core the
    offline ``CuesEvaluation`` uses). If ``report_dir``/``log_artifact`` are given, the
    full breakdown is also written as a per-epoch JSON and attached to the run, so a
    checkpoint's scores survive as a diffable artifact, not only as metric points.

    ``build_decoder(state) -> Decoder`` is the seam the objective varies: the frame
    heads are grouped by the ``GreedyDecoder``, while a semi-CRF run must be scored
    through the potentials in its own snapshot. Defaulting it to the greedy decoder keeps
    the frame path unchanged, but hardcoding it would mean a semi-CRF run silently
    reported the *greedy decoder's* note F1 over its learned cues -- plausible numbers for
    a model nobody trained.

    ``build_model`` is **required and keyword-only**, and is always
    ``recipe.build_model``. It used to default to a module-level ``build_model`` here
    that built a TCN with onset and voicing heads and no pitch branch -- a second,
    weaker implementation of what the recipe already knows. Nothing took the default,
    but anything that did would have validated a model missing the head the run was
    training, and reported its scores as the run's. The recipe is the only thing that
    knows what this run builds, so it is the only thing allowed to build it."""
    if build_decoder is None:
        def build_decoder(_state: Mapping[str, Any]) -> Decoder:
            return GreedyDecoder()

    def eval_fn(state: Mapping[str, Any], epoch: int) -> Mapping[str, float]:
        model = build_model()  # the recipe's encoder
        # `state` is a Trainer.state_cpu snapshot: the model's weights under "model"
        # and the objective's own parameters under "loss". The decoder is built from
        # the same snapshot, so a structured objective is scored through the
        # potentials that snapshot actually trained -- not a fresh set.
        model.load_state_dict(dict(state["model"]))
        # Wrap the checkpoint once (CPU, off the training device) as the encoder/decoder
        # pair deployment composes into a Transcriber. The decoder scores notes over the
        # encoder's cues; the encoder alone scores its own cues -- one forward, both.
        encoder = LearnedEncoder(model, device="cpu")
        decoder = build_decoder(state)

        def encode(ex: ValItem) -> Frames:
            return encoder.encode_features(ex.load_features())

        def segment(ex: ValItem) -> list[NoteEvent]:
            return decoder.decode(encode(ex))

        metrics: dict[str, float] = {}
        report: dict[str, Any] = {"epoch": epoch, "conditions": {}}
        for tag, validator in validators.items():
            note = validator.score(segment)
            for key, leaf in NOTE_LOG_KEYS.items():
                metrics[f"note/{tag}/{leaf}"] = note[key]
            # Cues-level (encoder) quality: onset-curve AP and voicing-head AUROC,
            # threshold-free. Empty voicing for an onset-only model.
            cues = {r.name: r.scalars for r in validator.score_cues(encode)}
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
