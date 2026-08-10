"""Cached validation for the async validator: note-level and cues-level.

``CheckpointValidator`` builds the CQT and reference notes for the val tracks *once* at
construction and holds them, so each subsequent score is a model forward and the
scoring -- seconds, not minutes. That is what makes async validation every few epochs
cheap enough to leave on.

**The DSP contour is deferred, not precomputed.** It used to be: pyin ran over every
val track at construction, on the reasoning that its contour is fixed across
checkpoints while the model is not. That reasoning stopped holding when the pitch head
shipped. ``LearnedEncoder.encode_features`` *replaces* the contour it is handed with
the head's own output, so for every current run this was a full pyin pass per track
whose result was overwritten on the next line. It survived because nothing measured it
and the docstring still explained it.

So ``ValItem`` holds a *source* rather than a contour, and the caller -- which holds
the encoder and can read ``pitch_source`` -- decides whether to read it. A pitch-head
checkpoint never does. A headless one (five on disk, and ``--use-pyin``) reaches
through and pays once per track, because features alone cannot give pitch.

From that one forward per track it scores both altitudes: ``score_examples`` runs the
decoder for note-level metrics, ``score_cues_examples`` scores the encoder's cues
(onset-curve and voicing-head quality). Both are pure cores (given precomputed
examples) that delegate to the shared ``metrics`` scorers -- ``score_notes`` and
``score_cues`` -- so the in-training numbers cannot drift from the offline harness,
and both are tested with synthetic inputs and a fake model, no audio.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import numpy as np

from stemify.transcription.pitch.cache import ensure_contour
from stemify.persistence.store import Store
from stemify.transcription.decoders.base import Decoder
from stemify.transcription.decoders.greedy import GreedyDecoder
from stemify.models.learned_encoder import LearnedEncoder
from stemify.models.cuenet import CueNet
from stemify.music.contour import Contour
from stemify.transcription.frames import Frames
from stemify.corpus.base import Dataset, NoteAnnotated
from stemify.audio.features import FrameFeatures, extract_features, stack_channels
from stemify.core.metrics import MetricResult, AlignedPairs, score_cues, score_notes
from stemify.music.notes import NoteEvent
from stemify.transcription.pitch import PyinPitch
from stemify.persistence.serde import features_npy
from stemify.core.conditions import CONDITIONS, Condition
from stemify.core.examples import prepare_examples
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
    # How to obtain a DSP contour for channel 0, *if* anything asks. Deferred rather
    # than precomputed: a model with a pitch head produces its own contour and
    # overwrites whatever it is handed, so for every such run this was a pyin pass per
    # val track whose result was discarded on the next line. Only a headless
    # checkpoint (or --use-pyin) reaches through here, and then it pays once.
    contour_source: Callable[[], Contour] | None = None
    _contour: Contour | None = None

    @property
    def contour(self) -> Contour:
        """The DSP contour, computed on first read. Raises rather than returning a
        placeholder: a caller reaching for this has no pitch head, so an empty contour
        would silently transcribe every track as unvoiced."""
        if self._contour is None:
            if self.contour_source is None:
                raise ValueError(
                    "this ValItem was built without a contour source, so it can only "
                    "score an encoder that supplies its own pitch (a model with a "
                    "pitch head). Pass contour_source to score a headless checkpoint."
                )
            self._contour = self.contour_source()
        return self._contour

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
# than ``(contour, features)`` so that *the caller* decides whether a contour is
# needed at all -- it holds the encoder and can read ``pitch_source``, which the
# validator cannot. That is what keeps a pitch-head run from touching pyin: nothing
# reads ``ex.contour``, so nothing computes it.
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
    its contour + cached CQT to ``segment`` and compare the notes it returns to the
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
    # tolerances the offline NoteEvaluation reports. Flatten to the MLflow-friendly
    # note_* keys the callback logs.
    scalars = {r.name: r.scalars for r in score_notes(refs, ests)}
    prf, frag = scalars["note-prf"], scalars["fragmentation"]
    return {
        "note_f1": prf["f1"],
        "note_precision": prf["precision"],
        "note_recall": prf["recall"],
        # All four false-positive counts, not just fragmentation. They are computed
        # together anyway, and precision alone cannot say *which* error a change
        # fixed: a "rest" FP is a note invented where nothing sounds, a
        # "fragmentation" FP is one real note reported as several, and an
        # "interior_offpitch" FP lands inside a real note at the wrong pitch. Those
        # have different causes and different fixes, so collapsing them into one
        # precision number throws the diagnosis away.
        #
        # "fragmentation" keeps its bare name: it is the key every earlier run in
        # docs/experiments.md is recorded under, and renaming it would silently end
        # those series rather than extend them.
        "fragmentation": float(frag["fragmentation"]),
        "fp": float(frag["fp"]),
        "fp_rest": float(frag["rest"]),
        "fp_interior_offpitch": float(frag["interior_offpitch"]),
    }


def score_cues_examples(
    encode: EncodeFn,
    examples: Sequence[ValItem],
) -> list[MetricResult]:
    """Cues-level (encoder) quality over precomputed val examples: onset-curve and
    voicing-head, both threshold-free, in one model pass per track.

    For each example, encode its cached features + contour into ``Frames`` and turn
    that into the aligned per-channel pairs ``score_cues`` pools: onset (curve vs an
    impulse target built from the reference note onsets on the CQT grid) and voicing
    (P(voiced) vs the cached voiced label). Pitch is *not* scored here -- pyin pitch
    does not move between checkpoints, so it belongs to the offline ``CuesEvaluation``,
    not the per-epoch loop. Voicing drops out for an onset-only model. Like
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
    """Precompute CQT + reference notes once; score models cheaply. The DSP contour is
    deferred to a source read only if the scored encoder has no pitch head."""

    def __init__(
        self,
        dataset: Dataset[NoteAnnotated],
        condition: Condition | None = None,
        store: Store | None = None,
    ) -> None:
        # A bare stem condition is the neutral default -- one channel, no transform
        # -- so a caller scoring clean audio need not name one.
        condition = condition or CONDITIONS["clean"]
        estimator = PyinPitch()

        # One ``store`` replaces the old separate ``contour_cache``/``feature_cache``
        # paths, and unifies what their two ``None``s meant: with a store, both the
        # CQT and the contour are disk-backed; without one, features are held in
        # memory and pyin reruns per launch (the test/small-set path).
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
            # The contour describes channel 0: it is the audio being transcribed,
            # and the context channels are never pitch-tracked.
            primary = condition.primary.audio(item)
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
                contour_source = partial(ensure_contour, store, estimator, primary)
            else:
                # No store: featurise each channel now and hold them resident. Only
                # the small-set/test path takes this branch.
                feats = stack_channels(
                    [extract_features(c.audio(item)) for c in condition.channels]
                )
                frame_times = feats.frame_times
                features = feats.values
                voicing_target = None
                fpaths = ()
                contour_source = partial(estimator.estimate, primary)
            self.examples.append(
                ValItem(
                    frame_times=frame_times,
                    contour_source=contour_source,
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

# score_examples' metric keys -> the short leaf name under note/<condition>/, so
# MLflow groups them: note/clean/f1, note/separated/recall, ... one per metric.
_NOTE_KEY = {
    "note_f1": "f1", "note_precision": "precision",
    "note_recall": "recall", "fragmentation": "fragmentation",
    # The remaining false-positive buckets pass through under their own names
    # (fp, fp_rest, fp_interior_offpitch) -- see ``score_examples``.
}


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

    Every condition shares the one ``store``: contours and CQTs are keyed on each
    condition's *resolved* audio path, so they cannot collide and no per-tag
    subdirectory is needed. pyin then runs once per file ever rather than at every
    launch -- what makes a large val set affordable. The demucs condition separates
    its val audio through the (cached) DemucsSeparator at construction.

    The per-track loop is serial, and whether that costs anything **depends on the
    numeric backend**. On macOS/Accelerate pyin's FFTs already spread across cores and
    a process pool only oversubscribes; on Linux/OpenBLAS (scipy's pocketfft is
    single-threaded unless handed ``workers``) it pins one core and a pool measures
    ~4.9x. Serial is the right default because the store means this is paid once per
    file ever -- but see ``docs/refactor-plan.md`` before concluding a pool is
    pointless on a given machine."""
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
    build_model: Callable[[], CueNet],
) -> Callable[[Mapping[str, Any], int], Mapping[str, float]]:
    """An async ``eval_fn``: rebuild the model on CPU from the handed-in weights and
    score it through each cached validator (off the training device). Every metric is
    namespaced ``<level>/<condition>/<key>`` so MLflow charts one series per condition
    and the robustness gap between stem and demucs is visible.

    Each checkpoint is scored at both altitudes from the *same* model pass per track:
    note-level (the decoder over the encoder's cues) and cues-level (the encoder's
    onset/voicing head quality, via ``metrics.cues.score_cues`` -- the same core the
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

        # Whether this checkpoint needs a DSP contour at all. The encoder is the only
        # thing that can answer -- ``pitch_source`` is "learned" when a FrequencyHead
        # names the pitch and "pyin" when nothing else can -- and the validator holds
        # no opinion. This is what keeps a pitch-head run from touching pyin: nothing
        # reads ``ex.contour``, so the deferred source is never called.
        needs_contour = encoder.pitch_source == "pyin"

        def encode(ex: ValItem) -> Frames:
            contour = ex.contour if needs_contour else None
            return encoder.encode_features(contour, ex.load_features())

        def segment(ex: ValItem) -> list[NoteEvent]:
            return decoder.decode(encode(ex))

        metrics: dict[str, float] = {}
        # ``pitch_source`` is recorded because it silently decides ~5 F1: an encoder
        # with a ``FrequencyHead`` names its notes from that head, one without falls
        # back to pyin, and the two produce byte-identical report shapes. Without this
        # a saved report cannot say which pipeline made it -- the same defect as a run
        # whose name did not describe the run.
        report: dict[str, Any] = {
            "epoch": epoch,
            "pitch_source": encoder.pitch_source,
            "conditions": {},
        }
        for tag, validator in validators.items():
            note = validator.score(segment)
            for key, value in note.items():
                metrics[f"note/{tag}/{_NOTE_KEY.get(key, key)}"] = value
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
