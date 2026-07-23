# Experiment log — bass onset/note transcription

Running log of what we've trained, on what, and how it scored. Newest first.
Note-level metrics are `mir_eval` note P/R/F (onset ±50 ms, pitch ±50 cents,
offsets ignored), scored on the Slakh validation split. All conditions share the
same reference notes and val tracks, so rows are directly comparable.

## The comparison that matters

Each row swaps exactly **one** component of the segmenter, so the deltas isolate
each learned head's contribution over the DSP floor. Pitch is always pyin.

| segmenter | onsets | voicing | stem F1 | demucs F1 |
|---|---|---|---|---|
| DSP floor (`HeuristicSegmenter`, δ=0.20) | spectral flux | pyin | **0.750** | **0.637** |
| onset head (`tcn-v1`) | learned | pyin | **0.813** | **0.686** |
| onset + voicing head (`tcn-v2`) | learned | learned | **0.878** | **0.729** |

Full P/R breakdown, matched val sets (stem N=252, demucs N=213), floor at δ=0.20:

| set | segmenter | P | R | F1 |
|---|---|---|---|---|
| stem | DSP floor | 0.713 | 0.792 | 0.750 |
| stem | onset head | 0.768 | 0.865 | 0.813 |
| stem | onset + voicing | 0.855 | 0.903 | **0.878** |
| demucs | DSP floor | 0.624 | 0.651 | 0.637 |
| demucs | onset head | 0.649 | 0.727 | 0.686 |
| demucs | onset + voicing | 0.708 | 0.751 | **0.729** |

**Every step is Pareto** (precision *and* recall rise, both conditions) — real
model gains, not threshold trades:
- **Onset** buys recall (splits same-pitch re-articulations the floor merged):
  stem R +7.3, demucs R +7.6.
- **Voicing** buys precision (trims pyin's over-voiced rests, the 34% FP bucket):
  stem P +8.7, demucs P +5.9.
- Cumulative over the floor: **stem +12.8, demucs +9.2**. demucs (the deployment
  condition) is the number that ships.
The demucs floor uses the stem-tuned δ=0.20; demucs's own optimum is milder
(~0.13, ~0.647), so even vs a demucs-tuned floor the head stack wins clearly.

## Runs

### tcn-v1 (onset head) — 2026-07-22
- **Model:** `TCNEncoder` (~1M params: 32-ch conv frontend, hidden=128, 4 dilated
  levels, RF ~0.7 s) + single `onset_head`. BCE-with-logits, Adam lr=1e-3, window
  256 frames (~3 s), batch 16.
- **Data:** full Slakh, domain-randomised input (stem/mix/demucs, ~3290
  track-examples), targets always clean-stem MIDI. Validation on all three
  conditions separately.
- **Throughput:** ~24 min/epoch on M3 Max (MPS), ~50k steps/epoch.
- **Result — note-F1 saturates at epoch 1:**

  | epoch | val loss | note_f1 stem | demucs | mix |
  |---|---|---|---|---|
  | 1 | 0.0179 | 0.813 | 0.686 | 0.175 |
  | 3 | 0.0166 | 0.813 | 0.684 | 0.176 |

- **Key finding — loss/F1 decoupling:** val *loss* keeps dropping (0.0198 → 0.0169
  → 0.0166) while note-*F1* is flat from epoch 1. The onset probs sharpen (lower
  BCE) but peak-pick + grouping produce the same notes. Implication: onset-model
  scaling won't move note-F1; the ceiling is downstream (voicing/pitch from pyin).
  Future onset runs can use `--epochs 2`.
- **mix ≈ 0.18 confirms separation is non-negotiable** — bass buried in the full
  mix is untrackable; demucs recovers 84% of stem F1.

### tcn-v2 (onset + voicing heads) — 2026-07-22
- **Model:** same `TCNEncoder`, now two heads (onset + voicing) on the shared
  encoder; per-head weighted BCE. `--epochs 4 --val-every 1`.
- **Result (best = epoch 3; plateaus at epoch 0):** stem **0.878** (P .855/R .903),
  demucs **0.729** (P .708/R .751), mix 0.269. Pareto over onset-only on both.
- **Voicing head is near-solved.** Threshold-free voicing AUROC: stem **0.984**,
  demucs **0.961**, mix 0.937. Frame-F1-optimal threshold ~0.52 everywhere, so the
  default 0.5 cut is right — no tuning warranted. Frame voicing-F1 ≈ 0.95–0.97.
- **Mechanism confirmed:** voicing lifted *precision* (stem P +8.7, demucs +5.9) by
  trimming pyin's over-voiced rests — exactly the 34% FP bucket predicted below.
- Checkpoint saved to `data/cache/onset/onset_model.pt` (holds both heads; this
  overwrote the onset-only tcn-v1 file — name future outputs per-run to keep both).

## Error decomposition

The pre-onset decomposition of note-level false positives was **58% over-seg / 34%
voicing / 8% pitch**. As of tcn-v2 the first two are addressed: onset head took the
over-seg (recall), voicing head took the 34% (precision, near-solved). **The
residual is now onset over-segmentation + pyin pitch/octave errors** — pitch is the
untouched third component and the candidate next lever, gated on the octave-error
analysis (`scratchpad/octave_analysis.py`) that sizes how much of the residual a
learned pitch head (dethroning pyin, Onsets-&-Frames-style) would actually recover.

## Deployment gap

Stem → demucs costs every stage ~15 F1 (floor −11.3, onset −12.7, onset+voicing
−14.9), so the learned lead survives separation but the gap does not shrink.
demucs is the number that ships (we always separate before tracking): **0.729** as
of tcn-v2, up from the 0.637 DSP floor.

## Hardware / ops notes
- M3 Max, 36 GB RAM. Long runs need `caffeinate -is` (sleep kills them silently).
- Full-scale fits in RAM only via memmapped CQT features + lazy val features.
- MLflow UI: `mlflow ui --backend-store-uri file:./data/mlruns --port 5001`.
- DSP floors reproduce in seconds from cached pyin contours:
  `scratchpad/dsp_floor_val.py`, `scratchpad/dsp_floor_demucs.py`.
