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
| onset + voicing head | learned | learned | _(in progress)_ | _(in progress)_ |

Full P/R breakdown, matched val sets (stem N=252, demucs N=213), all at δ=0.20 for
the floor:

| set | segmenter | P | R | F1 |
|---|---|---|---|---|
| stem | DSP floor | 0.713 | 0.792 | 0.750 |
| stem | onset head (e1=e3) | 0.768 | 0.865 | **0.813** |
| demucs | DSP floor | 0.624 | 0.651 | 0.637 |
| demucs | onset head (e1) | 0.649 | 0.727 | **0.686** |

**Learned onset head beats the DSP floor Pareto** (precision *and* recall up) on
both conditions: **stem +6.3 F1, demucs +4.9 F1**. Not a threshold trade — a real
model gain. The demucs floor uses the stem-tuned δ=0.20; demucs's own optimum is
milder (~0.13, ~0.647), so even vs a demucs-tuned floor the head wins ~+3.9.

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

## Error decomposition (why voicing is next)

Note-level false positives break down as **58% over-segmentation / 34% voicing /
8% pitch**. Onset head addresses the 58% (and beat DSP on it). **Voicing (34%) is
the next-biggest lever**, orthogonal to onset, ~zero new infra. Pitch (8%) is
small but tab is octave-sensitive — gate behind an octave-error check before
building. Direction of the voicing win: the 34% are *false positives* (pyin
over-voices — places notes in rests), so a learned voicing mask should trim them
and lift precision.

## Deployment gap

Stem → demucs costs both methods ~similarly (onset head −12.7 F1, DSP −11.3), so
the learned lead survives separation. demucs (~0.686) is the number that ships,
since we always separate before tracking.

## Hardware / ops notes
- M3 Max, 36 GB RAM. Long runs need `caffeinate -is` (sleep kills them silently).
- Full-scale fits in RAM only via memmapped CQT features + lazy val features.
- MLflow UI: `mlflow ui --backend-store-uri file:./data/mlruns --port 5001`.
- DSP floors reproduce in seconds from cached pyin contours:
  `scratchpad/dsp_floor_val.py`, `scratchpad/dsp_floor_demucs.py`.
