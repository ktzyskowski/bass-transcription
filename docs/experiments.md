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

## Semi-CRF decoder — Config A findings (frozen probe, 2026-07-23)

A semi-CRF learned decoder to replace `group()` (design: `docs/semi-crf.md`),
probed as **Config A**: frozen tcn-v2 emissions, ~15 CRF params fit by NLL of the
gold segmentation via the differentiable segmental forward algorithm; batched DP
(START-state 3-label scan). Prototype: session scratchpad `semicrf.py`.

**Three failure causes found, two fixed on the frozen probe:**

1. Missing pitch-change cue → added a pyin pitch-jump emission (dropout-bridged,
   like group()'s running pitch).
2. **MIDI-gap gold** → gold rests derived from MIDI note-off gaps that the audio
   rings through (see note-off-vs-silence quirk) trained the CRF that `note→note`
   was unlikely, collapsing legato runs. Fix: merge sub-threshold (<0.12 s) MIDI
   gaps into contiguous `note→note` gold (extend the note offset; onsets never
   move — the metric ignores offsets). **Confirmed the cause:** `trans[note→note]`
   −1.34 → −0.59, NLL 1620 → 1339, **stem recall 0.35 → 0.55 and stem precision
   0.841 now beats group()'s 0.826.**
3. **Weak demucs emissions** → NOT fixable while frozen. The gold fix left demucs
   recall flat (0.24 → 0.24): on separated audio the onset head is sub-threshold and
   pyin is noisy, so the split cues aren't in the frozen emissions.

**Verdict:** the architecture is *validated on clean signal* but hits the
**frozen-emission ceiling** on demucs. As a frozen decoder it still loses to group()
on recall (stem 0.55 vs 0.88), so it does not ship. group() remains the shipping
baseline. The demucs gap is an emission-quality problem → the fix is **end-to-end**.

### Stage 1 POC — end-to-end CRF, first run (2026-07-29, IN FLIGHT)

The semi-CRF had never been trained end to end: everything existed
(`SemiCRFRecipe`, `SemiCRFLoss`, `SequenceDataset`, `nll_batch`) but `train.py`
hardcoded the frame objective. `--objective {frame,semicrf}` wired in 4e5639b.

Deliberately small, to answer "does it learn at all" before any scale work:
40 stem tracks, **from scratch** (no tcn-v2 checkpoint exists on the 5090 box — it
stayed on the M3), whole tracks at B=2, 10 epochs, val 20 stem tracks.
**Not comparable to the 0.729 rows above**, which used the full corpus and three
sources. The comparison that counts is a `--objective frame` run on the same 40.

First three epochs, ~3.5 min each:

| epoch | train NLL | val NLL | note F1 | P | R | onset AP |
|---|---|---|---|---|---|---|
| 0 | 5025.7 | 2528.3 | 0.171 | 0.450 | 0.106 | 0.380 |
| 1 | 3310.5 | 2520.5 | 0.146 | 0.412 | 0.088 | 0.439 |
| 2 | 3383.9 | 2334.9 | — | — | — | — |

**Early read (3 of 10 epochs — do not treat as a verdict).** The objective trains:
NLL falls and `onset_ap` *rises* (0.380 → 0.439), so the encoder's cues are getting
better. But note F1 **falls** (0.171 → 0.146), driven by recall (0.106 → 0.088).

Cues improving while notes get worse points at the decoder, not the emissions — and
recall ~0.09 against `group()`'s 0.90 on stems is the **same recall-collapse signature
Config A showed** (0.55 there). The end-to-end thesis was that joint training would
fix that by sharpening emissions; three epochs of a from-scratch encoder is far too
early to conclude it has not, but it is not yet showing signs of doing so.

## Future experiments

### Priority order — revised 2026-07-29

The staging below was written CRF-first, on the reasoning that Config A had validated
the architecture and that three simultaneous changes would be undebuggable. Both still
hold. But re-reading the error decomposition against what each lever can actually
reach changes the ordering, so the revised priority is:

| # | lever | reaches | cost | status |
|---|---|---|---|---|
| 0 | pitch-error sizing analysis | — (measurement) | hours, no training | not started |
| 1 | bichannel (bass, mix) input | demucs emission quality | small | not started |
| 2 | learned pitch head | pyin pitch/octave errors | large | gated on #0 |
| 3 | semi-CRF end-to-end | onset over-segmentation | built | POC running |

**Why the reorder.** The three levers are *orthogonal* — they change the encoder's
input, where pitch comes from, and how cues become notes respectively — so they
compose rather than compete, and any of them can be run on top of the others. What
differs is the error mass each can reach:

- **The semi-CRF changes only how cues become notes.** Pitch is read off the pyin
  contour by `HeuristicDecoder` and `SemiCRFDecoder` alike; the CRF never invents it.
  So its reachable mass is the over-segmentation bucket alone — and per the error
  decomposition above, the residual after tcn-v2 is *over-segmentation **plus** pyin
  pitch/octave errors*. It is powerless on the second half. Config A's measured upside
  was precision 0.841 vs `group()`'s 0.826, i.e. **~1.5 points**, conditional on
  clawing back a 33-point recall deficit.
- **Bichannel is cheaper than the CRF was and aims at a bigger gap.** The stem→demucs
  drop is ~15 F1 at every stage, the largest single identified loss.
  `_ConvFrontend` already takes a channel count, so `(B,1,F,T) -> (B,2,F,T)` is nearly
  a one-liner; **both channels' CQTs are already cached** (`SOURCES` builds stem, mix
  and demucs-separated bass, all content-addressed); and it runs on the proven frame
  BCE + heuristic path, needing none of the semi-CRF training machinery. It is a
  dataset change plus a channel dimension.
- **Size the pitch lever before building it.** #0 is nearly free and needs no training:
  cached pyin contours plus reference notes are enough to measure offline what fraction
  of note errors are *pitch-only* -- correct onset, wrong pitch. That replaces the
  missing `scratchpad/octave_analysis.py` and decides whether #2 is worth the build.

**The caveat that keeps #1 honest:** the drum-bleed hypothesis is unverified. We do not
know how much of the demucs gap is bleed the mix channel can explain away versus
transients demucs simply destroys. The bichannel run *is* the measurement.

### Semi-CRF end-to-end + mix-conditioned input — committed next (needs 5090)

Retrain the onset/voicing heads **jointly with the CRF** so the separated-audio
emissions get sharpened for the decoder — the only thing that addresses the demucs
ceiling above. The differentiable forward-algorithm scan moves *into* the training
loop (backprop through heads + CRF), which is sequential and MPS-hostile → run on
CUDA (5090). Bake in the two proven Config-A fixes: **dropout-bridged pitch cue +
audio-aligned gold**. The note-verifier (learn to prune group()'s FPs) is **dropped**
— it was the fallback for when the CRF looked dead; the CRF is now the real path, and
group() holds the shipping line meanwhile.

Stage it so each result is interpretable (three simultaneous changes would be
undebuggable):

**Stage 1 — end-to-end CRF, current (bass) input.** Answers the one big unknown:
does joint training lift demucs recall past the frozen ceiling? Build work: promote
the prototype from frozen cached probs to an integrated loss (segmental forward NLL
as the training objective, decoding via segmental Viterbi), reusing the batched DP.

**Stage 2 — add (bass, mix) bichannel input, as an ablation.** Feed the separated
bass *and* the original mix (bass head stays bass-only; mix is pure context).
Rationale: a chunk of demucs over-seg is likely **drum bleed** (kick/bass overlap in
the low end), read as spurious bass onsets — given the mix, the model can learn "this
low transient lines up with a kick → not a re-pluck → suppress it." Training with vs
without the mix channel *measures* the bleed contribution directly (replaces the
standalone drum-onset-coincidence diagnostic, since we're retraining anyway).

- **Buildable:** demucs already emits the mix/all stems in one pass, available *at
  inference*, not just training — we currently discard everything but bass. Slakh
  gives ground-truth stems for training.
- **Channels:** start **(bass, mix)** — the mix is `bass + residual`, so the model
  derives the interferers (drums included) as `(mix − bass)`; a separate drum stem is
  redundant (drums ⊂ mix). Keep **(bass, drums, mix)** only as a later ablation if
  2-channel underperforms.
- **Mechanics:** `_ConvFrontend` already takes channels, so `(B,1,F,T)` →
  `(B,2,F,T)`. Train with **stem dropout** (randomly zero the mix channel) so an
  isolated-bass-DI upload with no mix degrades to bass-only instead of breaking.

## Hardware / ops notes
- M3 Max, 36 GB RAM. Long runs need `caffeinate -is` (sleep kills them silently).
- Full-scale fits in RAM only via memmapped CQT features + lazy val features.
- MLflow UI: `mlflow ui --backend-store-uri file:./data/mlruns --port 5001`.
- DSP floors reproduce in seconds from cached pyin contours:
  `scratchpad/dsp_floor_val.py`, `scratchpad/dsp_floor_demucs.py`.
