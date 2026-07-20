# Modelling notes

Decisions and candidates for the learned pitch stage. Nothing here is built yet —
this records what the measurements imply so the reasoning is not re-derived.

## What the measurements say to prioritise

pyin on MDB-bass-synth, 71 tracks, identical labels across conditions:

| | stem | mix | Demucs(mix) |
|---|---|---|---|
| RPA | 88.6% | 38.0% | 81.8% |
| Overall | 89.9% | 39.8% | 73.8% |
| voicing recall | 96.8% | 61.8% | 93.7% |
| voicing false alarm | 6.7% | 55.3% | 47.4% |
| octave gap | 0.3% | 5.4% | 1.1% |

Read in priority order:

1. **Voicing is the dominant failure, not pitch.** Recall collapses 96.8% → 61.8%
   on mixes. Separation restores it (93.7%) but leaves false alarms almost
   untouched (47.4% against a 6.7% ceiling) — Demucs emits bass-like residue
   through rests. **A voicing head is worth more than a better pitch head.**
2. **Octave errors are not the problem**, despite the framing in
   docs/pitch-primer.md. Measured at 0.1–5.4% everywhere, ~1% post-separation.
   Do not design around them without re-measuring.
3. **Headroom on clean stems is 11 points** (88.6%, not 100%). That is
   separation-independent and the honest target for a learned model.

## Output head: two-hot over cents bins, cross-entropy

**Decision: uniform bins in cents, two-hot targets, CE loss. No symlog.**

Rationale:

- **Multimodality is the real argument.** Pitch posteriors are genuinely bimodal
  — "E1 or E2?" when the fundamental is weak. An MSE regression head predicts
  the *mean* of the two modes, which is the one answer guaranteed to be wrong.
  A categorical distribution can say "60% E1, 40% E2" honestly. This is why
  CREPE and PESTO classify over bins rather than regressing.
- **Two-hot recovers the resolution plain classification gives up.** A target at
  43.7 cents becomes a proportional mixture of the 40 c and 60 c bins; decode by
  expectation over the softmax for a continuous estimate.
- **No symlog.** It exists in DreamerV3 for unbounded, wildly-scaled rewards.
  Pitch is bounded (bass ≈ MIDI 28–67) and **cents is already the log
  transform**. Symlog on cents double-compresses a bounded quantity; symlog on
  Hz just reinvents cents worse.
- **Prior art:** this is close to CREPE's encoding (360 bins at 20 cents,
  Gaussian-blurred targets, CE). Two-hot is a sharper two-bin variant. Lean on
  those hyperparameters rather than inventing.

Sizing: bass spans ~3900 cents (MIDI 28–67). At 20-cent bins that is ~195
classes. Note the range **must exceed 28 semitone classes** — Slakh bass MIDI
reaches 67, and `PitchConfusion` currently ceilings at 56.

Two bonuses:

- The per-frame distribution feeds **Viterbi/HMM decoding** for temporal
  smoothing — something pyin does internally and a point-estimate regressor
  cannot support, since there is no uncertainty to decode over.
- It represents **expression** (vibrato, bends, slides) as pitch movement within
  a note, which a semitone classifier structurally cannot.

Calibration: this addresses precision, which is *secondary*. Measured, a
semitone classifier is not even handicapped at our 50-cent tolerance (deviation
from the nearest semitone is bounded at 50 c by construction). Real bass sits a
median 13.6 cents off-grid with 27.7% of frames beyond 25 cents, so the win
shows up in median-cents and at tighter tolerances, not in RPA at 50 c.

## Architecture candidates, unresolved

- **CQT + CNN, frame-wise.** The cheap control. `features.py` already has the
  front-end (3 bins/semitone, 6 octaves from D#1). Establishes whether *any*
  learned model beats pyin before spending on pretrained representations.
- **Frozen MERT + PESTO with a small head.** Proposed for octave correction via
  musical context — but the measurements put octave error at ~1% post-separation,
  so that rationale is largely gone. If it earns its place it will be for knowing
  what bass *sounds like* well enough to reject the residue Demucs leaves in
  rests, which is closer to a separation objective than a pitch one.
- **Fine-tuning PESTO on bass.** Cheapest large win available: its architecture
  suits the task, its default checkpoint is vocal-trained and floors near 50 Hz,
  below much of the bass register. Fixes the range at the source and serves as
  a clean ablation baseline for anything fancier.

## Training data

See docs/datasets.md. Short version: Slakh for training (only source with volume
and a clean licence), MDB-bass-synth as held-out evaluation **never training**
(resynthesis artifacts a network could learn), IDMT for sanity checks.

Train on **all three input conditions** — clean stem, Demucs stem, raw mix.
Slakh supplies all three with identical labels, so it is free domain
randomisation, and it de-risks the separation decision rather than betting on it.