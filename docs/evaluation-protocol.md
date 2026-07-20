# Evaluation protocol

Exactly how the numbers in this repo are computed, written to be comparable
against someone else's implementation. If two RPA figures disagree, the cause is
far more often a protocol difference than a model difference — see
[what to check first](#what-to-check-first).

Implementation: [`stemify/evaluation/`](../src/stemify/evaluation/). Everything
below is `mir_eval.melody` at its documented defaults unless stated.

## 1. Representation

A pitch track is `(times, freqs)`, both 1-D and the same length, frequencies in
Hz, with **`0.0` meaning unvoiced**. There is no separate voicing array —
`0.0` *is* the voicing signal. This is the `mir_eval` convention.

The same type describes references and estimates
([`Contour`](../src/stemify/contour.py)).

## 2. Time grids

**The reference's grid wins.** The estimate is resampled onto it, never the
reverse. Rationale: the reference decides where the question "what pitch is
sounding?" gets asked; resampling the reference down to a coarse estimate grid
would let a baseline improve its score by predicting less often.

| source | grid | origin |
|---|---|---|
| MDB-bass-synth reference | 128/44100 s ≈ **2.9 ms** | the dataset's annotation |
| Slakh reference | **10 ms** | our choice — MIDI is event-based, has no grid |
| IDMT reference | **10 ms** | our choice |
| pyin estimate | 512/22050 s ≈ **23.2 ms** | librosa default, pinned |
| PESTO estimate | **10 ms** | its `step_size` default |

So on MDB a pyin estimate is upsampled roughly 8×. That is standard practice,
but it does mean **frame counts are set by the reference**, and a per-frame
count is not comparable across datasets with different grids.

## 3. Reference construction

Differs per dataset, and this is where label quality varies most:

- **MDB-bass-synth** — read directly from the annotation CSV. Voicing is
  annotated ground truth.
- **Slakh** — derived from the MIDI that rendered the audio. Each note fills
  frames from onset to offset; overlapping notes collapse to the **lowest**
  sounding pitch. Stems above 5% polyphony are excluded entirely.
- **IDMT-SMT-Bass** — the filename's labelled pitch, held wherever the clip has
  energy. **Voicing is inferred, not annotated**: an RMS gate at 40 dB below the
  clip's peak. This is the weakest reference here; the decay tail gets cut at an
  arbitrary level.

Items whose reference is entirely silent are **skipped**, not scored as failures.

## 4. Alignment

```python
est_freqs, est_voicing = mir_eval.melody.resample_melody_series(
    times=estimate.times,
    frequencies=abs(estimate.freqs),
    voicing=(estimate.freqs > 0).astype(float),
    times_new=reference.times,          # reference grid
    kind="linear",                      # mir_eval default
)
pred = where(est_voicing > 0.5, est_freqs, 0.0)
```

Voicing is carried **separately** through the resample and re-thresholded at
`0.5` afterwards. Interpolating Hz straight through an unvoiced gap would invent
a glide between two unrelated notes.

## 5. Metrics

Frequencies convert to cents via `mir_eval.melody.hz2cents`, **base frequency
10 Hz** (the default). Tolerance is **50 cents** — half a semitone, so tolerance
bands never overlap.

Let `V` = reference-voiced frames, `U` = reference-unvoiced frames, `N = V + U`.

| metric | definition | denominator |
|---|---|---|
| **RPA** | est voiced **and** \|Δcents\| < 50 | `V` |
| **RCA** | as RPA, octaves folded out (Δ mod 1200) | `V` |
| **Overall** | (both voiced and within tol) **or** (both unvoiced) | `N` |
| voicing recall | est voiced, given ref voiced | `V` |
| voicing false alarm | est voiced, given ref **un**voiced | `U` |
| median cents | median \|Δcents\| after octave folding | frames both voiced *and* chroma-correct |

Two consequences that trip people up:

- **An unvoiced estimate at a reference-voiced frame counts as wrong for RPA**,
  identically to a wrong pitch. Verified directly. A model that declines to
  answer on half the frames caps at 50% RPA regardless of its pitch quality.
- **RPA and RCA cannot see anything that happens during rests.** A model
  hallucinating a pitch through every silence scores *identical* RPA and RCA to
  one that correctly stays quiet. Only Overall and the false-alarm rate see it.

## 6. Aggregation

**Frame-pooled (micro-average) across the entire dataset.** All frames from all
items are concatenated into one pair of arrays, and each metric is computed once
over the pool.

This is *not* the mean of per-item scores. A six-minute track contributes ~12×
the frames of a thirty-second one and is weighted accordingly. Macro-averaging
instead — the more common accidental choice — produces a different number, often
by several points, and is a prime suspect when two implementations disagree.

Per-group breakdowns (per string, per technique, per track) are the same metric
computed on a boolean-masked subset of the same pool.

## 7. Baselines that emit one pitch per clip

A baseline implementing only `predict_pitch` (one MIDI number for a whole clip)
is expanded into a contour holding that value across the reference's voiced
frames, then scored identically. No separate note-level path.

## What to check first

Ordered by how much damage each does, when comparing against another
implementation:

1. **Aggregation** — micro (pooled frames) vs macro (mean of per-track scores).
   Easily several points.
2. **Voicing convention** — does an unvoiced prediction count as *wrong*, or is
   it *excluded from the denominator*? Excluding it inflates the score
   substantially for any model that declines to answer.
3. **Cent tolerance** — measured sensitivity on real data, same pyin run scored
   at different tolerances:

   | tolerance | RPA |
   |---|---|
   | 10 c | 66.4% |
   | 25 c | 82.8% |
   | **50 c** | **89.9%** |
   | 100 c | 94.7% |
   | 200 c | 97.0% |

   Loosening 50 → 100 is worth **+4.8 points**. Always state the tolerance.
4. **Discrete classifier vs continuous estimator.** Measured rather than
   assumed, because the intuition here is misleading. Deviation from the
   *nearest* semitone is bounded at 50 cents by definition, so **at our 50-cent
   tolerance a semitone classifier is not handicapped at all** — picking the
   right semitone always passes.

   It does bite at tighter tolerances and on precision. Across 971,529
   MDB reference frames, real bass sits a median **13.6 cents** off the semitone
   grid, and **27.7% of frames are more than 25 cents off** — all fine at 50 c,
   all failures at 25 c. So a discrete classifier's median-cents figure will be
   roughly 13-14 against a continuous tracker's ~6, without any RPA difference
   at the default tolerance.
5. **Denominator** — reference-voiced frames only (RPA/RCA) vs all frames
   (Overall). Reporting Overall as "accuracy" against someone's RPA compares
   different quantities.
6. **Time grid and resampling** — whose grid, and whether voicing was carried
   separately through the interpolation.
7. **Silence representation** — `0.0 Hz` here; others use `NaN` or a separate
   boolean array. A silent frame read as a *0 Hz pitch* rather than as *unvoiced*
   will be counted as a wildly wrong pitch instead of a correct rest.

## Sanity check for a second implementation

The fastest way to confirm two setups agree is to run the *same baseline* on the
*same data*. pyin is a good instrument for this: no training, no randomness, and
librosa is widely available.

Any baseline can be scored through this harness by implementing
[`PitchBaseline`](../src/stemify/baselines/base.py) — either `predict_f0`
(frame-level contour) or `predict_pitch` (one pitch per clip). That is the whole
interface, and it makes an external model's numbers directly comparable to
everything reported here.
