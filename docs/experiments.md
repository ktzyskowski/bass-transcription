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

### Stage 1 POC — end-to-end CRF, first run (2026-07-29) — ENCOURAGING

The semi-CRF had never been trained end to end: everything existed
(`SemiCRFRecipe`, `SemiCRFLoss`, `SequenceDataset`, `nll_batch`) but `train.py`
hardcoded the frame objective. `--objective {frame,semicrf}` wired in 4e5639b.

Deliberately small, to answer "does it learn at all" before any scale work:
40 stem tracks, **from scratch** (no tcn-v2 checkpoint exists on the 5090 box — it
stayed on the M3), whole tracks at B=2, 10 epochs, val 20 stem tracks, ~3.5 min/epoch.

| epoch | train NLL | val NLL | note F1 | P | R | frag | onset AP |
|---|---|---|---|---|---|---|---|
| 0 | 5025.7 | 2528.3 | 0.171 | 0.450 | 0.106 | 4 | 0.380 |
| 1 | 3310.5 | 2520.5 | 0.146 | 0.412 | 0.088 | 10 | 0.439 |
| 2 | 3383.9 | 2334.9 | 0.211 | 0.512 | 0.133 | 11 | 0.399 |
| 3 | 2909.9 | 1714.6 | 0.853 | 0.895 | 0.815 | 206 | 0.409 |
| 4 | 2353.6 | 1658.9 | 0.704 | 0.849 | 0.601 | 52 | 0.383 |
| 5 | 2028.1 | 1528.3 | 0.398 | 0.690 | 0.280 | 5 | 0.440 |
| 6 | 2023.0 | 1318.2 | 0.860 | 0.871 | 0.849 | 381 | 0.456 |
| 7 | 2302.3 | 1514.0 | 0.858 | 0.838 | 0.878 | 604 | 0.403 |
| 8 | 1915.3 | 1293.3 | **0.862** | 0.877 | 0.848 | 345 | 0.474 |
| 9 | 2054.6 | 1349.0 | — | — | — | — | — |

**Stem F1 0.862 from 40 tracks and 40 minutes**, against tcn-v2's 0.878 from 1260
tracks x 3 sources. Recall 0.848 — no collapse.

**Correction to an earlier read in this file.** At 3 epochs this was recorded as
showing "the same recall-collapse signature Config A showed" (recall 0.09 vs
`group()`'s 0.90). That was wrong: the collapse is a *transient early-training phase*
and it broke out at epoch 3, recall 0.13 -> 0.82 in one epoch. Do not read the first
three epochs of a from-scratch CRF as the failure mode.

The striking part is that F1 0.862 is reached on **weak cues** — `onset_ap` only 0.474,
`onset_best_f1` 0.569. If that holds up it is the argument *for* the CRF: it is
extracting good notes from emissions the heuristic would struggle with. It also
partly answers the Config A diagnosis, since joint training did lift the decoder past
what frozen emissions allowed.

**Not yet established, and this is not a verdict:**

- **No baseline.** A `--objective frame` run on the same 40 tracks / same 20 val
  tracks was never run (launched, then cancelled). Without it 0.862 is uninterpretable
  — a 20-track val subset may simply be easier than the 252-track one behind 0.878.
- **Unstable.** F1 swings 0.85 -> 0.70 -> 0.40 -> 0.86 across consecutive epochs, and
  val NLL was still falling at epoch 8. Ten epochs from scratch is not converged.
- **Confounded step count.** Frame BCE at B=16 over 256-frame windows gets ~412
  gradient steps/epoch; the semi-CRF at B=2 over whole tracks gets 20. Same "epochs"
  is not the same amount of learning, and any comparison has to say which it controls.
- **Small val.** 20 tracks vs the 252 behind every other row in this file.

#### Frame baseline on the same 40 tracks (2026-07-30) — the semi-CRF loses

| run | best F1 | P | R | frag | onset AP |
|---|---|---|---|---|---|
| **frame** (ep 1) | **0.9449** | 0.9279 | 0.9625 | 163 | 0.9302 |
| frame, independent rerun (ep 1) | 0.9388 | 0.9184 | 0.9601 | 173 | 0.9419 |
| **semicrf, cold** (ep 8) | 0.8623 | 0.8767 | 0.8483 | 345 | 0.4737 |
| **semicrf, warm-started** (ep 0) | 0.8317 | 0.8704 | 0.7964 | 277 | 0.5561 |

Frame BCE + `group()` wins by **8.3 F1**. Two frame runs agree to 0.6 points (the
second is the run cancelled mid-session, which had already finished 9 of 10 epochs at
~15 s each), so this is not a seed fluke.

> **Revisited later the same day — this comparison does not settle anything, and the
> "degrades the encoder" reading below is overstated.** "10 epochs" is not equal work:
> frame ran 333 batches/epoch (**3330** gradient steps), the semi-CRF 17 (**170**), a
> 20x gap. The semi-CRF was plainly still moving when it stopped:
>
> | epoch | 0 | 2 | 4 | 6 | 8 | 9 |
> |---|---|---|---|---|---|---|
> | val NLL | 2528 | 2335 | 1659 | 1318 | **1293** | 1349 |
> | note F1 | 0.171 | 0.211 | 0.704 | 0.860 | 0.862 | 0.753 |
> | onset AP | 0.380 | 0.399 | 0.383 | 0.456 | 0.474 | **0.518** |
>
> Val NLL roughly halved and was still falling; F1 still swung ±0.45 between
> consecutive epochs. And note the `onset AP` row: on the **cold** run it *rises*
> monotonically. The objective does not degrade cues from scratch — finding 1 below
> generalises from the warm-start run alone, where a pretrained encoder met a uniform
> `lr=1e-3` for 17 steps, which is at least as consistent with "the optimiser wrecked
> a good init" as with "the objective prefers bad cues".
>
> A fair test is **step-matched**. At ~120–220 s/epoch a 200-epoch whole-track run is
> ~12 h; the affordable routes are `ChunkedSequenceDataset` (excerpts: ~7x cheaper per
> step *and* ~7x more steps per epoch) or CUDA-graph capture of the DP (measured 22x).
> Both are built and parked.

#### Truncated BPTT, so the step-matched run is affordable (2026-07-30)

Chunked training is now wired (`--objective semicrf-tbptt`). Measured on the 37-track
`stem-solo` set:

| | whole-track | chunked (TBPTT) |
|---|---|---|
| gradient steps / epoch | 19 | **198** |
| seconds / epoch | 219 | **60** |
| **steps / second** | 0.087 | **3.3 (38x)** |
| batch size | 2 | 8 |
| padding waste | pads to the longest track (~40%) | none, static shapes |

So a step-matched run went from ~12 hours to ~80 minutes. The CUDA-graph capture of the
DP (measured 22x earlier) is **still parked and now lower priority** — chunking bought
the throughput more cheaply, and the two multiply if it is ever needed.

**Why this is the same objective, not an approximation.** The per-chunk losses
telescope: with `A_0 = 0` and the carry detached between chunks,
`Σ_k [(A_k − A_{k−1}) − S_k(gold)] == log Z − score(gold)` exactly, while each term's
gradient reaches only its own chunk's activations. `tests/training/test_tbptt.py`
asserts the summed per-chunk losses equal the whole-track NLL.

**A silent bug this uncovered.** `ChunkedSequenceDataset` had a `right_context` — frames
past a chunk's end so the TCN's 65-frame receptive field does not read zero-padding at
the edge — but `chunk_log_partition` scanned to the *full* window width, so any non-zero
margin left the carry `right` frames past where the next chunk resumed and those frames
were scanned twice. Every shape still matches, nothing raises, and the loss still falls;
only the log Z equality catches it. The default of 0 is why the existing carry tests were
green — they never exercised the parameter the dataset already had. There is now a
negative control pinning that scanning the margin *does* corrupt the result.

Three findings, in order of how much they matter:

1. **The structured objective degrades the encoder.** `--init-from` (added 2026-07-30)
   warm-starts the encoder from a frame checkpoint. Loaded at `onset_ap` **0.9634**;
   after *one* epoch of semi-CRF training it read **0.5561**. This reframes the cold
   run: 0.47 is roughly where this objective drives the encoder regardless of where it
   starts, so the cold run was not failing to learn good cues -- it was converging to
   the cue quality the segmental NLL actually prefers. Mechanically that tracks: the
   NLL only cares about *relative* scores between segmentations, and nothing in it
   rewards a calibrated per-frame onset probability. **Caveat:** `learning_rate` is
   1e-3 applied uniformly, and one epoch at full LR is a known way to wreck a
   pretrained encoder. A low-encoder-LR warm start is the one cheap test not yet run.
2. **Fragmentation is worse, which is the pointed one** -- killing over-segmentation is
   the CRF's entire purpose. Frame: 163 fragmentation FPs at R=0.96. semi-CRF: 345 at
   R=0.85. It finds *fewer* notes and produces *more* shredded-note false positives.
3. **Three regimes, three losses.** Config A (frozen good emissions) lost to `group()`;
   cold end-to-end loses; warm-started collapses the cues it was handed.

**The confound, stated plainly:** frame gets ~410 gradient steps/epoch against the
semi-CRF's 20 -- roughly **20x more updates** at equal "epochs". The semi-CRF's val NLL
was still falling at epoch 8 while the frame run had plateaued by epoch 2. This is not
a converged comparison, and it is not a clean architectural verdict.

Also note this val subset is *easier* than the real one -- frame scores 0.945 here
against tcn-v2's 0.878 on 252 tracks. Same subset for both, so the comparison holds;
the absolute numbers do not transfer to the table at the top of this file.

#### Step-matched result (2026-07-30) — it was not undertraining

The fair test the earlier comparison could not afford. Same 37 `stem-solo` tracks, same
19 val tracks, TBPTT at 198 steps/epoch; early-stopped at epoch 31 (patience 20, best
val NLL at epoch 11). **~6,300 gradient steps against the frame baseline's 3,330 —
nearly 2x, not 1/20th.**

| epoch | train NLL | val NLL | note F1 | P | R | onset AP |
|---|---|---|---|---|---|---|
| 4 | 13.47 | 11.55 | 0.8193 | 0.8884 | 0.7602 | 0.474 |
| 9 | 13.93 | 15.45 | 0.7980 | 0.8423 | 0.7581 | 0.452 |
| 14 | 8.61 | 14.63 | 0.7900 | 0.8316 | 0.7524 | 0.466 |
| 19 | 6.59 | 9.49 | **0.8733** | 0.8648 | 0.8819 | 0.462 |
| 24 | 2.70 | 8.86 | 0.7913 | 0.8110 | 0.7725 | 0.506 |
| 29 | 1.54 | 12.05 | 0.8403 | 0.8323 | 0.8484 | 0.528 |

**Best semi-CRF 0.8733 vs frame BCE 0.9375 on identical tracks — a 6.4 F1 gap that more
steps did not close.**

**The failure mode flipped, which is what makes this conclusive.** Train NLL fell 43.96
→ 1.54 while val NLL oscillated between 7 and 35: the model memorised 37 tracks. The
previous run stopped while still underfit; this one runs until it is thoroughly overfit,
and the best note F1 along the way (0.8733) is barely above the old undertrained
0.8623. There is no longer a "train it longer" story.

**What this does *not* establish.** 37 tracks is tiny, and the two objectives do not see
it the same way: the frame model draws ~90 overlapping 256-frame windows per track, so
each of its steps sees a different excerpt, while the semi-CRF walks each track as one
ordered sequence. The frame path has far more effective diversity per step, which is
exactly the asymmetry that would make a small-data comparison flatter the frame model.
So this rules out *undertraining*, not *data starvation*. **The open question is now
scale, not steps** — and it is a different experiment (more tracks), not a longer one.

`onset_ap` stayed ~0.47–0.53 throughout, against frame's 0.961. Worth noting the CRF
reads the onset logit as an unnormalised feature with a learned weight, so it is under
no pressure to produce calibrated per-frame onsets and a low AP is not by itself a
defect — but here the *note* F1 is worse too, so the decoder is not compensating.

#### Is the semi-CRF implementation actually correct?

Worth asking before trusting a negative result. As of 2026-07-30 the DP is **verified
against ground truth**, not just against itself:

- `test_log_partition_equals_brute_force_enumeration` -- enumerates every tiling of a
  short sequence explicitly and logsumexps them; matches the forward recurrence.
- `test_viterbi_equals_brute_force_argmax` -- returns the exact argmax segmentation.
- `test_batched_log_partition_equals_brute_force` -- same for the batched scan training
  actually runs.

This matters because every *other* DP test compares one of our implementations against
another (batched vs single, chunked vs whole, cpu vs cuda) -- checks that two copies of
the same wrong recurrence would pass. What remains unverified by ground truth is
`gold_segments`' audible-gap rule, but that is a labelling *policy* rather than a
recurrence: there is no ground truth for it beyond whether the rule is the right one.

#### On the `fragmentation` column

It is a **raw count of false-positive notes**, not a ratio: estimates whose onset
falls inside a reference note's span *and* carries that note's pitch — a real note the
segmenter shredded (`metrics/note/fragmentation.py`). So it scales with how many notes
are emitted at all, which is why the early epochs read low (4, 10, 11): the model was
barely emitting notes, not behaving well. Reading it as "over-segmentation got 30x
worse by epoch 6" would be backwards.

**Two caveats before using it as a diagnostic.** First, without the false-positive
total it cannot be turned into a share, so it is not comparable across runs with
different recall. Second — and worth fixing — `Fragmentation` splits false positives
into *three* buckets (`fragmentation` / `rest` / `interior_offpitch`), and that split
is precisely the "which head is worth building" diagnostic this file's error
decomposition was built on. `training/validation.py` surfaces only the first and
discards the other two.

**Done 2026-07-30** (commit `7eb78ff`): all four are logged now — `fragmentation`,
`fp_rest`, `fp_interior_offpitch` and the `fp` total. It paid for itself immediately;
the bichannel result below is invisible without it.

### Bichannel (bass, mix) input — 2026-07-30

Feed the encoder two stacked CQTs: the bass to transcribe in channel 0, the mixture it
came from in channel 1. 37 train tracks / 19 val tracks, 10 epochs, frame BCE +
`GreedyDecoder`, best-F1 epoch reported. Each pair is on an **identical track set**
(see "the control that was nearly missed" below).

| condition | ch | F1 | P | R | FP | frag | rest | offpitch | onset AP |
|---|---|---|---|---|---|---|---|---|---|
| `stem-solo` | 1 | 0.9375 | 0.9134 | 0.9628 | 565 | 196 | 268 | 101 | 0.9609 |
| `stem-mix` | 2 | **0.9419** | 0.9242 | 0.9602 | 487 | 206 | **179** | 102 | 0.9395 |
| `demucs` | 1 | 0.7103 | 0.6590 | 0.7704 | 2467 | 644 | 714 | 1109 | 0.7376 |
| `demucs-mix` | 2 | 0.7111 | 0.6608 | 0.7696 | 2444 | 538 | 744 | 1162 | **0.8503** |

`stem-solo` replicated across two independent runs to **0.0003** F1 (0.9372 / 0.9375),
so ~1-point differences are above seed noise even at this scale.

**On clean stems the mix channel does exactly what the hypothesis predicted.** F1
+0.44. More to the point, the only bucket that moved is `rest` — notes invented where
no bass sounds — down 33% (268 → 179), with `frag` and `offpitch` flat. That is the
drum-bleed prediction: the mixture tells the model whether a transient in the bass
belongs to the bass. Aggregate precision alone would have read as "+0.4, probably
noise".

**On separated audio it is a null on F1 (+0.0007) and a large win on cue quality
(onset AP 0.7376 → 0.8503, +11.3).** That dissociation is the finding. The mix channel
makes the encoder substantially better at saying *where the notes are* on damaged
audio, and almost none of it reaches note F1. The next section measures why.

**The model is genuinely reading channel 1**, so the null is not "it ignored the mix".
Zeroing channel 1 at inference (scratchpad ablation, best-val checkpoints):

| model | F1 as trained | F1 mix muted | delta | `fp_rest` muted |
|---|---|---|---|---|
| `stem-mix` | 0.9418 | 0.9281 | −0.014 | 103 (from 207) |
| `demucs-mix` | 0.7040 | **0.5358** | **−0.168** | 1622 (from 764) |

On separated audio the model *depends* on the mixture: without it, F1 collapses 17
points and invented notes more than double. Caveat: a zeroed channel is
out-of-distribution rather than "absent", so the magnitude overstates the dependence —
but the direction and the fact of it are not in doubt.

#### What actually caps note F1 — pitch or the decoder?

> **Replicated 2026-07-30 on independently retrained checkpoints** (6 epochs rather
> than 10, so the F1s differ slightly): `demucs-mix` 0.7006 → 0.7986 pitch-blind
> (+0.098 against +0.101 below), `stem-mix` 0.9444 → 0.9606 (+0.016 against +0.017).
> The ratio is what matters and it is stable: **pitch is ~1/3 of the remaining error,
> segmentation ~2/3**, on both conditions. The measurement is now a script rather than
> an ad-hoc scratch file — `scripts/pitch_ceiling.py`.

The FP buckets suggested pitch: `interior_offpitch` is 1109 of 2467 FPs on `demucs`
and grows 11x from `stem` while `rest` grows only 2.7x. **That reading was wrong**,
and it is worth recording why, because the bucket counts invite it. FP counts are not
F1 impact — they say nothing about recall — and `interior_offpitch` conflates a true
pitch error with an over-segmentation fragment whose short span gives pyin a bad pitch
estimate. An onset failure can present as an offpitch FP.

The direct measurement is to score the *same* estimated notes twice, once with the
50-cent pitch tolerance and once with pitch ignored (onset-only matching). The gap is
exactly what a perfect pitch head could buy; what remains below 1.0 is
onset/voicing/grouping error.

| condition | F1 | F1 pitch-blind | perfect pitch buys | still missing |
|---|---|---|---|---|
| `stem-solo` | 0.9342 | 0.9523 | +0.0181 | 0.0477 |
| `stem-mix` | 0.9418 | 0.9588 | +0.0169 | 0.0412 |
| `demucs` | 0.7103 | 0.8027 | +0.0924 | **0.1973** |
| `demucs-mix` | 0.7040 | 0.8046 | +0.1006 | **0.1954** |

**Onset/voicing/grouping is ~2x the pitch lever, on clean stems and separated audio
alike.** On `demucs`, 0.197 F1 is still missing with pitch already perfect, against
0.092 recoverable by fixing pitch entirely. Pitch-blind precision is **0.746** against
recall 0.873 — a quarter of emitted notes are spurious when pitch cannot be blamed —
so the residual is dominated by *spurious notes*, not missed ones and not pitch.

**And the decoder, not the emissions, is what wastes the bichannel gain.** Onset AP
rose 11.3 points; pitch-blind note F1 rose **0.24** points. Sweeping
`GreedyDecoder.onset_threshold` across 0.1–0.7 moves demucs pitch-blind F1 by less
than 0.005, with the 0.5 default already at optimum — so this is not a miscalibrated
threshold. Peak-pick + voicing-threshold + min-gap grouping simply cannot convert a
better-ordered onset curve into better notes.

**What this changes.** Bichannel is worth keeping — nearly free, and it strictly
improves the encoder. But the lever with the error mass behind it is the **cue→note
step**, not the pitch head: a note verifier over frozen cues has a ceiling of ~+0.13
pitch-blind F1 (perfect precision at unchanged recall 0.873 → 0.931), against +0.09
for perfect pitch. That supersedes the priority table below, which put the pitch head
at #2 and assumed bichannel would close the demucs gap.

**Caveats.** One seed per arm. 19 val tracks, so absolute numbers are not comparable
to the 252-track rows elsewhere in this file. `stem-mix` improved notes while its
onset AP *fell* (0.9609 → 0.9395), which is unexplained. And the checkpoints the
ablation used are best-*val-loss*, not best-F1, so its baseline differs slightly from
the table above.

#### The control that was nearly missed

`stem` keeps two-bass tracks — a stem is unambiguous whatever else sounds — while any
condition that feeds the mixture must filter them out, since one reference contour
cannot describe a mixture with two basses. At `--limit 40` that is **33 tracks against
37, sharing 26**. Comparing `stem` to `stem-mix` would have varied the input channels
*and* the corpus, and the F1 differences here are small enough that it would have
swamped the result. `stem-solo` exists as the matched single-channel control;
`demucs` needs no twin because it already filters the same way.

### Full corpus, rungs 0-2 — 2026-07-30 overnight

First runs on the pinned protocol (`docs/ablation-plan.md`): 1015 train tracks x 2
conditions, 40 val tracks, 8 epochs, B=32. ~36 min/arm.

| arm | ch | pos_weight | clean F1 | **separated F1** | separated AP |
|---|---|---|---|---|---|
| `rung0-mono` | 1 | — | 0.9317 | **0.7451** | 0.8839 |
| `balanced` | 2 | — | 0.9308 | **0.7431** | 0.9264 |
| `recall-w5` | 2 | onset=5 | 0.9325 | **0.7342** | 0.9218 |

*(the two 2-channel arms log under the pre-rename tags `stem-mix`/`demucs-mix`)*

**Rung 1 — bichannel does not help note F1, at any data scale.** 0.7431 against the
1-channel control's 0.7451: a 0.002 *deficit*, replicating the 37-track null (+0.0007)
on 27x the data with matched track sets. And it replicates the *mechanism* too — onset
AP rises (0.8839 -> 0.9264, **+4.3**) while note F1 does not move. Two independent
measurements now say the same thing: the mix channel genuinely improves the encoder, and
the cue->note step throws the gain away. **Bichannel can be dropped** — it costs an
input channel and buys nothing the decoder can use.

**Rung 2 — recall weighting does not break the recall ceiling.** Pitch-blind candidate
recall on `separated`:

| arm | thresh 0.05 | 0.20 | 0.50 | verifier ceiling |
|---|---|---|---|---|
| `rung0-mono` | 0.8932 | 0.8893 | 0.8786 | 0.9354 |
| `recall-w5` | 0.8998 | 0.9031 | 0.9032 | 0.9491 |

`pos_weight=5` buys **+2.5 points of recall at the default threshold** and essentially
nothing at a low one. **The ~0.90 recall ceiling now holds across three independent
axes** — 37 tracks vs 1015, unweighted vs weighted, threshold 0.05 vs 0.7. That makes it
structural rather than a tuning failure: ~10% of reference notes are not in the onset
curve, or are destroyed by grouping, and no loss weight or threshold recovers them.

**Data scale, same codebase at last.** 1-channel `separated`: 0.7103 at 37 tracks ->
0.7451 at 1015, i.e. **+3.5 F1 for 27x the data** (the earlier cross-era estimate was
+1.9). Larger than the archaeology suggested, still small.

**What this sets up for rung 3.** The verifier's ceiling is a pitch-blind F1 of ~0.949
against `recall-w5`'s current 0.824 — about **+12 points of pitch-blind headroom**, the
largest single lever measured so far. Use `recall-w5` as the generator: it has the best
candidate recall (0.903) and its worse *F1* is by construction, not a defect.

> **Superseded 2026-07-31.** Every note-level number above was decoded with
> `PITCH_SPLIT_SEMITONES = 0.8`, which was worth ~7.5 strict F1 on its own. They are
> comparable to each other and to nothing measured after the fix below. The verifier
> headroom in particular fell from +12 to +2.9/+4.5. See the next section.

## The decoder was throwing the results away (2026-07-31)

### Where the recall ceiling actually goes

`scripts/recall_ceiling.py`, `recall-w5` on `bichannel-separated`, 40 val tracks,
14238 reference notes, 1376 missed, pitch-blind matching.

**Not label noise.** Only 22 notes (0.2%) are inaudible in the clean stem; recall over
audible-only notes is 0.9046 against 0.9034 overall. The missing tenth is real notes.

| cause (earliest stage that explains it) | share of misses |
|---|---|
| separation stripped >10 dB, pyin then had nothing | 33% |
| absorbed into a neighbour (pyin's transition lag) | 24% |
| pyin failed on intact audio | 11% |
| onset head never fired | 11% |
| voicing head said rest | 9% |
| unexplained / matching contention | 11% |

**~69% of the ceiling is pyin**; only ~11% is the onset encoder that rungs 0–2 tuned.

### The mechanism: pyin's lag turned a good onset into a late note

The 24% bucket had a picked onset peak a median **6.6 ms** from truth, yet the nearest
estimated note sat **57.6 ms** away — 84% of them in a 50–75 ms band, i.e. just outside
the match tolerance. pyin (hop 512, 23.2 ms) reaches a note's pitch a median 23 ms
*after* the onset (37.5 ms mean on separated audio; past 50 ms for 14.2% of pitch
changes), so a real attack walks a staircase through intermediate pitches and every
step of it split. `group_contour` *had* placed a note correctly at the onset — the
fragment before the late pitch transition was ~2 contour frames, under `MIN_NOTE_S`,
so `emit` dropped it. One reference note, one FN **and** one interior-offpitch FP.

### Turning the pitch split off — strict note F1

| checkpoint | validation | test (held out) |
|---|---|---|
| `rung0-mono` | 0.7470 → 0.8213 | 0.7421 → **0.8172** |
| `recall-w5` | 0.7311 → 0.8110 | 0.7331 → **0.8073** |

Monotone across 0.8/1.0/1.25/1.5/2/2.5/3/6/off with no interior optimum. Wins on both
sides: precision 0.78 → 0.92, recall 0.878 → 0.886, fragmentation FPs down ~45%.

### Constant audit (`scripts/decoder_audit.py`, validation)

| constant | verdict |
|---|---|
| `pitch_split_semitones` | **removed** for the neural path; `DSP_PITCH_SPLIT_SEMITONES` keeps the DSP floor honest |
| `min_onset_gap_s` | inert for a trained head (0.0–0.05 score identically); kept for the DSP path |
| `min_note_s` | **earns its place** — removing costs −0.023 |
| `onset_threshold`, `voicing_threshold` | genuine operating points; best alternative +0.0012, inside noise. Unchanged at 0.5 |
| `merge_gap_s` | stays off. Found a latent coupling: it reused the *split* tolerance as its "same pitch" test, so disabling the split silently made it merge every adjacent note. Now `MERGE_PITCH_SEMITONES` |

### Rungs 0–2 re-scored under the fixed decoder (`scripts/rescore.py`)

| arm | ch | split | recall | prec | blind F1 | strict F1 | pitch gap |
|---|---|---|---|---|---|---|---|
| `rung0-mono` | 1 | validation | 0.8860 | 0.9244 | 0.9048 | 0.8213 | +0.0835 |
| `rung0-mono` | 1 | **test** | 0.8631 | 0.9353 | 0.8977 | **0.8172** | +0.0805 |
| `balanced` | 2 | validation | 0.9252 | 0.9250 | 0.9251 | 0.8252 | +0.0999 |
| `balanced` | 2 | **test** | 0.8962 | 0.9307 | 0.9131 | **0.8149** | +0.0982 |
| `recall-w5` | 2 | validation | 0.9279 | 0.8926 | 0.9099 | 0.8110 | +0.0989 |
| `recall-w5` | 2 | **test** | 0.9002 | 0.9055 | 0.9028 | **0.8073** | +0.0955 |

**This partly rehabilitates bichannel.** On strict F1 it is still a wash (`balanced`
0.8149 vs `rung0-mono` 0.8172 on test — within noise, and the sign flips on
validation). But the mix channel is now visibly buying *placement*: **+3.3 recall and
+1.5 pitch-blind F1 at equal precision**. That gain is cancelled by a pitch gap ~1.8
points larger, which is coherent — the extra notes it recovers are the quiet, hard ones
where pyin's pitch is least reliable.

So the earlier "drop bichannel" call should be read as **"bichannel is a null while
pyin is the pitch source"**. Its pitch-blind ceiling is higher; re-test it after the
pitch head lands rather than treating it as settled.

### What this leaves for F1

The pitch gap is now the largest measured lever: **+8.1 points** on test for
`rung0-mono`, ~+9.8 for the bichannel arms. A pitch head also attacks ~69% of the
recall ceiling, and lands pitch on the 11.6 ms CQT grid — which removes the 23.2 ms
contour quantisation that made a correctly-placed fragment fall under `MIN_NOTE_S` in
the first place. The note verifier's headroom fell to +2.9 (`rung0-mono`) / +4.5
(`recall-w5`) pitch-blind, and by construction it cannot raise recall at all.

## Rung 4: the learned pitch head (2026-07-31)

`rung4-pitch` — one variable from rung 0: `--pitch` added, otherwise identical
(`--sources clean,separated`, 1ch, 8 epochs, B=32, full corpus). Cost ~9.3 min/epoch,
**2–3× rung 0**: the branch adds four 32-channel 3×3 convs over the full (216, T)
plane, and the 2-D front-end was already dominant.

### It does not damage the other heads

The run's own validator predates the learned-contour swap (process started 14:13, the
swap landed 14:28), so it scored with the **pyin** contour throughout. That accident is
useful — it isolates the multi-task question:

| | separated F1 (val) |
|---|---|
| `rung0-mono` (re-scored) | 0.8213 |
| `rung4-pitch`, pyin contour | 0.8222 |

Unchanged. Adding a third head to the shared trunk costs the binary heads nothing.

### The A/B: learned contour vs pyin (`scripts/pitch_ab.py`)

One frozen checkpoint, one forward pass, only the contour source varies. Under the
retuned decoder below:

| condition | split | pitch | recall | prec | strict F1 |
|---|---|---|---|---|---|
| separated | validation | pyin | 0.9005 | 0.9217 | 0.8219 |
| separated | validation | **learned** | 0.9251 | 0.9078 | **0.8675** (+4.56) |
| separated | **test** | pyin | 0.8764 | 0.9326 | 0.8135 |
| separated | **test** | **learned** | 0.9386 | 0.9058 | **0.8638** (+5.03) |
| clean | test | pyin | 0.9940 | 0.9943 | 0.9582 |
| clean | test | learned | 0.9970 | 0.9878 | 0.9613 |

**The recall ceiling was pyin.** Candidate recall goes 0.876 → 0.939 on test. That
number had held at ~0.90 across three independent axes (37 vs 1015 tracks, unweighted
vs `pos_weight=5`, threshold 0.05 vs 0.7) and was called structural. It was — the
structure was pyin, and the size of the move matches the decomposition that put ~69% of
misses there. The pitch gap also narrows, 0.090 → 0.058.

**Clean does not regress** (+0.3), which was the standing risk: pyin is much stronger on
clean audio. Also resolves the earlier flag — clean ~0.98 is real, holding at 0.961 on
test; it is the pitch-split fix lifting clean the way it lifted separated.

**Correction to the harness's own claim.** `pitch_ab.py` reports pitch-blind F1 as a
guard, on the reasoning that both arms share a segmentation so it must come out
identical. It does not (0.9036 → 0.9219 on test), and the reasoning was wrong, not the
run: `group_contour` works on the *contour's* grid, so switching contours also moves
grouping from 23.2 ms to 11.6 ms and changes how the voicing mask resamples. The A/B
measures the learned pitch source as a package — accuracy plus grid resolution — which
is the right thing to measure, but not the narrower claim.

### Re-audit: the finer grid moved two constants

`MIN_NOTE_S` and `VOICING_THRESHOLD` were tuned against pyin's grid. Re-swept against
the learned contour, selected on validation, confirmed on test:

| setting | val Δ | test Δ |
|---|---|---|
| min=0.08 voi=0.3 | +0.0074 | +0.0102 |
| **min=0.10 voi=0.3** | +0.0095 | **+0.0128** |
| min=0.12 voi=0.3 | +0.0109 | +0.0101 |

Validation kept climbing to 0.12 while test turned over — riding a boundary on 40
tracks. **0.10 is the middle of a flat 0.08–0.12 plateau**, taken over the argmax. It
does cap recall by discarding real notes under 100 ms (~7% of the corpus), accepted
because those fragments are more often FPs than notes; revisit if short-note recall
ever binds.

Breaking a silent coupling was a precondition: `LearnedEncoder` imported the decoder's
`VOICING_THRESHOLD` to gate the contour, so retuning the decoder would have retuned the
contour too and shipped a combination nobody measured. It now has its own
`CONTOUR_VOICING_THRESHOLD`, still 0.5; the numbers above hold only for that pair.

### Where this leaves the headline

Held-out test, `separated`, strict note F1, over one day:

| | test F1 |
|---|---|
| `rung0-mono`, as it stood that morning | 0.7421 |
| + pitch-split fix | 0.8172 |
| + learned pitch head | 0.8509 |
| + re-audited decoder constants | **0.8638** |
| + mix channel (`rung1-bichannel-pitch`, next section) | **0.8849** |

**+14.3 points**, none of it from a bigger model or more data.

## Rung 1 revisited: the mix channel, once pitch is not the bottleneck — 2026-07-31

`rung1-bichannel-pitch` — one variable from `rung4-pitch`: `--sources
bichannel-clean,bichannel-separated` instead of `clean,separated`. Same 8 epochs, same
batch 32, same `train_batches` (10347), no class weights, learned pitch head in both.

### The headline

`pitch_ab.py` on the trained 2-channel checkpoint, held-out **test**:

| condition | split | pitch | recall | prec | strict F1 |
|---|---|---|---|---|---|
| bichannel-separated | validation | pyin | 0.9175 | 0.9377 | 0.8291 |
| bichannel-separated | validation | **learned** | 0.9469 | 0.9240 | **0.8953** |
| bichannel-separated | **test** | pyin | 0.8955 | 0.9338 | 0.8162 |
| bichannel-separated | **test** | **learned** | 0.9617 | 0.9175 | **0.8849** |
| bichannel-clean | test | pyin | 0.9917 | 0.9906 | 0.9538 |
| bichannel-clean | test | learned | 0.9952 | 0.9750 | 0.9533 |

Against `rung4-pitch` (1-channel, learned pitch) on the same split:

| | recall | prec | test F1 |
|---|---|---|---|
| `rung4-pitch` (1ch) | 0.9386 | 0.9058 | 0.8638 |
| `rung1-bichannel-pitch` (2ch) | 0.9617 | 0.9175 | **0.8849** |
| Δ | +0.0231 | +0.0117 | **+0.0211** |

**Both sides move.** The mix channel is not trading precision for recall the way the
pyin-era bichannel runs did.

### This overturns rung 1's "DONE — no"

Bichannel was recorded as a null twice, and the reasoning each time was that its cue
advantage (onset AP +4.3, later +11.3) failed to convert into notes. That was a true
observation with a wrong conclusion attached. The gain was real and was being *absorbed
downstream*: the extra notes a mix channel recovers are the quiet, separation-damaged
ones, which are exactly the notes pyin pitched worst, so every note it recovered arrived
with a pitch error and scored as a false positive. Removing pyin removed the sink.

The no-training screen predicted +1.33 on test and explicitly flagged itself as a lower
bound, because its pitch head came from a 1-channel checkpoint and had never seen a mix
channel. The trained model gives +2.11, which is consistent with that flag: a shared
trunk lets the pitch head use the mix channel too.

### Mechanism check

The gain is where the mechanism says it should be. On **clean**, 2ch scores 0.9533
against 1ch's 0.9613 — flat to slightly down, no mix channel needed when the bass stem
is undamaged. The entire +2.11 sits on `separated`. Validation `note/*` per epoch tells
the same story from the other direction: `bichannel-clean` starts at 0.9781 and ends at
0.9848 (the 1-channel run's range), while `bichannel-separated` starts at **0.8850**,
above the 1-channel run's best-ever 0.8222 at any epoch.

The pitch gap on test separated narrows 0.0981 → 0.0542 with the learned contour, the
same size of move the 1-channel run showed (0.090 → 0.058).

### Caveats

- **The checkpoint is not the best epoch on the metric we care about.** Early stopping
  monitors inline val loss, which is computed on *clean* windows only; it picked epoch
  5. `note/bichannel-separated/f1` was still climbing at epoch 7 (0.8952 → 0.8984). The
  quoted 0.8849 is therefore from a checkpoint that is ~0.3 points short of where the
  run ended on validation. Worth fixing the selection criterion before reading small
  deltas off this pipeline; 8 epochs also looks short of convergence.
- Single run per arm, no seed variance. A ~2 point gap on 40 test tracks is larger than
  the decoder-constant deltas that were treated as boundary-riding, but it is one draw.

## Divergence: the first semi-CRF + pitch run (2026-08-03)

The first run of the composite loss (`--objective semicrf --pitch --sources
clean,separated`, batch 8, `pitch_weight 1.0`, lr 1e-3) **went NaN and kept training.**

| step | 100 | 800 | 4400 | 6900 |
|---|---|---|---|---|
| `nll` | 70.5 | 25.5 | **nan** | nan |
| `pitch` | 1.50 | 0.40 | **nan** | nan |

Both terms failed on the *same* step, which implicates the shared trunk rather than
either loss term — the composite loss itself had 4300 healthy steps behind it, and
`pitch_ce` 1.50 → 0.40 says the pitch head does train at `pitch_weight=1.0` despite
the scale gap. Two defects, and the second cost far more than the first:

1. **No gradient clipping existed anywhere.** `zero_grad → backward → step` with
   nothing between, Adam at 1e-3.
2. **Nothing checked that the loss was finite.** `nan < best_val` is False, so early
   stopping never fires and never checkpoints; the run would have reached
   `--epochs 50` and saved NaN weights through the ordinary success path. It burned
   ~30 min of 5090 time on NaN before being killed by hand.

### Gradient-norm scale, measured (2026-08-03)

Both are fixed: the loop now clips and refuses to continue past a non-finite loss or
gradient. Picking the threshold needed the norm scale, which nothing had ever
measured — a 1-epoch smoke run (12 tracks, batch 4, same objective) gives:

| step | 10 | 50 | 100 | 200 | 270 |
|---|---|---|---|---|---|
| `grad_norm` (EMA) | 5380 | 2680 | 1300 | 465 | 485 |
| `nll` | 151 | 95 | 62 | 39 | 39 |

So the semi-CRF settles around **~500**, from ~5000 at initialisation. The first
default tried was 5.0 — which would have clipped *every* step by ~100x. That is worth
recording because it is not a harmless mistake in a way Adam hides: Adam normalises by
the second moment, so uniform clipping barely changes the update and the run still
descends convincingly. What it destroys is the *relative* magnitude between steps,
which is the only thing distinguishing a divergent step from a normal one. A clip
below the ordinary norm looks fine and protects nothing.

Default is now **1000** — above the semi-CRF's steady state, inert at the frame path's
much smaller scale (BCE loss ~0.2 against the CRF's ~44), so it does not disturb
comparability with the recorded `tcn-*` rows. The per-batch norm is logged as
`grad_norm`, so the threshold is tunable from the run instead of guessed.

### Root cause: a NaN gradient from a finite forward (2026-08-04)

Neither divergence was the missing clip, and neither was the learning rate. Re-running
with clipping reproduced the failure at step 6548 — and the guard reported the thing
that localised it: **a non-finite `gradient`, with a finite loss.**

Three facts pin it. `grad_norm=nan` not `inf`, so a singularity rather than overflow.
The loss guard did not fire that step, so the forward was finite and the backward was
not. And the two runs blew up at *different* steps (4400, 6548) from **identical loss
curves through step 4300**, so it is data-dependent.

`pitch_cumsums` built the Bhattacharyya coefficient as `p.sqrt()` on a softmax.
`d(√p)/dp = 1/(2√p)` is unbounded as `p → 0`, and a softmax over 216 bins underflows
to *exactly* 0.0 in float32 once the logit spread passes ~104 — which a pitch head
reaches by simply becoming confident. Backward computes `1/0 → inf`, softmax's own
backward turns that into `0 · inf → NaN`, and every forward value stays finite.

**This also explains the slow death, which is the part clipping and lr could not.**
Both runs learn to ~step 1000 and then *unlearn*:

| step | 100 | 500 | 1000 | 2000 | 3000 | 4300 |
|---|---|---|---|---|---|---|
| `nll` | 70.5 | 39 | **23.3** | 27.7 | 38.9 | 28.2 |
| `pitch` | 1.50 | 0.933 | **0.456** | 2.25 | 3.52 | 3.01 |

`pitch` climbs from 0.456 back to ~3.5–4.0 against a uniform-head value of
ln(216) ≈ 5.38 — most of the way back to untrained. The exact
zero is only where it tips over — the whole region below ~1e-20 already contributes
gradients of order 1e10 into the shared trunk, growing as the head sharpens. **The
pitch branch was punished precisely for becoming confident.** A feedback loop, not an
optimisation problem.

Fix: `p.clamp_min(1e-9).sqrt()`, the idiom already on the adjacent line where `log`
got the same treatment. `clamp_min` pins the underflowed tail's derivative at zero,
where it belongs. The BC value moves <3e-7 on ordinary logits.

**Audit — `sqrt` was the only unguarded singular op.** `torch.log(voicing)` and
`torch.logit(onset)` are clamped to `[1e-6, 1-1e-6]`, `p.log()` in `pitch_spread` to
1e-9, and `log2` in `models/heads/pitch.py` runs on targets carrying no gradient.
Related non-bug, now pinned by a test: those clamps also *zero* the derivative once
onset/voicing saturate, so the CRF stops teaching them on those frames.

**Lesson for the test suite.** Assertions on returned values cannot catch this class —
the forward was correct throughout, so 300+ passing tests and every guard watching the
loss were blind. Only a gradient assertion sees it, and it has to run at the
confidence a *trained* encoder reaches: an untrained head is diffuse and safe, so a
suite exercising only initialisation tests the one regime where the bug does not
exist.

## The composite objective — what the structured NLL alone would not train (2026-08-04)

Four defects fixed together, deliberately: each is a defect rather than a tuning, and
each costs ~2 h to isolate. It does mean a good result cannot say which one did it —
ablate back with `--onset-weight 0` / `--voicing-weight 0` / `--accumulate-steps 1`.

1. **Gradient accumulation** (`--accumulate-steps`, run uses 16). The detached alpha
   carry was ~53 optimiser steps stale by a track's end (median 53 chunks/track, p90 69,
   max 96) — R2D2's representational drift. Recomputing the prefix is O(N²) ≈ 27× ≈ 50 h
   per epoch; freezing the weights across the window costs nothing. **This is what
   stopped the divergence**, not the `sqrt` fix above, which was at most the terminal
   event.
2. **Per-frame NLL.** `log Z − score(gold)` is the surprisal of one structured object;
   the heads report means. Adding a total to an average is dimensionally incoherent and
   no weight repairs it — the weight just absorbs the frame count. Measured gradient
   share into the shared trunk: **1233:1 → 3.31:1** on dividing by scanned frames.
3. **Auxiliary per-frame terms** — BCE on onset and voicing, categorical CE on pitch.
   The NLL is discriminative *over segmentations*: a feature earns gradient only in
   proportion to how much it helps rank gold above the alternatives, and every span
   feature carries a learned scalar that can shrink toward zero. Nothing asks a head to
   be **calibrated**, only to be useful for ranking.

   | | onset AP | voicing AUROC |
   |---|---|---|
   | frame BCE (5 runs) | 0.994–0.996 | 0.995–0.996 |
   | semi-CRF (3 runs) | 0.20–0.53 | 0.510–0.522 |

   Consistent across every run of each kind, so structural rather than early-epoch
   noise. `cos(nll, voicing_bce) = +0.871` — the NLL already "wants" what the aux term
   supplies and was simply far too weak to get it. No pair of terms has a negative
   cosine, so nothing is fighting and gradient surgery is unnecessary.
4. **Selection on a harness metric** (`--monitor`), forcing synchronous scoring. Val
   loss was caught choosing the *worse* model: epoch 1 best by loss (16.25 < 20.04) with
   **half** the F1 (0.4086 vs 0.8175).

Gradient attribution at init, by projection onto the actual update direction (sums to
100%, unlike norm shares): `nll 20.9%  onset 33.2%  voicing 45.2%  pitch 0.6%`. Pitch's
0.6% is not starvation — `cos(pitch, total) = 0.080`, nearly orthogonal, because
`FrequencyHead` taps the front-end *before* the temporal stack.

### Result: epoch 0, same validator, same 37 val tracks

| | before | after |
|---|---|---|
| clean F1 | 0.8175 | **0.9215** |
| separated F1 | 0.6523 | **0.7236** |
| onset AP (clean) | 0.3746 | **0.9170** |
| voicing AUROC (clean) | 0.5160 | **0.9907** |
| fragmentation | 691 | **186** |

Best semi-CRF result on record, at epoch 0. The cues moved from near-chance into
frame-BCE territory, and fragmentation −73 % is the over-segmentation bucket this
objective exists to fix finally moving.

**This reframes "semi-CRF loses to frame BCE by 6.4 F1"** (0.8733 vs 0.9375 on identical
tracks). That comparison was substantially 0.99-quality cues against 0.4-quality ones,
not a verdict on the objectives. With aux terms the semi-CRF objective *contains* frame
BCE, so the open question becomes whether the structured term adds anything on top.

Also worth recording: the DP reached note F1 **0.8175 on near-chance cues** (onset AP
0.375, voicing AUROC 0.516) before any of this. A point in the architecture's favour.

## The recall collapse was the decode, not the model (2026-08-04)

Epochs 1 and 2 of that run looked like a catastrophe. Every loss term improved, every
cue metric improved, and note F1 fell off a cliff.

| clean | ep 0 | ep 1 | ep 2 |
|---|---|---|---|
| notes emitted | 13,516 | 9,310 | 8,783 |
| precision | 0.9461 | 0.9328 | 0.9415 |
| recall | 0.8981 | 0.6099 | 0.5808 |
| F1 | 0.9215 | 0.7376 | 0.7184 |
| fragmentation | 186 | 47 | 30 |
| `fp_rest` | 210 | 232 | 198 |
| `fp_interior_offpitch` | 333 | 347 | 286 |
| val `nll` | 0.0667 | 0.0510 | 0.0453 |
| onset AP | 0.9170 | 0.9415 | **0.9546** |

Separated the same shape, worse: 10,305 → 4,965 → 5,145 emitted, F1 0.7236 → 0.4142 →
0.4479. References: **14,238 notes**.

Two things rule out "it got worse at the task". It dropped **true positives faster than
false ones** (TP −32 %, FP −14 %), which is the opposite of a decoder declining risky
notes. And of the three false-positive classes only fragmentation improved — spurious
notes in rests and interior off-pitch notes both *rose*. It kept its mistakes and
deleted boundaries around them. That is a granularity knob moving, not a judgement.

### Mechanism: `log Z` sums, Viterbi maxes

Training minimises `log Z − score(gold)`, and **there is no argmax anywhere in it** —
`log Z` is a logsumexp over every segmentation and `score(gold)` needs no search. Only
inference maximises.

`log Z` therefore counts the `C(T−1, k−1)` ways to place `k` boundaries. For the right
*total* mass to sit on "k notes", each individual fine segmentation must score low —
roughly `−log(mean segment length)` per segment. Viterbi compares one structure against
one structure: it pays that per-span cost and never sees the multiplicity that justified
it. **The best single path is coarser than the distribution's own expected note count.**

It also worsens as the model sharpens, because the residual uncertainty concentrates in
boundary *placement* — mass split across 20 plausible split points at 3 % each loses to
a single no-boundary structure at 40 %, even though the split reading holds 60 %.

Not a new phenomenon: ASR corrects it with the **word insertion penalty**, NMT with
length normalisation, and Eikema & Aziz (2020) showed NMT's exact mode is often the
empty string while the distribution is well calibrated.

### Confirmation 1 — the bias sweep (`scripts/sweep_note_bias.py`)

`note_bias` adds a constant per `note` span at decode, so a `k`-note segmentation shifts
by `k · bias`. Training is untouched. `bias 0.00` reproduces epoch 0's logged numbers
exactly, so the harness path is sound.

| bias | clean F1 | clean /gold | sep F1 | sep /gold |
|---|---|---|---|---|
| −1.00 | 0.6176 | 0.56 | 0.3034 | 0.28 |
| −0.50 | 0.8362 | 0.81 | 0.5259 | 0.48 |
| **0.00** | 0.9215 | 0.95 | 0.7238 | 0.72 |
| **+0.20** | **0.9287** | 0.99 | 0.7696 | 0.81 |
| **+0.40** | 0.9210 | 1.04 | **0.7916** | 0.90 |
| +1.00 | 0.8527 | 1.24 | 0.7426 | 1.20 |
| +1.50 | 0.7877 | 1.44 | 0.6642 | 1.53 |

**+6.8 separated F1 and +0.7 clean, from one scalar on unchanged weights.** Both peaks
are broad — clean within 0.002 across 0.10–0.30, separated within 0.009 across
0.30–0.60 — so this is not a knob fitted to noise on 37 tracks.

Three things fell out that fitting F1 does not force:

1. **The peak lands near where emitted count crosses the reference count.** Clean peaks
   at 0.99 of gold, which is the prediction almost exactly. Separated peaks at 0.90,
   *under* gold, though its plateau spans 0.85–0.98 so gold sits inside it. Confirmed on
   clean, approximately right on separated.
2. **The negative side reproduces epochs 1 and 2.** All four collapsed points sit
   **+0.035 to +0.057 above** the epoch-0 model biased to the same note count, and the
   gap grows with epoch — their better cues showing through. They were better models
   read through a worse readout.
3. **The correction scales with boundary uncertainty.** Clean (onset AP 0.917) needs
   +0.20; separated (AP 0.736) needs +0.40. Blunter evidence spreads mass over more
   candidate positions, so the penalty is larger.

### Confirmation 2 — expected count, with no labels (`scripts/expected_note_count.py`)

For a log-linear model `d log Z / d w_f = E_model[f(y)]`, and `b_note`'s feature *is*
the note-span count. So one `autograd.grad` through the existing differentiable forward
gives what the distribution believes, with no reference notes and nothing tuned.

Clean, epoch-0 checkpoint: **`E[count]/gold = 1.04` against `viterbi/gold = 0.96`**, and
the same ordering on every track individually. The model expects more notes than its own
decode emits, and lands near the references.

This matters because the sweep is measured against labels and so cannot separate "the
readout is biased" from "a count-shaped knob happens to help". This can.

### What follows

- `note_bias` is a **diagnostic** and stays a `decode()` argument that nothing persists.
  It is a val-fitted constant; shipping it would mean every future result silently
  carrying someone's tuning of 37 tracks.
- **The fix is marginal / MBR decoding** — sum instead of max. Cheaper than it sounds:
  reverse-mode autodiff through the forward scan *is* the backward algorithm (Eisner
  2016, *Inside-Outside and Forward-Backward Algorithms Are Just Backprop*), so there is
  no second DP to write and verify. Caveats: `viterbi` is `@torch.no_grad()`, so
  marginals need their own entry point; and posterior decoding maximises expected
  per-position correctness, which can emit structures the duration and transition priors
  would have rejected. A constrained MBR is likely the right form, since those priors are
  what the semi-CRF exists to provide.
- **Selection is only as good as the decode it runs through.** `--save-epochs` now keeps
  every epoch so selection can happen offline, where the readout can change without
  re-training. Three improving epochs were discarded and could not afterwards be
  re-scored, which is what made this expensive to establish.
- **Soft onset targets move from parked to live** — see the revised note below.

**Caveats.** Every number here is validation-tuned on 37 tracks and needs a held-out test
re-run before being quoted. And the semi-CRF at its best bias (0.7916 separated / 0.9287
clean) still trails `rung1-bichannel-pitch` and `rung4-pitch` — closer than 0.7236 made
it look, not ahead.

## Posterior decoding is closed: the typical set is worse than the mode (2026-08-05)

The entry above predicted "the fix is marginal / MBR decoding". It is not. Five
decision rules were built on the same epoch-4 checkpoint and all five lost to
`MAP + note_bias 1.5`. The first four were maximum-expected-accuracy rules, each a guess
at which **additively decomposable** proxy stands in for the note metric:

| rule | emitted / gold | why it failed |
|---|---|---|
| `sum log P(span)` | 0.26 | scores a segmentation by its *probability*, so each extra span multiplies in another factor < 1 — the length bias, reintroduced |
| `sum (P(span) − ½)` | 0.31 | only 63 of ~584 expected spans clear ½; the rest cost ½ and earn nothing, so the DP minimises span count |
| `sum (P(boundary) − ½)` | 0.27 | aggregating over starts gives 253 confident boundaries, 4× better and still not enough |
| windowed over ±1 frame | 1.28 | count finally right; **precision 0.41, fragmentation 1834** |

Read on their own these look like four bad guesses, and they invite a fifth. They are
not. The fifth rule — **sampled MBR** — has no proxy in it at all: draw whole
segmentations, score candidates against the real matcher, keep the most central one. It
was the one member of the family whose loss could not be the wrong choice.

### The oracle probe

`scripts/mbr_oracle_probe.py` does not build MBR, it *bounds* it. Forward-filtering
backward-sampling draws K=32 exact posterior samples per track, and four rules are scored
side by side over identical evidence: MAP+bias, the mean sample, the unlabelled consensus
pick, and the **oracle** — the best sample per track, chosen with labels. The oracle is
the ceiling of any selection rule over that pool, so it can return a verdict rather than
a fifth ambiguous null.

Separated, 11 val tracks, macro-averaged per-track F1:

| rule | F1 | notes/gold |
|---|---|---|
| MAP + bias 1.5 | **0.7634** | 0.91 |
| sample mean | 0.5383 | 1.02 |
| MBR consensus (K=32) | 0.5771 | 1.05 |
| **ORACLE best of 32** | 0.5883 | 1.03 |

The oracle loses by **0.175 F1**, and loses on **11 tracks out of 11**. Nothing selects
its way out of that: no larger K, no better utility, no cleverer consensus. And the
consensus pick sits within 0.011 of the oracle, so the sampler is not the problem either
— MBR is *working*, and working perfectly still loses.

### The finding, which explains all five at once

**The samples have the right note count and are still much worse.** That was the entire
premise of this line of work — the posterior knows there are ~14,000 notes, Viterbi emits
0.38 of them — and it is true (1.02 of gold, against MAP+bias's 0.91) and it does not
help. Getting the count right is not getting the notes right.

So the posterior's mass is spread across many mediocre segmentations while its **mode is
genuinely the good one**. Any rule that integrates over the distribution gets dragged
toward the typical set; only maximisation finds the mode; and `note_bias` works because
it moves the mode to the right granularity without integrating over anything. That is one
explanation for five failures rather than five explanations, and it is what makes this a
closed branch instead of a losing streak.

It does not touch the diagnosis in the entry above, which stands: the collapse *was* the
readout. It says the crude fix is the fix.

**Still open, and unaffected:** max-margin training, which does not correct the readout
but makes training and decoding the same operation — it optimises the argmax the decode
actually uses. Costs a retrain, an L2 term (the hinge is satisfiable by inflating score
scale, which `log Z` self-normalises against), a re-derived aux-term balance (a hinge is
in score units, not nats/frame), and permanently loses probabilistic outputs — including
the per-note confidence column and the `E[count]` diagnostic that found this bug.

## Future experiments

### Learned similarity for the span features (idea, 2026-08-04)

`path` (`1 - Bhattacharyya`) and `spread` (generalised JSD) are **hand-crafted
similarity metrics** over the pitch distribution. Two limits are known:

- `path` is blind to pitch *distance*. Two sharp distributions a semitone apart and two
  an octave apart both give `BC ~ 0`, so the CRF cannot use interval structure at all --
  and for bass lines (root-fifth, octave jumps, walking steps) interval is among the
  most informative things pitch knows about where notes begin.
- Both read a distribution the pitch head produces for a *different* purpose, so the
  features are only as good as that head's shape.

**The idea:** replace the hand-designed metric with a *learned* one -- a small
projection trained contrastively so frames of the same note embed close and frames
across a boundary embed far. The span feature becomes a learned distance rather than a
fixed divergence, and semitone-vs-octave falls out for free.

**Why it is parked, not scheduled.** We have dense per-frame supervision, and
contrastive objectives earn their keep mainly where labels are scarce or the target
genuinely is a similarity structure. Nothing in the current evidence says the span
features are the bottleneck -- voicing AUROC at chance (0.516 vs frame BCE's 0.984) and
a dimensionally malformed composite loss are. Revisit only once those are fixed and the
span features are the thing left standing between the model and the metric.

Cheaper first step that captures part of the same gain without any representation
learning: a cumsum of frame-to-frame `|delta expected bin|`, which is additive, O(1)
per span, and adds "how far did pitch move" to the existing "did it move".

### Soft onset targets (idea, 2026-08-04)

`onset_targets` widens each onset by `tolerance_frames=1` into a flat 3-frame plateau.
That is **temporal label smoothing**, and it exists for a real reason: the true onset
falls *between* frames, so which frame is "the" onset is a quantisation artifact, and a
single hard frame is both too sparse to learn (<1% positives) and asserts a precision
the data does not have.

But a *flat* plateau asserts something else false -- that three frames are equally
correct -- so its argmax is ambiguous across ~35 ms, which is exactly what blunts
localisation. It also disagrees with the semi-CRF, whose gold boundary is exact: with
an auxiliary onset BCE, the BCE would teach "onset is a plateau" while the NLL teaches
"the boundary is here".

**The principled version is a soft target** -- triangular or Gaussian, centred on the
*exact* onset time. The centre of mass is then the true onset, neighbours still get
gradient in proportion to proximity, nothing claims a plateau, and the target's
centroid **agrees with the CRF's boundary**, so the disagreement dissolves rather than
being traded for another.

Ranking: **soft > flat +-1 > 0**. Do not set `tolerance_frames=0`; it is strictly worse
on every axis.

**Why it was parked, and why that no longer holds (revised 2026-08-04).** The original
argument: the plateau spans ~35 ms, *inside* the note metric's ±50 ms tolerance;
peak-picking collapses it and the CRF's duration priors disambiguate further, so the gap
between "soft" and "flat ±1" sits below the metric's resolution. Revisit if boundary
sharpness ever measures as the bottleneck.

That reasoning is about the **note matcher**, and it does not transfer to the semi-CRF's
decode. A flat 3-frame plateau trains the head to call three adjacent frames equally the
onset, which splits every boundary's evidence ~3 ways. Under `log Z` those three
positions **sum** (≈ +log 3 ≈ 1.1 nats); under Viterbi only the max survives. That is
~1.1 nats of self-inflicted penalty on every boundary, on top of the combinatorial term
— paid whether or not the matcher would have forgiven the timing. See "The recall
collapse was the decode, not the model" above; the condition this note set for revisiting
has been met.

Two qualifications before acting on it. **Marginal decoding partly subsumes this** —
summing over the plateau recovers the mass that maxing discards — so the remaining
benefit is boundary localisation and the BCE/NLL target disagreement, which are real but
smaller. And soft targets need a full retrain to evaluate, where the decode change is
testable on a checkpoint that already exists. **Sequence rather than bundle**: bias probe
→ marginal decoding → soft targets. The composite objective above shipped four changes at
once and cannot attribute; doing it again would buy another unattributable result.

**If it is ever done:** apply it to the auxiliary term only. Changing the frame path's
targets silently breaks comparability with `tcn-v1`/`tcn-v2`, which every row in "the
comparison that matters" depends on.


### Priority order — revised 2026-07-29

The staging below was written CRF-first, on the reasoning that Config A had validated
the architecture and that three simultaneous changes would be undebuggable. Both still
hold. But re-reading the error decomposition against what each lever can actually
reach changes the ordering, so the revised priority is:

| # | lever | reaches | cost | status |
|---|---|---|---|---|
| 0 | pitch-error sizing analysis | — (measurement) | hours, no training | not started |
| 1 | bichannel (bass, mix) input | demucs emission quality | small | **done — see above** |
| 2 | learned pitch head | pyin pitch/octave errors | large | gated on #0 |
| 3 | semi-CRF end-to-end | onset over-segmentation | built | POC running |

> **Superseded by the bichannel result (2026-07-30).** #1 is done, and its reasoning
> below was right about the *mechanism* and wrong about the *payoff*: it reaches demucs
> emission quality (onset AP +11.3) and emission quality is not what caps note F1.
> The pitch-blind scoring above says the cue→note step is worth ~2x the pitch head
> (0.197 vs 0.092 F1 on `demucs`), so the revised order is:
>
> | # | lever | reaches | evidence |
> |---|---|---|---|
> | 1 | note verifier head over frozen cues | spurious notes (pitch-blind P 0.746) | ceiling ~+0.13 F1 |
> | 2 | pitch-error sizing, then a pitch head | pyin pitch/octave | ceiling +0.09 F1 |
> | 3 | semi-CRF | same cue→note step as #1 | lost twice; see below |
>
> Keep bichannel on regardless; it is nearly free and strictly improves the encoder.

**Why the reorder.** The three levers are *orthogonal* — they change the encoder's
input, where pitch comes from, and how cues become notes respectively — so they
compose rather than compete, and any of them can be run on top of the others. What
differs is the error mass each can reach:

- **The semi-CRF changes only how cues become notes.** Pitch is read off the pyin
  contour by `GreedyDecoder` and `SemiCRFDecoder` alike; the CRF never invents it.
  So its reachable mass is the over-segmentation bucket alone — and per the error
  decomposition above, the residual after tcn-v2 is *over-segmentation **plus** pyin
  pitch/octave errors*. It is powerless on the second half. Config A's measured upside
  was precision 0.841 vs `group()`'s 0.826, i.e. **~1.5 points**, conditional on
  clawing back a 33-point recall deficit.
- **Bichannel is cheaper than the CRF was and aims at a bigger gap.** The stem→demucs
  drop is ~15 F1 at every stage, the largest single identified loss.
  `_ConvFrontend` already takes a channel count, so `(B,1,F,T) -> (B,2,F,T)` is nearly
  a one-liner; **both channels' CQTs are already cached** (the conditions table builds
  stem, mix and demucs-separated bass, all content-addressed); and it runs on the proven frame
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
