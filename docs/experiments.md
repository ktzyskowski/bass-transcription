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
`HeuristicDecoder`, best-F1 epoch reported. Each pair is on an **identical track set**
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
`HeuristicDecoder.onset_threshold` across 0.1–0.7 moves demucs pitch-blind F1 by less
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

## Future experiments

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
  contour by `HeuristicDecoder` and `SemiCRFDecoder` alike; the CRF never invents it.
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
