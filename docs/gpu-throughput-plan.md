# Semi-CRF throughput: why it was slow, and the refactors that fix it

Measured 2026-07-29 on the RTX 5090 box (torch 2.13.0+cu130, WSL2). This supersedes an
earlier conclusion in this repo's history that the segmental DP belongs on CPU. It does
not. That conclusion came from a batch sweep that stopped at B=16 — one notch below the
crossover — and from a micro-benchmark that, by accident, measured an already-optimised
loop body rather than the one we ship.

## The diagnosis

**CUDA per-step time is flat in batch size.** 0.44 ms at B=1 and 0.47 ms at B=256 for
`batched_log_partition`. Flat cost under 256x the work means none of it is arithmetic.
The profiler agrees: **72 CUDA kernels per step**, and 72 x ~6 us ≈ the observed 0.44 ms.
The launch count *is* the runtime.

CPU time, by contrast, grows with B, because the CPU is actually doing the maths. So the
two cross, and where they cross depends entirely on how many kernels the step launches.

| B | CPU shipped | CUDA shipped | CUDA +hoist | CUDA +hoist+graph |
|---:|---:|---:|---:|---:|
| 1 | **0.054** | 0.46 | 0.23 | 0.026 |
| 8 | 0.158 | 0.46 | 0.23 | **0.026** |
| 64 | 0.394 | 0.47 | 0.12 | **0.029** |
| 256 | 0.668 | 0.47 | 0.12 | **0.030** |

(ms per DP step, T=300, dmax=115, forward only.)

This also explains why `torch.compile(mode="reduce-overhead")` measured *worse* than
plain compile on CUDA: cudagraph trees re-clone inputs per call, and `prev` is a fresh
tensor every step, so per-step graph wrapping pays copies it can never amortise. The
granularity was wrong, not the tool.

## The three levers

### 1. Batch, which excerpts make possible

Today `SequenceDataset` yields whole tracks and `collate` pads to the longest in the
batch — on a real batch, 16,617 frames padded out to 23,355, so ~40% of the compute is
padding and batch size is capped by the longest track. Worse, whole tracks at B=16 need
338k frames x 160 MiB/1k = **54 GiB**, which does not fit. Some form of truncation is
forced regardless of architecture.

The chosen form is **truncated BPTT over fixed-size chunks**, not independent excerpts.
Process a track in chunks in order, carrying the DP's rolling alpha window across the
boundary and detaching it. Chunks are still fixed-size, so every throughput and graph
result below still applies, but two modelling problems that plain excerpts create simply
do not arise:

- the forward value is the **true whole-track log Z**, not a per-window approximation —
  only gradient flow is truncated;
- gold tiles the track continuously, so no note is clipped by a window edge and the
  learned duration potential `dur` is never taught a length the note does not have.

| chunk | CQT frames | DP steps | chunks/epoch (1 source) |
|---:|---:|---:|---:|
| 5 s | 431 | 144 | 61,841 |
| 10 s | 861 | 287 | 30,920 |
| 30 s | 2,584 | 861 | 10,307 |

**The loss must be per-chunk or gradient reaches only the last chunk.** Since
`log Z = logsumexp(alpha_T)` and the carry is detached, a single end-of-track loss would
stop at the final boundary. With `A_k` the log-mass at the end of chunk *k*:

```
loss = sum_k [ (A_k - A_{k-1}) - S_k(gold) ]
```

This telescopes to exactly `log Z - S(gold)` in value, while each term's gradient touches
only chunk *k*'s encoder activations — so each chunk backwards independently and frees
its activations immediately. `A_{k-1}` is detached and contributes only a constant.

**Context margins.** A chunk needs `dmax` = 4 s of *left overlap* in its emissions, and
this turned out to be a **correctness requirement of the scan itself**, not just a
nicety for gold. The carry supplies the *alpha* of positions before the chunk, but a
span crossing the boundary also reads *cue values* from before it — `onset[i]`,
`Vc[j]-Vc[i]`. Without the overlap the scan clamps those starts to the chunk's own frame
0 and scores them against the wrong cues, which is wrong rather than approximate, and
silently so. `test_dropping_the_left_context_changes_the_answer` is the negative control
that keeps the parity tests honest.

A chunk whose overlap already reaches global frame 0 may pass a shorter context: there
the clamp is correct, and the carry holds `NEG` in the unreachable slots.

The encoder's own margin is smaller — **33 frames**, half its 65-frame receptive field;
beyond that, chunk and full-track outputs agree to 1.8e-5, i.e. float32 noise (measured).
So `dmax` dominates: ~13% extra encoder compute at 30 s chunks.

**A bonus that falls out of the carry.** With a carry there is no `j < dmax` ramp — every
candidate start is a real frame — so reading the window at full `dmax` width at every
step gives *one static shape per step*. Out-of-range positions hold `NEG`, and
`exp(NEG - max)` underflows to exactly 0 in float32, so this is identical arithmetic. It
removes the varying-`W` problem previously flagged as the obstacle to `torch.compile` and
graph capture.

Plain fixed excerpts remain the fallback if carrying state proves awkward under graph
capture; in that case the edge rules in *Risks* below become load-bearing.

### 2. Hoist the potentials out of the j-loop

Every cue enters the DP as a cumsum difference, so the span potential factors into three
independent pieces:

```
note(i->j) = A[i]      +  cj[j]     +  dur[bucket[j-i]]
             i only       j only       offset only

A[i]  = w_on*onset[i] - w_v*Vc[i] - w_pitch*Jc[i+1] + b_note
cj[j] = w_v*Vc[j] + w_pitch*Jc[j]
```

All three are loop-invariant and can be built once before the scan; `unfold` then makes
each step's candidate window a **view**, so the per-step gather costs nothing. This is a
pure refactor with no modelling change — verified against the shipped scan to 1.6e-7
forward and **1.28e-13 on gradients in float64** (the float32 delta of 2.3e-5 is
accumulation order, not a discrepancy).

Effect: **72 -> 35 kernels per step, ~2x.**

**A hypothesis that was tested and rejected**, recorded so nobody spends a day on it:
storing the transition-combined message `beta[k,n] = logsumexp_p(alpha[k,p] +
trans[p,n])` in the rolling window instead of `alpha`. The algebra is correct and it does
delete the `(B, dmax, 3, 2)` `term` tensor, but it *raises* launch count 35 -> 54,
because the extra tiny logsumexp costs ~6 kernels while the reshape it saves costs ~2. It
would still pay off for *activation memory* under autograd, which is a separate axis.

### 3. Capture the whole j-loop in a CUDA graph

This is the big one, and it **works with backward** via
`torch.cuda.make_graphed_callables`. DP alone, fwd+bwd:

| excerpt | B | eager ms | graphed ms | gain |
|---:|---:|---:|---:|---:|
| 10 s | 64 | 194.8 | 23.0 | **8.5x** |
| 20 s | 32 | 333.5 | 44.4 | 7.5x |
| 30 s | 32 | 448.6 | 70.0 | 6.4x |
| 30 s | 256 | 684.9 | 519.6 | 1.3x |

The gain shrinks as work grows, which is the *good* direction — it means the scan has
stopped being launch-bound and started doing real arithmetic.

Correctness was checked the way a graph can silently lie: capture once, copy fresh
emissions into the static inputs, replay. Four distinct log Z values, each matching the
shipped scan to 1.5e-7 — so it genuinely recomputes rather than returning stale buffers.

## Memory

Encoder activation memory is **~160 MiB per 1,000 CQT frames**, and depends only on
`B x frames` — how that product is split between batch size and excerpt length does not
matter (160.4 vs 159.7 MiB/1k across a 24x range). The DP is not the constraint: 900 MiB
at 30 s x B=256, about 1/20th of the encoder.

Combined step (encoder + graphed DP + backward), measured:

| excerpt | B | B x frames | alloc GiB | reserved GiB |
|---:|---:|---:|---:|---:|
| 5 s | 128 | 55,168 | 8.7 | 15.7 |
| **5 s** | **256** | **110,336** | **17.2** | **25.9** |
| 10 s | 128 | 110,208 | 17.2 | 28.7 |
| 20 s | 64 | 110,272 | 17.2 | 28.6 |
| 30 s | 32 | 82,688 | 12.9 | 18.0 |

**B=256 fits only at 5 s excerpts** on the 32 GB card; at 10 s it wants ~35 GiB and OOMs.
The practical ceiling is ~110-125k frames per step, because reserved runs ~1.5x
allocated. Above ~26 GiB allocated the card thrashes badly rather than failing cleanly
(30 s x B=64 measured 15.7 **seconds** for one encoder step).

## Projected epoch

Compute only, from the combined-step measurements, on 1260 train tracks x 4.09 min:

| excerpt | B | ms/step | epoch (1 src) | epoch (3 src) |
|---:|---:|---:|---:|---:|
| 5 s | 256 | 215 | **52 s** | 2m36s |
| 10 s | 128 | 232 | 56 s | 2m48s |
| 30 s | 32 | 225 | 1m12s | 3m36s |

Data path: 22.1 GiB/epoch/source at a measured 1.49 GiB/s ≈ 15 s streamed — and it is
22.1 GiB regardless of excerpt length, since either way the epoch passes over each track
once. Per-excerpt CPU cost is ~3.2 ms (1.4 ms mmap'd slice + 1.7 ms contour load +
0.1-0.6 ms `pitch_jumps`), so ~3.3 min single-threaded at 5 s excerpts and ~25 s with 8
workers. Contours total only 165 MB and should simply live in RAM.

So **~1-2 min/epoch for one source, ~3-5 min for all three**, against the ~3.5 h *per
epoch* previously recorded. With `max_epochs=50, patience=8`, a realistic 20-30 epochs
puts a full 3-source run at **1-2.5 hours**.

**These projections depended on `gold_score` being vectorised first — now done.** The
combined-step benchmark fed the loss a *synthetic* gold tensor, so the table above
excluded it. Measured separately, the original per-segment loop was 7-70x the cost of
the graphed DP step:

| device | excerpt | gold segs | B | ms/track | ms/batch | vs graphed DP step |
|---|---:|---:|---:|---:|---:|---:|
| cpu | 5 s | 13 | 256 | 0.38 | 96.4 | 6.6x |
| cpu | 30 s | 75 | 32 | 2.09 | 66.9 | 1.0x |
| cuda | 5 s | 13 | 256 | 3.95 | **1011.8** | **69.8x** |
| cuda | 10 s | 25 | 128 | 4.86 | 622.4 | 20.8x |
| cuda | 30 s | 75 | 32 | 16.45 | 526.4 | 7.5x |

At 5 s x B=256 that turns a 215 ms step into ~1,230 ms and the epoch from 52 s into
~5 min — `gold_score` would be ~80% of training. It is pathological on CUDA for exactly
the reason the DP was: a Python loop over ~13-614 segments, each building a one-element
tensor and launching a handful of kernels on it. Whole-track it is 212 ms/track on CUDA
versus 17 ms on CPU.

The fix was mechanical and has no modelling content: gather every segment's
`(label, i, j)` across the whole batch into flat index tensors, evaluate both label
branches and select with `where`, index `trans` once for the transitions, and fold back
per track with `index_add`. O(1) ops per batch instead of O(B x segments).

**Result** — `gold_scores` (batched) against the original loop, same shapes as above:

| device | chunk | segs | B | loop ms/batch | vectorised ms/batch | speedup |
|---|---:|---:|---:|---:|---:|---:|
| cuda | 5 s | 13 | 256 | 1407.8 | **1.5** | 939x |
| cuda | 10 s | 25 | 128 | 840.3 | **1.2** | 677x |
| cuda | 30 s | 75 | 32 | 555.8 | **1.4** | 408x |
| cpu | 5 s | 13 | 256 | 119.6 | 1.0 | 123x |
| cpu | 30 s | 75 | 32 | 66.5 | 0.7 | 91x |

Whole-track (614 segments) went from 240.9 ms to 0.37 ms on CUDA, 645x. `gold_score` is
now ~2-10% of the graphed DP step rather than 7-70x it, so the projected epoch times
stand. Pinned by `test_gold_score_matches_the_per_segment_loop` and its gradient and
batched counterparts, which check against a reference loop kept in the test file.

## Validation is the next bottleneck, and it inverts the device answer again

`viterbi` builds its backpointer table as a list of Python dicts, calling
`int(argf[lab])` twice per step — on CUDA each is a device-to-host sync, ~14,000 per
whole track.

| device | variant | sec/track | 252 val tracks |
|---|---|---:|---:|
| cpu | shipped (dict) | 0.61 | 2m33s |
| cpu | tensor backptr | 0.56 | 2m20s |
| cuda | shipped (dict) | 4.42 | **18m33s** |
| cuda | tensor backptr | 2.58 | 10m50s |

Removing the syncs buys 1.7x on CUDA, but CUDA still loses 4x to CPU — because `decode`
is B=1, so there is no parallel work to find. Same disease as the forward pass, same
cure: there is no `batched_viterbi` today, and that is the missing piece. Segmentations
from the tensor-backpointer variant are identical to the shipped ones.

Note the asymmetry that keeps this honest: note F1 is a **whole-track** metric, so the
eval path cannot be excerpted the way training can. Batching there means batching across
*tracks*, which reintroduces variable lengths and therefore either length-bucketed graphs
or padding.

## Risks and unknowns

- **`gold_score` dominates if left alone** — measured above at 7-70x the graphed DP step.
  Vectorising it is the first item on the list, not an afterthought.
- **Gradients do not cross chunk boundaries.** This is the one thing truncated BPTT gives
  up. Since `dmax` = 4 s is much smaller than a 20-30 s chunk, most segments sit wholly
  inside one chunk and are unaffected; the effect is confined to the ~one segment per
  boundary that straddles. Worth an ablation against whole-track training on a `--limit`
  subset before trusting it at scale.
- **Batches mix chunks from different tracks at different positions.** The carry is
  per-track state, so the loader has to keep each track's chunks in order and route the
  right carry to the right batch row — a stateful sampler, not a shuffled one. This is
  the main implementation risk, and it constrains shuffling: you can shuffle *tracks*
  across batch rows but not *chunks* within a track.
- **Chunk-level NLL is not whole-track NLL per epoch**, so the inline val loss stops
  being directly comparable to figures in `experiments.md` (the summed telescoping value
  is comparable; a per-chunk mean is not).
- **Graph capture demands static shapes**, so chunk length and batch size must be fixed
  and the final partial batch dropped (`drop_last`) or padded. The variable-length `ends`
  readout in `batched_log_partition` has to go. A track whose length is not a multiple of
  the chunk size needs its tail padded or dropped.
- **fwd+bwd graph capture emits an `AccumulateGrad` stream-mismatch warning.** Benign
  here, but worth confirming under the real optimiser rather than assuming — and graph
  capture plus a detached carry that changes identity each step is exactly the
  combination that made `reduce-overhead` misbehave earlier.

## Refactor order

Each step is independently verifiable, and the first three are what unblock a real run.

1. **Vectorise `gold_score`** — per-track and batched. Remove the per-segment Python
   loop. Do this first: it is the one cost that could invalidate the projections above.
2. **Carried state in the DP** — let `batched_log_partition` accept an initial alpha
   window and return the final one, so a chunk can resume where the last left off.
3. **`ChunkedSequenceDataset` + stateful sampler** — fixed-size chunks in track order,
   `dmax` left-context margin, mmap'd feature slice, `drop_last`.
4. **Per-chunk telescoping loss** — `(A_k - A_{k-1}) - S_k(gold)` with a detached carry,
   gold segments assigned to the chunk containing their end.
5. **Move `pitch_jumps` into the dataset** so its Python frame loop runs in a worker.
6. **Hoist the potentials** in `batched_log_partition` (and keep `log_partition` /
   `viterbi` algebraically consistent with it).
7. **Vectorise `build_emissions` / `nll_batch`** over a uniform batch instead of a
   per-track Python loop plus `pad_emissions`.
8. **Graph the DP** — static-shape module wrapper, `make_graphed_callables`, `drop_last`.
9. **Tensorise Viterbi backpointers**, then **add `batched_viterbi`**.
10. **Wire the CLI** — `--objective`, `--device`, `--chunk-seconds`, `--batch-size`
    through `recipes.py`; raise the `batch_size=16` / `num_workers=0` defaults, which were
    set for whole-track training on an M3.

Tests to add alongside: **chunked-with-carry log Z equals whole-track log Z** on a
synthetic track (the property that justifies the whole approach), vectorised `gold_score`
parity, hoist exactness (the float64 gradient check), graphed-vs-eager parity, and the
recipe swap.
