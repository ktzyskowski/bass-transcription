# The semi-CRF note decoder

## Where this sits

The pipeline is **pitch tracker (pyin) + segmenter**. The segmenter turns a
per-frame pitch contour into discrete `NoteEvent`s: where does each note start,
where does it end, is a held sound one note or two re-articulations. Today that job
is done by `HeuristicSegmenter.group()` — a hand-written pass that splits the
contour at rests, pitch changes, and onsets, then optionally merges same-pitch
fragments back together. `LearnedSegmenter` swaps the DSP onset/voicing *cues* for
trained heads but inherits that same `group()` decode.

`group()` is a **greedy, local, threshold-driven** procedure. It commits to a
boundary the instant a cue fires, with no notion of whether the resulting *whole
sequence* of notes is plausible. That locality is the source of our #1 residual:
**over-segmentation** (21.5% of estimated notes on demucs are spurious — a held
note chopped into fragments by a mid-sustain flux blip or a one-frame pyin
dropout). The `merge_gap_s` post-pass is a patch for this, and it has a ceiling: it
can only rejoin, never reconsider, and a merge threshold that fixes fragments also
eats genuine fast repeats.

The semi-CRF replaces `group()` with a **structured global decoder**: instead of
committing boundary-by-boundary, it scores *every possible segmentation of the
whole track* and returns the single best one. "Best" is defined by learned
potentials, so the decision "is this a real note or a fragment?" is made globally,
against a learned model of what note sequences look like, not by a local threshold.

## What a semi-CRF is (building up from HMM → CRF → semi-CRF)

- An **HMM** labels each frame with a hidden state (`note` / `rest`) and models
  `P(states, observations)` as a product of per-frame emission and
  frame-to-frame transition probabilities. It's *generative* and *per-frame*: a
  note is just a maximal run of `note` frames.
- A **linear-chain CRF** keeps the per-frame state chain but is *discriminative* —
  it scores `P(states | observations)` directly with arbitrary (unnormalised)
  potentials, trained to match gold labelings. Still per-frame: no first-class
  notion of a "note" as an object with a duration.
- A **semi-Markov CRF (semi-CRF)** promotes the *segment* to a first-class citizen.
  The output isn't a per-frame label sequence — it's a **partition of the timeline
  into labeled segments** `(label, start, end)`. Because a segment is explicit, the
  model can score properties of the *whole span*: its **duration**, the onset
  evidence *at its start*, the voicing evidence *summed across it*, pitch
  consistency *within it*. None of those are expressible in a per-frame chain.

The name: *semi-Markov* because the Markov (memoryless) transitions happen between
*segments*, not frames — within a segment, arbitrary-length structure is allowed.

## Why it's the right fit for monophonic bass

1. **Monophony is a hard structural prior, and the semi-CRF encodes it exactly.**
   At most one note sounds at a time → the output is literally a non-overlapping
   partition of the timeline. A semi-CRF's output space *is* that partition. Valid
   by construction — the decoder can never emit two overlapping notes, unlike a
   frame model you have to post-process into notes.

2. **A duration potential kills over-segmentation at the root.** Give each segment a
   learned score as a function of its length, and an implausibly short "note"
   scores badly *globally*. To insert a spurious fragment the decoder must pay a
   weak-onset cost + a short-duration penalty + a re-articulation transition, while
   the alternative (one longer note, no split) pays none. So the split survives
   **only** when the evidence justifies it. This is `merge_gap_s`, `min_note_s`, and
   the onset threshold all replaced by one learned, globally-optimised potential —
   no hand-tuned knobs.

3. **It recovers sub-threshold onsets without lowering a threshold.** Peak-picking
   throws away every onset below 0.5. But ~59% of the notes we miss on demucs have
   onset evidence sitting at 0.1–0.5 — real signal the greedy decode discards. In
   the semi-CRF the onset probability enters a segment's score as a *soft emission*:
   a 0.3 onset contributes `log 0.3`, weighed by the decoder against the cost of
   *not* starting a note there. The **decoder**, not a threshold, decides — and it
   decides with the duration/transition context that tells a real weak onset from
   noise. (This is why lowering the onset threshold globally fails: it recovers
   recall at −5.4 precision because it has no such context. The structure is the
   lever, not the threshold.)

4. **It's a scorer + a pruner, cooperating.** The frame heads are a *proposer*
   (high-recall onset/voicing evidence); the semi-CRF is a *structured pruner*
   (precision via global consistency). Same factorisation as a region-proposal
   network + classifier in object detection — deliberately split by responsibility,
   both trained toward one objective. Not adversarial (no GAN-style minimax); they
   cooperate. This factorisation is what makes a *recall-shaped* onset loss
   meaningful: the proposer's job is not to miss, the pruner's job is to be right.

5. **It's tiny and data-efficient.** The decoder is ~15 parameters (emission
   weights, a small duration table, a 2×2 transition matrix, biases) on top of the
   frozen frame heads. It adds structured decoding without a bigger backbone or more
   data — appropriate, because our error analysis says over-seg is a *decoding*
   failure, not a *representation* failure (voicing AUROC is already 0.98).

## The model

> **Retired, 2026-08-11.** The log-linear scorer this section describes is no longer in
> the tree. It was the arm every neural result was quoted against, and both live
> checkpoints are neural, so it was cut along with the machinery only it read — the
> `path_c`/`entropy_c`/`pitch_c` cumsums, the five span accessors over them, and the
> duration-bucket map. The shipped scorer is `semicrf/neural.py`; the *recurrence*
> below is unchanged, which is the property the `SpanScorer` seam exists to have. This
> section is kept because the ablations in `docs/experiments.md` are measured against
> it, and because the pitch discussion further down is what motivated its successor.

**Output:** a partition of the CQT frame grid into segments, each labeled `note` or
`rest`; `note→note` allowed (back-to-back re-articulation), `rest→rest` collapses.

**Segment (unary) score**, all read off the frozen frame heads:

- `note [i, j)`: `w_on · onset_logit[i]` (onset evidence *at the start*) `+ w_v ·
  Σ_{t∈[i,j)} log P_voiced(t)` (voiced across the span) `+ w_pitch · (# pitch changes
  strictly inside)` `+ dur(j−i)` (learned duration potential) `+ b_note`.
- `rest [i, j)`: `w_u · Σ_{t∈[i,j)} log(1−P_voiced(t)) + b_rest`.

Cumulative sums make each span score O(1) — that constraint is load-bearing and
shapes every feature that can ever be added; see *Can the pitch head feed the CRF?*.

**Five features, ~15 learned scalars, one dot product.** This is a *log-linear*
model in the classical Sarawagi & Cohen sense: the encoder is deep, the thing scoring
segments is `wᵀf(segment)`. "Not a neural CRF" means exactly this.

**Three pitch terms, from two sources.** `w_pitch` weights `pitch_jumps` — a binary
indicator thresholded at 0.8 semitones on the *pyin* contour, the same cue `group()`
retired because its transition lags the onset. `w_path` and `w_spread` weight the two
features derived from the learned head's per-bin logits (see *Can the pitch head feed
the CRF?*); both are zero-initialised, so a run that gains them starts from the model
that did not have them, and both read zero when a checkpoint has no pitch head.

The *decoded* pitch is still read from the contour (median over the span) **after**
segmentation, so segmentation and pitch value remain decided independently — the
limitation the pitch-labelled segment set below would remove.

**Transition score:** a 2×2 learned matrix over `{note, rest}` plus an initial
distribution — the cost of `note→note` vs `note→rest` vs `rest→note`, encoding the
monophonic grammar.

**Inference — segmental Viterbi.** DP over `(frame, label)` with a max-segment
length cap `D_max` (~4 s). Complexity `O(T · D_max · L²)`, trivial for `L=2`.
Returns the globally optimal, guaranteed-monophonic note sequence.

**Training — negative log-likelihood** of the gold segmentation (derived directly
from the reference MIDI notes on the frame grid): `−[score(gold) − log Z]`, where
`log Z` (the partition function over *all* valid segmentations) comes from the
differentiable **segmental forward algorithm**. Two regimes:

- **Frozen heads + CRF on top** (the cheap prototype): train only the ~15 CRF
  params on cached frame-head probabilities. Proves the decoder is the lever
  without touching the network. If the CRF is starved for sub-threshold onset
  signal, a *recall-shaped* onset-head retrain (focal / higher `pos_weight`) lifts
  weak onsets into the band the decoder can use.
- **End-to-end**: backprop the CRF NLL into the frame heads too. The CRF objective
  then shapes the onset emissions automatically — recall-shaping is subsumed, and
  proposer/pruner co-optimise to one coherent operating point rather than two
  hand-tuned knobs.

**Where it slots in:** it replaces `group()` inside
`LearnedSegmenter.segment_features` — the frame heads (the scorer) are untouched;
only the decoder changes. pyin still supplies pitch for v1.

## Forward vs. Viterbi: one recurrence, two semirings

Training and decoding run the *same* dynamic program over the *same* quantity —
`alpha[j][L]`, the score of all ways to segment frames `[0, j)` ending in a segment
of label `L`:

    alpha[j][L] = COMBINE over (start i, prev label Lp) of
                      alpha[i][Lp] + trans[Lp, L] + seg_score(L, i, j)

They differ only in how competing options are combined:

- **Forward** (training) combines with **`logsumexp`**, so `alpha[T]` sums over
  *every* segmentation → `log Z`, the partition function.
- **Viterbi** (inference) combines with **`max`** (plus argmax backpointers), so
  `alpha[T]` is the single *best* segmentation → the notes we emit.

`logsumexp` is a *smooth, differentiable* relaxation of `max` (as a temperature
τ→0, `τ·logsumexp(x/τ) → max(x)`). So the forward algorithm is "soft Viterbi." That
smoothness is the whole point: because `log Z` is built only from differentiable
primitives (`+`, indexing, `logsumexp`), the Python loop over frames just *unrolls*
the recurrence into one autograd graph — exactly like backprop-through-time in an
RNN — and `loss.backward()` differentiates the entire DP automatically. We never
hand-write a backward pass; autograd does it because every step is differentiable.
(Everything lives in log-space, sums instead of products, both for that
differentiability and because a segmentation's raw probability is a product of many
small factors that would underflow to zero.)

`max`/`argmax`, by contrast, has no useful gradient — but Viterbi is only ever run
at *decode* time, under `torch.no_grad()`, to turn a trained model into notes. No
gradient is needed there; learning already happened through the differentiable
forward pass. So the division of labour is clean: **forward (soft, differentiable,
all paths) for learning; Viterbi (hard, best path) for inference.**

Why this trains the right thing: for a CRF, `∂ log Z / ∂ score(segment s)` equals
`P(segment s | observations)`, the model's marginal. So the gradient of the loss
`log Z − score(gold)` is `E_model[features] − features(gold)` — the model's
*expected* segmentation minus the *gold* one. Gradient descent drives the expected
segmentation toward the reference (e.g. if the model expects many short spurious
notes but gold has few, the short-duration bucket's potential is pushed down). That
is the semi-CRF learning to prune over-segmentation, expressed as one gradient — and
in the end-to-end regime the same graph carries that signal all the way back into
the frame heads.

## Note on the metric

Note-F1 uses onset ±50 ms + pitch ±50 cents and **ignores offsets**
(`offset_ratio=None`). So the semi-CRF's rest modeling and exact segment ends don't
score directly — what matters is *how many* note segments it emits, *where they
start*, and *their pitch*. Rests and the duration potential earn their keep
indirectly: by suppressing spurious note starts (precision) and by letting real
weak-onset notes exist (recall).

## Faithfulness ledger

What the implementation actually guarantees, and where it departs from the model
above. Checked by reading `gold.py`, `forward.py`, `score.py` and `potentials.py`
against the theory — 2026-08-01.

### Verified sound

For `log Z − score(gold)` to be a proper NLL, gold has to be a **member of the space
the partition function sums over**: an exact tiling of `[0, T)` with every segment
of length `≤ dmax`. Both hold.

- **Gold tiles exactly.** Every branch of `gold_segments` advances `cur` to the
  segment end; interior gaps fill through `add_rest`; the tail fills at the end.
- **Every gold segment is `≤ dmax`.** Notes clipped at `min(j, i + dmax)`; the
  extend-over-artifact branch bounded by `min(i, prev_start + dmax)`; rests chunked
  by `add_rest`.
- **`START` is reachable only at position 0.** Its alpha slot is written `NEG` after
  the first push, so no later segment can transition out of it.
- **`note→note` is representable and gold produces it** (the `GAP_REST_S` rule), so
  the model is not forced to invent a rest between slurred notes.
- **The recurrence is checked against brute-force enumeration** — `log_partition`,
  `viterbi` *and* the batched scan, at `T=7–8, dmax=3`. Every other DP test compares
  one implementation against another, which two copies of the same wrong recurrence
  would pass. This one does not.

### Known departures

1. **A note longer than `dmax` gets a `rest` tail.** `gold_segments` clips the note
   to `dmax`; the remainder falls through the gap logic and is labelled `rest`. Gold
   then asserts silence where the reference says a note sounds. Rare at `DMAX_S=4.0`
   for bass, but it is a silent mislabel rather than an approximation.

2. **Gold is one arbitrary tiling among several that mean the same thing.** A 10 s
   rest becomes `4+4+2`; `3+3+4` describes identical audio, is an equally valid
   member of the space, and the NLL treats it as a *negative*. The model compensates
   by learning `trans[rest][rest] ≈ 0`. Properly one would marginalise over
   equivalent golds. Mis-specification, not a bug.

3. **No STOP transition.** `log Z = logsumexp(alpha[T][:2])` with no end potential,
   so "tracks tend to end in a rest" is inexpressible. Looks deliberate; nothing
   records it as a choice.

4. **The pitch feature is pyin's** (see *The model*). One of five features is derived
   from the source the rest of the pipeline replaced.

### Structural risk

The unary is written out in **four places**: `potentials.note_potentials` /
`rest_potentials` (used only by the unbatched paths), `forward._scan` (the batched
and chunked hot path), and `score._terms` (gold, single and batched). The latter two
read `pot.w_on`, `pot.w_v`, `pot.b_note` and friends directly.

If `log Z` and `score(gold)` ever compute *different* unaries, the NLL still
decreases and gradients still flow — the model simply optimises a quantity that
means nothing. Nothing in the training loop can detect this. The interior-only
convention `Jc[j] − Jc[i+1]` is the sharpest edge: it appears four times, and in
`score.py` the `+1` is applied by the **caller** while `_terms` merely documents
"caller offsets by 1".

## Can the pitch head feed the CRF?

The pitch head emits `(T, 216)` bin logits per frame. Folding them in is the natural
fix for departure (4), but *what* can be folded in is fixed by the O(1)-per-span
constraint: there are `O(T · dmax)` spans, so an interior contribution must be
**additive over frames** (hence a prefix cumsum) or depend only on the endpoints and
the length.

Writing `p_t = softmax(logits_t)` over the 216 bins:

### Tier 0 — soft pitch instability (O(1), drop-in)

Per frame, `1 − ⟨p_t, p_{t−1}⟩`: near 0 while the pitch holds, rising as the
distribution moves. Cumsum it and a span's total is `C[j] − C[i+1]` — the *same
shape and cost as `Jc` today*. It replaces a thresholded pyin indicator with a
threshold-free, differentiable quantity from the learned head, and unlike `Jc` its
gradient reaches the encoder.

This is the free upgrade: no new machinery, one feature swapped.

### Tier 1 — span pitch disagreement (O(K) per span)

The segment-level question is "does this span hold *one* pitch?", which no per-frame
feature answers. Cumsum the distributions themselves (`Pc`, a 216-wide prefix sum),
take the span's mean `P̄ = (Pc[j] − Pc[i]) / d`, and score

    JSD = H(P̄) − (1/d) Σ_{t∈[i,j)} H(p_t)

the generalised Jensen–Shannon divergence of the span's frames: exactly zero when
every frame agrees on the pitch, large when the span straddles a change. The second
term is a per-frame scalar (cumsum, O(1)); only `H(P̄)` costs `O(K)`.

Cost at `B=8, dmax=115, K=216, chunk=128`: ~795 KB per scan step for `P̄`, ~100 MB
retained per chunk, ~300 MB including the entropy intermediates. Affordable.

**This is where multimodality earns its keep** — and it is worth being exact about
what that means here. The head *can* represent a bimodal posterior (an octave
ambiguity shows as two peaks rather than one averaged pitch), but it was trained on
monophonic bass against a single-pitch two-hot target, so a second mode means
**uncertainty, not a second simultaneous note**. Under Tier 1 a bimodal `P̄` is
evidence the span *contains a pitch change and should be split*, which is precisely
the boundary decision the CRF exists to make.

### What this does *not* buy: polyphony

Three independent reasons, any one of which is sufficient:

1. The pitch head never saw polyphonic supervision.
2. A segmentation is a **tiling** — one label per frame. Two overlapping notes are
   not expressible in the output space at all, whatever the features say.
3. `{note, rest}` has no way to name *which* note. Polyphony needs a factorial model
   (per-string chains, or a multi-track CRF), which is a different model class rather
   than a richer potential.

### The larger prize the question points at

Pitch is currently read **after** decoding — `_segs_to_notes` takes the median pyin
pitch over each emitted span — so segmentation and pitch are decided *independently*.
A span the CRF got wrong (straddling two notes) still receives a confident median,
wrong for both halves.

Extending the label from `{note, rest}` to `{rest} ∪ {note at pitch p}` makes the DP
choose pitch and boundaries **jointly**: the same recurrence at `O(T · dmax · K)`,
with pitch evidence able to *veto* a boundary rather than being applied downstream of
it. Expensive at `K=216` (bucketing to semitones gives `K≈48`), still monophonic —
but it addresses a real independence assumption, which polyphony does not.

### Why the pitch head still needs its own loss

Both features above measure *change* and *disagreement*. Neither measures **value** —
nothing in `path_length` or `pitch_spread` says the pitch is correct, only that it is
self-consistent and moves where the boundaries are.

That is the right division of labour (the head answers *what pitch*, the CRF answers
*where the notes are*), but it has a consequence: a pitch head trained under the NLL
**alone** can place every boundary correctly and still score near zero on strict F1,
because decoded note pitch is read from the contour those same logits produce and
nothing constrained its value. Random initialisation compounds it — a near-uniform
softmax over 216 bins makes consecutive frames near-identical, so both features start
near zero and the gradient through them is small.

A constant-output head is *not* an attractor, though it is easy to assume so: it zeroes
both features on every span, gold and competing alike, which makes them uninformative
rather than rewarded. The CRF's gradient is discriminative — it wants low spread on
gold note spans and high spread on spans straddling a boundary.

So: keep an auxiliary `pitch_ce` whenever the head is trainable under the CRF. The
alternative that needs no auxiliary loss is the pitch-labelled segment set above —
gold says "note at bin 40", the unary reads the head's probability *there*, and the CRF
supervises pitch value directly at `O(T · dmax · K)`.
