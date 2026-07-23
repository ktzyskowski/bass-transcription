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

**Output:** a partition of the CQT frame grid into segments, each labeled `note` or
`rest`; `note→note` allowed (back-to-back re-articulation), `rest→rest` collapses.

**Segment (unary) score**, all read off the frozen frame heads:

- `note [i, j)`: `w_on · onset_logit[i]` (onset evidence *at the start*) `+ w_v ·
  Σ_{t∈[i,j)} log P_voiced(t)` (voiced across the span) `+ dur(j−i)` (learned
  duration potential) `+ b_note`.
- `rest [i, j)`: `w_u · Σ_{t∈[i,j)} log(1−P_voiced(t)) + b_rest`.

Cumulative sums make each span score O(1). Pitch is **not** in the v1 score — it's
still read from the pyin contour (median over the span) after decoding, exactly as
`group()` does. A learned pitch head folds in as a pitch-consistency term later.

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
