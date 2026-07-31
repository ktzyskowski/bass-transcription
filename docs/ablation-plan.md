# Ablation plan

Written 2026-07-30 because experiments had started repeating each other. `docs/experiments.md`
is the *log* — what was run and what it showed, in order. This is the *plan*: which cells
are filled, which are empty, and what protocol makes them comparable at all.

**Read this before launching a run.** Two of today's runs re-derived a number we already
had, and a third was not comparable to the baseline it was implicitly measured against.

---

## 1. The comparability defect, and the fix

Every condition except one filters the corpus to single-bass tracks:

| condition | source | channels | `unambiguous_only` |
|---|---|---|---|
| `stem` | stem | 1 | **False** |
| `stem-solo` | stem | 1 | True |
| `mix` | mix | 1 | True |
| `demucs` | mix | 1 | True |
| `stem-mix` | stem | 2 | True |
| `demucs-mix` | mix | 2 | True |

`stem` is the odd one out, and it is the condition tcn-v1/tcn-v2 reported. So
**tcn-v2's stem 0.878 and this session's stem-solo 0.9375 are not the same measurement**
— multi-bass tracks are harder and only the first includes them. The gap is mostly
track set, not model. `stem-solo` exists precisely to fix this and should be preferred
for anything compared against a mix-input condition.

The `demucs` numbers *are* comparable across both eras (both single-bass, both through
the same separator), which is what makes the data-scale anchor below trustworthy.

### Pinned protocol

Every number reported from here on:

- **Tracks:** `unambiguous_only=True` everywhere. Required whenever the mixture reaches
  the model, so pinning it globally is what makes stem / demucs / bichannel one ladder
  instead of three.
- **Val:** 40 tracks of the validation split, fixed.
- **Headline:** `demucs` (or `demucs-mix`) note F1 — separation is non-negotiable at
  deploy (raw `mix` scores 0.269), so this is the number that ships.
- **Always also report:** pitch-blind F1 (`scripts/pitch_ceiling.py`), the four FP
  buckets, onset AP, voicing AUROC. Aggregate F1 alone hid the bichannel mechanism for
  most of a day.

---

## 2. What is already known

Do not re-run these.

| finding | evidence | confidence |
|---|---|---|
| Separation is mandatory | raw `mix` 0.269 vs `demucs` 0.729 (tcn-v2) | high |
| Voicing head is near-solved | AUROC 0.984 stem / 0.961 demucs; 0.5 threshold optimal | high |
| **Data scale barely moves F1** | demucs 0.7103 @37 tracks → 0.729 @1015 (**+1.9 for 27x**) | medium — different eras/codebases |
| Onset loss ≠ note F1 | tcn-v1: val loss falls, note F1 flat from epoch 1 | high |
| Bichannel: big cue gain, no note gain | demucs onset AP +11.3, F1 +0.0007; model does read ch 1 | medium — 1 seed, 19 val tracks |
| Error split ≈ 1/3 pitch, 2/3 segmentation | pitch-blind rescore, replicated on fresh checkpoints | high |
| Semi-CRF is not undertrained | step-matched ~6300 steps: 0.8733 vs frame 0.9375; overfits 37 tracks | high |
| Decode threshold cannot buy recall | recall saturates ~0.90 over threshold 0.05–0.7 | medium — 37-track encoder |
| `pos_weight` was never applied | declared, plumbed, never set; all runs unweighted BCE | high |

**The consequence of row 3 is the most important thing in this document.** If 27x the
data is worth under 2 F1, then ablations do not need the full corpus — **rank on a
subset, confirm the winner at scale.** That is what makes the ladder below affordable.

---

## 3. Cost model (measured 2026-07-30, RTX 5090)

| scale | batches/epoch | time/epoch |
|---|---|---|
| 1015 tracks x 2 conditions, B=32 | 10,347 | **~4.5 min** |
| 37 tracks x 1 condition, B=16 | 333 | ~30 s |

So a full-corpus 8-epoch arm is **~36 min**. Ablation arms are cheap; the expensive
things are pyin over new val tracks and demucs over uncached audio (both one-off).

---

## 4. The ladder

Each rung changes **exactly one** thing from the rung above, both arms on the pinned
protocol. Run at 200 tracks unless the result is close, then confirm at full corpus.

| # | rung | arms | question | status |
|---|---|---|---|---|
| 0 | **baseline** | 1ch, onset+voicing, `group()` | the reference on the pinned protocol | **MISSING — blocks everything** |
| 1 | mix channel | 1ch vs 2ch | does bichannel help at scale? | 37-track only |
| 2 | recall weighting | `pos_weight` 1 vs 5 | does candidate recall exceed 0.90? | running 2026-07-30 |
| 3 | note verifier | `group()` vs `group()`+verifier | does pruning beat the heuristic? | not built |
| 4 | pitch head | pyin vs learned | does the 1/3 pitch bucket shrink? | not built |
| 5 | semi-CRF decoder | `group()` vs CRF | parked — loses at 37 tracks, and rung 0's data anchor says scale will not save it | parked |

**Rung 0 is missing and everything else is measured against it.** Tonight's `balanced`
arm is 2-channel, so it is rung 1's *treatment*, not rung 0's control. The 1-channel
control on the pinned protocol has never been run on current code.

### Ordering rationale

Rungs 3 and 4 attack the two halves of the error split (2/3 segmentation, 1/3 pitch) and
are independent, so they compose. Rung 3 first: it is the larger bucket, and a verifier
is a strictly cheaper way to test "does a learned span scorer beat `group()`" than
reviving the semi-CRF — same capability, binary objective, no global normalisation, far
more sample-efficient. If it wins, upgrading the CRF's `note_potentials` to use it is a
small change against a DP that is already verified.

Rung 4 forces an encoder decision: `_ConvFrontend` flattens `(B,C,F,T) -> (B,C*F,T)`, and
that projection (~884k of 1.29M params) destroys frequency equivariance. Fine for onset
and voicing, wrong for pitch — so the pitch head branches *before* the flatten. Target
encoding: two-hot over the CQT's own 216 log-frequency bins (33 cents/bin, inside the
50-cent tolerance), not regression — octave errors are the project's central failure and
a regressor asked to choose between E1 and E2 outputs the midpoint.

---

## 5. Standing rules

1. **One variable per comparison.** Both today's confounded comparisons (epoch-matched
   semi-CRF, stem vs stem-solo) came from breaking this.
2. **Check the metrics landed.** Three broken runs in this repo have exited 0 — see
   `docs/experiments.md` on the silently-dropped MLflow metrics.
3. **Report the decomposition, not just F1.** FP buckets and pitch-blind F1 are cheap
   and they are what turn a null result into a mechanism.
4. **Checkpoints go in `data/checkpoints/`**, never `/tmp` — a session boundary destroyed
   four of them today and the pitch measurement had to retrain to proceed.
