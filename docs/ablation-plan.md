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
| `clean-multibass` | stem | 1 | **False** |
| `clean` | stem | 1 | True |
| `mixture` | mix | 1 | True |
| `separated` | mix | 1 | True |
| `bichannel-clean` | stem | 2 | True |
| `bichannel-separated` | mix | 2 | True |

`clean-multibass` is the odd one out, and it is the condition tcn-v1/tcn-v2 reported. So
**tcn-v2's stem 0.878 and this session's `clean` 0.9375 are not the same measurement**
— multi-bass tracks are harder and only the first includes them. The gap is mostly
track set, not model. `clean` exists precisely to fix this and should be preferred
for anything compared against a mix-input condition.

The `separated` numbers *are* comparable across both eras (both single-bass, both through
the same separator), which is what makes the data-scale anchor below trustworthy.

### Pinned protocol

Every number reported from here on:

- **Tracks:** `unambiguous_only=True` everywhere. Required whenever the mixture reaches
  the model, so pinning it globally is what makes clean / separated / bichannel one ladder
  instead of three.
- **Val:** 40 tracks of the validation split, fixed.
- **Headline:** `separated` (or `bichannel-separated`) note F1 — separation is non-negotiable at
  deploy (raw `mixture` scores 0.269), so this is the number that ships.
- **Always also report:** pitch-blind F1 (`scripts/pitch_ceiling.py`), the four FP
  buckets, onset AP, voicing AUROC. Aggregate F1 alone hid the bichannel mechanism for
  most of a day.

---

## 2. What is already known

Do not re-run these.

| finding | evidence | confidence |
|---|---|---|
| Separation is mandatory | raw `mixture` 0.269 vs `separated` 0.729 (tcn-v2) | high |
| Voicing head is near-solved | AUROC 0.984 clean / 0.961 separated; 0.5 threshold optimal | high |
| **Data scale barely moves F1** | separated 0.7103 @37 tracks → 0.729 @1015 (**+1.9 for 27x**) | medium — different eras/codebases |
| Onset loss ≠ note F1 | tcn-v1: val loss falls, note F1 flat from epoch 1 | high |
| Bichannel: big cue gain, no note gain | separated onset AP +11.3, F1 +0.0007; model does read ch 1 | medium — 1 seed, 19 val tracks |
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
| 0 | **baseline** | 1ch, onset+voicing, `group()` | the reference on the pinned protocol | **DONE** — `rung0-mono`, separated **0.7451** |
| 1 | mix channel | 1ch vs 2ch | does bichannel help at scale? | **DONE — no.** 0.7431 vs 0.7451; AP +4.3 does not convert |
| 2 | recall weighting | `pos_weight` 1 vs 5 | does candidate recall exceed 0.90? | **DONE — barely.** 0.878 -> 0.903; ceiling holds |
| 3 | note verifier | `group()` vs `group()`+verifier | does pruning beat the heuristic? | **NEXT** — ~+12 pitch-blind F1 headroom |
| 4 | pitch head | pyin vs learned | does the 1/3 pitch bucket shrink? | not built |
| 5 | semi-CRF decoder | `group()` vs CRF | parked — loses at 37 tracks, and rung 0's data anchor says scale will not save it | parked |

**Rungs 0-2 ran 2026-07-30 overnight** (see `docs/experiments.md`). Rungs 1 and 2 are
both nulls, and between them they relocate the problem: the mix channel improves the
encoder (+4.3 onset AP) without moving note F1, and recall weighting cannot lift
candidate recall past ~0.90. **The cue->note step is the binding constraint, and the
~0.90 recall ceiling is structural** — it now holds across data scale, loss weighting
and decode threshold alike. Rung 3 is where the remaining headroom is.

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
   semi-CRF, clean-multibass vs clean) came from breaking this.
2. **Check the metrics landed.** Three broken runs in this repo have exited 0 — see
   `docs/experiments.md` on the silently-dropped MLflow metrics.
3. **Report the decomposition, not just F1.** FP buckets and pitch-blind F1 are cheap
   and they are what turn a null result into a mechanism.
4. **Checkpoints go in `data/checkpoints/`**, never `/tmp` — a session boundary destroyed
   four of them today and the pitch measurement had to retrain to proceed.

---

## 6. Condition names

Renamed 2026-07-30. The old set conflated two axes: `mix` meant "transcribe the raw
mixture" while the `-mix` suffix in `stem-mix` meant "the mixture is a *context*
channel" — the same word for opposite roles. A tag now reads as
`[bichannel-]<what is transcribed>`.

| old | new | meaning |
|---|---|---|
| `stem-solo` | `clean` | clean bass stem, single-bass tracks — **the pinned default** |
| `stem` | `clean-multibass` | clean stem, multi-bass kept — historical (tcn-v1/v2) only |
| `demucs` | `separated` | demucs output — **the deployment condition** |
| `mix` | `mixture` | raw mixture, no separation — negative control |
| `stem-mix` | `bichannel-clean` | clean stem + mixture as channel 1 |
| `demucs-mix` | `bichannel-separated` | separated bass + mixture as channel 1 |

`separated` rather than `demucs` names the *state*, not the tool, so swapping
separators does not invalidate a tag — the transform already encodes
`demucs-htdemucs-bass` for cache keying.

Old tags still resolve, with a warning (`LEGACY_NAMES`), so runs recorded in
`docs/experiments.md` stay reproducible. **Note the 2026-07-30 overnight runs
`balanced` and `recall-w5` were launched before the rename and log their metrics under
`note/stem-mix/…` and `note/demucs-mix/…`.** Read those as `bichannel-clean` and
`bichannel-separated`.
