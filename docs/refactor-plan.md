# Codebase review & incremental refactor plan

An honest assessment of the review points. Each item: **verdict**, rationale, and
rough effort/risk. Verdicts are one of DO / DEFER / SKIP / DISAGREE. I push back on
two points (docstrings, alignment.py) — reasons inline.

Ordering matters: a few items unblock others. The dependency spine is
**metrics extraction (#5) → baselines-as-generic + LearnedBaseline (6a) → delete
segmentation/**. Do that spine first; the rest is independent.

---

## Tier 1 — high value, do first

### #5 — Split `evaluation/` into reusable metrics vs. the research harness  ✅ DONE (e43fd42)
The single best structural win here. `evaluation/` currently holds two unrelated
things:
- **`evaluation/metrics/`** — pure scoring primitives (`PRF`, `Fragmentation`,
  `VoicingAUROC`, frame/note metrics). Already imported by `training/validation.py`.
  These are general: any model, any split.
- **`frame.py` / `note.py` / `cli.py` / `report.py` / `alignment.py`** — the offline
  research harness that compares `PitchBaseline`s. Frame-centric, dataset-driven.

**Action:** promote `evaluation/metrics/` → top-level **`stemify/metrics/`**. Leave
`evaluation/` as the harness. This makes "scoring" a first-class, dependency-light
package that training and the harness both import, instead of training reaching *into*
`evaluation`. Low risk (mostly import moves), unblocks 6a and the segmentation delete.

### #7 — Break up `decoders/semicrf.py` (535 lines, highest bug risk)  ✅ DONE (51500fc)
You're right to be nervous — it's the largest file and the DP is where silent
off-by-one/label/index bugs hide. Split into a **`decoders/semicrf/`** subpackage of
independently testable units:
- `potentials.py` — the `SemiCRF` `nn.Module` (the ~15 params + `trans_to`).
- `emissions.py` — `Cues|TensorCues → _Emissions` (+ `TensorCues`, pitch-jump cumsum).
- `dp.py` — the two recurrences: `log_partition` (forward) and `viterbi`, batched.
  *(Split further on 2026-07-29 once it had grown a third job: `window.py` for the
  rolling alpha buffer, `forward.py` for the logsumexp semiring, `viterbi.py` for the
  max semiring, and `score.py` for scoring one given segmentation — see
  [gpu-throughput-plan.md](gpu-throughput-plan.md).)*
- `gold.py` — `gold_segments` (the audible-gap logic — test-dense, bug-prone).
- `decoder.py` — `SemiCRFDecoder` façade wiring them + `segs_to_notes`.

Each file gets its own focused test module. No behavior change; pure decomposition.
Biggest risk-reduction in the list. Medium effort, low risk (tests pin behavior).

### 6 / 6a — A `pitch` abstraction; stop `LearnedEncoder` hardcoding `PyinBaseline`  ✅ DONE (009e777)
Real coupling: `LearnedEncoder.__init__` defaults `baseline=PyinBaseline()` and calls
it for the contour, and `PyinBaseline` lives in `baselines/` (the *eval* package). So a
production model depends on an eval baseline — role confusion (your 6a).

**Precise scope — I partly disagree with the framing:**
- **YES** a `stemify/pitch/` module with a `PitchEstimator` (a.k.a. `ContourSource`)
  ABC: `estimate(audio|features) → Contour`. Impls: `PyinPitch` (wraps the pyin call),
  later `NeuralPitchHead`. `LearnedEncoder` takes a `PitchEstimator`, not a
  `PyinBaseline` — the learned head then drops in with no interface change (exactly the
  "encoder produces the contour" promise).
- **NO separate `onset`/`voicing` ABCs** — those are *outputs of the `CueNet`
  heads*, not pluggable strategies. Abstracting them would be ceremony around a tensor.
  Pitch is different: it has two genuine sources (pyin now, a head later), so it earns
  the seam. Onset/voicing have one source (the head).

`baselines/PyinBaseline` then becomes a thin eval-harness wrapper over `pitch.PyinPitch`
(or is defined in terms of it), so `baselines/` returns to meaning *only* "things the
harness scores." Medium effort, low risk. Depends on nothing; pairs naturally with #5.

---

## Tier 2 — worthwhile, independent

### #1 — Unify the cache  ✅ DONE (A: 885cecc; B + `Cache`)
**A:** `stemify/atomic.py` (shared `atomic_write`) + `stemify/serde.py` (formats;
`CachedItem` moved here) + `cache.py` as pure policy. Fixed the non-atomic
`CachedItem.save` corruption bug.

**B:** features are content-addressed on `(features.SIGNATURE, resolved_audio_path)`,
matching `contour`/`separation`; the `cache_dir/<split>/<tag>` partitioning is gone.
The condition no longer needs spelling into the directory tree — a stem, a mix and a
separated mix already resolve to three different files — so every condition shares one
features directory and the validator reusing the loader's work is automatic rather than
a stem-tag special case.

**The `Cache(root)` object earned its keep** (the conditional this item left open). It
removed 9 cache-path parameters across 7 signatures — `SequenceDataset`,
`CheckpointValidator` and `build_validators` each took two — and moved layout out of
`train.py`, which had been hand-building five subdirectories. It lives at
**`stemify/cache.py`**, not under `training/`: the eval harness runs the same
transforms over the same corpora and was recomputing pyin and the CQT every launch
with no way to reach what training had already built.

`Cache` owns the layout for all three artifacts (`features/`, `contours/`,
`transforms/`) and the keying convention (`content_key`), but the *transform* still
does its own writing — `cache.separator()` hands `DemucsSeparator` a directory. That
keeps the dependency one-way (`cache` imports `transforms`, never the reverse) so the
light audio path never pulls the cache in.

Two things fixed in passing: the targets backfill in `examples.py` wrote its npz
non-atomically (the same corruption class A fixed for `CachedItem.save`), and
`Identity.cache_path` inherited a digest path naming a `.flac` that is never written —
harmless while only `apply` used it, wrong once it doubles as the resolved-audio key.

Original design notes:

`cache.py` (just landed) is the seam. Remaining fragmentation, now visible:
- **Two keying schemes:** features key on `(name, transform)`; `cached_f0` keys on
  `sha1(baseline, resolved_audio_path)`. Pick one. Since you'll rerun on the 5090,
  we're free to change keys.
- **The transforms/Demucs cache** (`DemucsSeparator` writes FLAC with its own
  atomic-write) is a *third* cache outside `cache.py`.
- **Atomic write** is implemented twice (`cached_f0`, DemucsSeparator) and *not* used by
  `CachedItem.save`.

**Action:** a single `atomic_savez`/`atomic_write` primitive + one documented keying
convention, adopted by all three. This is where a small stateful `Cache(root)` object
*might* finally earn its keep (it didn't before — see the earlier call) if it removes
`(cache_dir, transform, contour_cache)` threading through the datasets. Design it, then
rerun. Medium effort.

### #9 — `callbacks/` package  ✅ DONE (80c6c26)
`callbacks.py` (184 lines: `Callback` ABC + Console/MLflow/AsyncValidation) should mirror
the package-per-family pattern now established by `losses/`, `datasets/`, `encoders/`,
`decoders/`: `callbacks/{base,console,mlflow,async_validation}.py`. Pure move, low risk,
consistency win. Do it when touching callbacks anyway.

### transforms `_best_device` vs `training/device.py`  ✅ DONE (7fed310)
Genuine duplication, but the naive "import `device.best_device`" doesn't work:
`training/device.py` imports torch at module top, while `transforms._best_device` guards
the import (keeping the light audio path torch-free). The *transforms* version is the
more robust one.

**Action:** promote device selection to a top-level **`stemify/device.py`** with the
lazy-import guard, and have both `training/device.py` (re-export) and `transforms` call
it. Resolves the dup without pulling torch into the light path. Tiny effort.

### #4 — Explicit `frame/note/sequence/track` naming  ✅ DONE (7fed310)
Agreed on altitude. Clearest win: **`Features` → `FrameFeatures`** (it's CQT-per-frame).
Audit the rest — `Cues` (frame-level, fine as a domain term), `Contour` (frame-level
pitch, established term), `*Example`/`*Batch` already say track/sequence. Mostly a
rename sweep; do it in one pass to avoid churn dribble. Low risk. Also update
`features.py`'s **stale status docstring** ("the CNN that does not exist" — it exists now).

### `decoders/frames.py` relocation (the real #8)  ✅ DONE (7fed310) — see #8 disagreement below
The nearest-neighbour grid resampler (`resample_mask`, `_nearest_frame`) is the actual
shared grid primitive, and it's oddly parked under `decoders/`. Move it to a neutral
top-level home (`stemify/grids.py`, or fold into `metrics`/`features`) so `labels.py`
and decoders both use it without a decoders dependency.

---

## Tier 3 — later / small

### #3 — `examples.py` "archaic"  ✅ DONE-ish
The cache split already slimmed it to `build_example` + `prepare_examples`. Remaining
smell: `build_example`'s per-head `if "onset"/"voicing"/"pitch"` chain. Replace with a
**target-head registry** (`{head: builder_fn}`) mirroring `encoders/heads.py`. Low
priority, small.

### Eval-harness redesign; delete `segmentation/`  ✅ DONE (cf7ceeb)
Note-eval now scores a `Transcriber` directly -- **no `LearnedBaseline` class**; the
trained model *is* a `Transcriber(LearnedEncoder(ckpt), decoder)`. `segmentation/`
deleted; coverage migrated to `tests/decoders/test_group_contour.py`. Dropped the
`Transcriber` Protocol / native adapter (no baseline implements `predict_notes` --
would be dead code; add when one does). Frame harness + `baselines/` left as the
pitch-tracker research tool.

### parallel.py reuse  ✅ DO — but the answer is **machine-dependent**
Natural `parallel_map` candidates: the `Cache.ensure_features` and `Cache.contour`
prep loops (embarrassingly parallel per track).

The original caveat — "pyin already parallelizes its FFTs internally, so a pool only
oversubscribes" — is **true on the M3 Max and false on the Linux box**, because it is
a property of the numeric backend, not of pyin:

| box | backend | pyin cores | pool speedup |
|---|---|---|---|
| M3 Max (macOS) | Accelerate (multithreaded) | ~all | ~none (oversubscribes) |
| 5090 (WSL2, 2026-07-29) | OpenBLAS + scipy pocketfft | **1.1 of 16** | **4.6–4.9x** |

Measured on the Linux box: 7 tracks serial 83.5 s vs 18.0 s with `workers=8`
(4.64x); in the live cache build, 7.4 contours/min serial vs 36.0/min with
`workers=12` (4.9x), and 465 contours in 7.7 min (8.1x) once the serial loop was
mostly hitting cache. scipy's FFT is single-threaded unless passed `workers`, and
nothing sets it — so on any non-Accelerate backend pyin is effectively serial.

So keep the loops serial *by default* (correct on macOS) but do not treat the
caveat as settled: **measure on the box you are on**. The win is available without
touching library code — a script calling `Cache.contour` under `parallel_map` fills
the same content-addressed keys, and `atomic_write` makes concurrent writers safe.

### segmenter.py docs  ✅ DO (trivial)
Add the module/class docstring; it's the one composition point without prose.

---

## Where I disagree

### #2 — Convert to Google/NumPy docstrings (Args/Returns/Raises)  ❌ DISAGREE (mostly)
The codebase's **narrative docstring style is a strength, not a gap.** The docstrings
explain *why* a decision was made (why reference-grid wins, why features live outside
the npz, why a START state) — information a type signature can't carry and a reader
genuinely needs. Converting wholesale to `Args:/Returns:` would restate types the
signature already gives and *dilute* the rationale prose. Net negative.

**Narrow agreement:** add explicit **`Raises:`** notes where an exception is part of the
contract and currently implicit (`KeyError` in `DemucsSeparator`, the `NotImplementedError`
sentinels in `baselines/base.py`, `SemiCRF`/gold shape guards). Callers can't see those
from the signature. So: keep the narrative style, add `Raises:` where it's real. Don't
mechanize the rest.

### #8 — Move `alignment.py` to root to reuse in `labels.py`  ❌ DISAGREE
They do **different operations**. `alignment.py` is *voicing-aware Hz contour resampling*
(`mir_eval.melody.resample_melody_series`) — interpolating pitch onto the reference grid
for eval. `labels.py` needs *nearest-neighbour mask/target resampling* (no interpolation
— you must not invent half-voiced frames or smear an onset). That primitive already
exists as `decoders/frames.py::resample_mask`. So the reuse target isn't `alignment.py`;
it's `frames.py` (relocate it, Tier 2). `alignment.py` stays eval-specific.

---

## Suggested execution order
1. **#5** metrics → `stemify/metrics/` (unblocks the rest of the eval story)
2. **#7** split `semicrf/` (highest risk reduction; independent)
3. **6/6a** `pitch/` module + un-hardcode `LearnedEncoder`
4. **device.py** unify + **frames.py** relocate + **#4** `FrameFeatures` (small, batchable) ✅ DONE (7fed310)
5. **#9** `callbacks/` package
6. **#1** cache unification (design, then rerun on 5090)
7. **eval-harness redesign** → `LearnedBaseline`, then delete `segmentation/`

---

## Separate track — semi-CRF throughput → **[gpu-throughput-plan.md](gpu-throughput-plan.md)**

A measured plan (2026-07-29) to take the semi-CRF epoch from ~3.5 h to ~3 min. Not a
code-quality refactor, so it lives in its own doc, but it touches the same files as #7
and reorders some of the work above: excerpt training, hoisting the segment potentials
out of the DP's inner loop, whole-loop CUDA-graph capture, and a batched Viterbi for
validation.

Two findings there correct things believed elsewhere in the repo's history: the
segmental DP is **launch-bound, not CPU-bound** (an earlier sweep stopped one notch below
the CPU/CUDA crossover), and **`gold_score` — not the DP — is the largest single cost**
at 7-70x the graphed DP step, because it loops in Python over gold segments.

---

## Encoder owns its own cues — the `encode_features` inversion (opened 2026-08-04)

**The defect.** `encoders/base.py` states the rule plainly: the encoder *produces* the
pitch contour rather than receiving it, "never in as a parameter", with pyin memoisation
"behind the encoder, not in its signature". `LearnedEncoder.encode_features(contour,
features)` breaks both halves of that sentence, and `encode()` is implemented in terms
of it — so the contract-compliant method wraps the contract-violating one.

**Why it is not cosmetic.**

1. `encode_features` is **not on the `Encoder` ABC** — `ClassicalEncoder` has only
   `encode`. The validator calls `encode_features`, so its `SegmentFn`/`EncodeFn` types
   hard-code the leaked signature and **the in-training validator cannot score a
   `ClassicalEncoder` at all.** The DSP-floor-vs-learned comparison that `base.py`
   advertises as "a change of one composed object" is unreachable from validation, so
   the floor cannot be tracked as a live baseline during a run.
2. **Dead work.** With a pitch head, `encode_features` *discards* the contour it is
   handed (`if pitch_logits is not None and not self.use_pyin: contour = ...`). The
   validator computes pyin for 40 tracks x 3 conditions on every cold cache, then
   throws it away. Warm cache makes it ~seconds; a fresh box or a new val set pays in
   full, on the critical path before training starts.
3. The comment at `validation.py:48` claims the shape keeps the validator
   "model-agnostic". True of the *model*; it is welded to `LearnedEncoder`'s signature.

**The shape.**

- `LearnedEncoder(model, device, cache=None)`; `encode(sources) -> Cues` memoises the
  CQT and computes pyin **only when it has no pitch head**. `Cache` is already
  content-addressed on the resolved audio path, which is exactly the memoisation
  `base.py` describes.
- validator holds `Sources`, calls `encode`; `SegmentFn` becomes
  `Callable[[Sources], list[NoteEvent]]`, `EncodeFn` `Callable[[Sources], Cues]`.
- `encode_features` goes private or disappears. ~10 test call sites move.

**Explicitly not the fix:** making `Cues.contour` optional. It was made required
deliberately in the mypy pass because no encoder can produce cues without a pitch
source, and that reasoning still holds — every `encode` yields one, from a head or from
pyin. Only the *justification* written in `cues.py` is stale (it cites
`encode_features` taking a required `Contour`, a method `ClassicalEncoder` lacks).

**Sequencing.** Not mid-run: it changes scoring semantics, which would make
epoch-to-epoch F1 non-comparable *within* a run. Do it once the accumulation run has
reported.

### Smaller items found alongside

- **`--num-workers` is silently ignored by the semi-CRF loader.** `_loader` hardcodes
  `num_workers=0` (legitimately — chunks must arrive in per-stream order), but the flag
  is still accepted and does nothing, while the comment above it describes workers
  owning disjoint slices. Same family as `--pos-weight` being accepted and ignored:
  either wire it or refuse it.
- **`pitch_source` was computed and never read.** ✅ DONE — now in the per-epoch
  validation report and as an MLflow param. It decides ~5 F1 and the two reports were
  otherwise identical in shape.
- **Per-frame NLL normalisation / the loss is dimensionally malformed.** `NLL + w *
  pitch_ce` adds a per-chunk *total* to a per-frame *mean*; measured gradient ratio into
  the shared trunk is ~1200:1 at init, so the pitch term contributes ~0.08%. Two
  principled repairs converge: divide the NLL by scanned frames (rate form), or make
  `pitch_ce` a sum (joint-likelihood form, where `w = 1` is forced rather than chosen).
- **`path` is blind to pitch *distance*.** `1 - Bhattacharyya` is ~1 for a semitone step
  and for an octave leap alike, so the CRF cannot use interval structure. A cumsum of
  frame-to-frame |Δ expected bin| is additive, O(1) per span, and adds "how far did
  pitch move" to the existing "did it move". Considered and **rejected**: folding pitch
  into the CRF label/likelihood — a joint likelihood allocates capacity by entropy and
  216 pitch bins would swamp the segmentation, inverting the current imbalance.
