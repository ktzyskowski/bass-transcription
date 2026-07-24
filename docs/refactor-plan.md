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
- `gold.py` — `gold_segments` (the audible-gap logic — test-dense, bug-prone).
- `decoder.py` — `SemiCRFDecoder` façade wiring them + `segs_to_notes`.

Each file gets its own focused test module. No behavior change; pure decomposition.
Biggest risk-reduction in the list. Medium effort, low risk (tests pin behavior).

### 6 / 6a — A `pitch` abstraction; stop `NeuralEncoder` hardcoding `PyinBaseline`  ✅ DONE (009e777)
Real coupling: `NeuralEncoder.__init__` defaults `baseline=PyinBaseline()` and calls
it for the contour, and `PyinBaseline` lives in `baselines/` (the *eval* package). So a
production model depends on an eval baseline — role confusion (your 6a).

**Precise scope — I partly disagree with the framing:**
- **YES** a `stemify/pitch/` module with a `PitchEstimator` (a.k.a. `ContourSource`)
  ABC: `estimate(audio|features) → Contour`. Impls: `PyinPitch` (wraps the pyin call),
  later `NeuralPitchHead`. `NeuralEncoder` takes a `PitchEstimator`, not a
  `PyinBaseline` — the learned head then drops in with no interface change (exactly the
  "encoder produces the contour" promise).
- **NO separate `onset`/`voicing` ABCs** — those are *outputs of the `FrameTagger`
  heads*, not pluggable strategies. Abstracting them would be ceremony around a tensor.
  Pitch is different: it has two genuine sources (pyin now, a head later), so it earns
  the seam. Onset/voicing have one source (the head).

`baselines/PyinBaseline` then becomes a thin eval-harness wrapper over `pitch.PyinPitch`
(or is defined in terms of it), so `baselines/` returns to meaning *only* "things the
harness scores." Medium effort, low risk. Depends on nothing; pairs naturally with #5.

---

## Tier 2 — worthwhile, independent

### #1 — Unify the cache  ⚙️ PART A DONE (885cecc); PART B queued for the 5090 rerun
**A (done):** `stemify/atomic.py` (shared `atomic_write`) + `stemify/serde.py` (formats;
`TrackExample` moved here) + `cache.py` as pure policy. Fixed the non-atomic
`TrackExample.save` corruption bug. **B (queued):** content-address features by
`(cqt, resolved_audio_path)` to match `contour`/`separation`; delete the `cache_dir/tag`
partitioning in `train.py`; update the `validation.py`/`sequence.py` readers. B changes
every features key → full rebuild, so land it **immediately before** the 5090 rerun, not
before.

Original design notes:

`cache.py` (just landed) is the seam. Remaining fragmentation, now visible:
- **Two keying schemes:** features key on `(name, transform)`; `cached_f0` keys on
  `sha1(baseline, resolved_audio_path)`. Pick one. Since you'll rerun on the 5090,
  we're free to change keys.
- **The transforms/Demucs cache** (`DemucsSeparator` writes FLAC with its own
  atomic-write) is a *third* cache outside `cache.py`.
- **Atomic write** is implemented twice (`cached_f0`, DemucsSeparator) and *not* used by
  `TrackExample.save`.

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

### transforms `_best_device` vs `training/device.py`  ✅ DO (your call is right)
Genuine duplication, but the naive "import `device.best_device`" doesn't work:
`training/device.py` imports torch at module top, while `transforms._best_device` guards
the import (keeping the light audio path torch-free). The *transforms* version is the
more robust one.

**Action:** promote device selection to a top-level **`stemify/device.py`** with the
lazy-import guard, and have both `training/device.py` (re-export) and `transforms` call
it. Resolves the dup without pulling torch into the light path. Tiny effort.

### #4 — Explicit `frame/note/sequence/track` naming  ✅ DO (batch it)
Agreed on altitude. Clearest win: **`Features` → `FrameFeatures`** (it's CQT-per-frame).
Audit the rest — `Cues` (frame-level, fine as a domain term), `Contour` (frame-level
pitch, established term), `*Example`/`*Batch` already say track/sequence. Mostly a
rename sweep; do it in one pass to avoid churn dribble. Low risk. Also update
`features.py`'s **stale status docstring** ("the CNN that does not exist" — it exists now).

### `decoders/frames.py` relocation (the real #8)  ✅ DO  — see #8 disagreement below
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
Note-eval now scores a `Segmenter` directly -- **no `LearnedBaseline` class**; the
trained model *is* a `Segmenter(NeuralEncoder(ckpt), decoder)`. `segmentation/`
deleted; coverage migrated to `tests/decoders/test_group_contour.py`. Dropped the
`Transcriber` Protocol / native adapter (no baseline implements `predict_notes` --
would be dead code; add when one does). Frame harness + `baselines/` left as the
pitch-tracker research tool.

### parallel.py reuse  ✅ DO (opportunistic)
Natural `parallel_map` candidates: the `cache_features` and `cached_f0` prep loops
(embarrassingly parallel per track). Caveat: pyin already parallelizes its FFTs
internally (see `validation.py`), so measure before stacking a process pool on top.
Minor.

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
3. **6/6a** `pitch/` module + un-hardcode `NeuralEncoder`
4. **device.py** unify + **frames.py** relocate + **#4** `FrameFeatures` (small, batchable)
5. **#9** `callbacks/` package
6. **#1** cache unification (design, then rerun on 5090)
7. **eval-harness redesign** → `LearnedBaseline`, then delete `segmentation/`
