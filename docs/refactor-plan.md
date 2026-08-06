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

---

## Layering: `core/`, and the protocol seams between layers (opened 2026-08-05)

External review of `docs/api-tree.txt` landed three notes: `training/` is large enough to
be its own library; `semicrf` is split *horizontally* across `decoders/` and `training/`
and wants to be one vertical slice; `training/` has slightly too many responsibilities.
Their tl;dr — **the risk is continued growth, not poor organisation** — is right, and the
measurements below say the fix is much smaller than the diagnosis suggests.

### The measurement that shapes everything

Classifying every module into a layer and computing cross-layer imports: **`core` has
zero upward edges.** Not one module in the proposed bottom layer imports `training`,
`semicrf`, `evaluation` or `serving`. The bottom layer is already a bottom layer; it has
never been named.

So this is not a rewrite. The whole programme is **12 edges**, all in the top three
layers, plus a relabelling.

| layer | lines | share | owns |
|---|---|---|---|
| `core` | 4,406 | 36% | runtime protocols, data contracts, DSP, torch-free reference impls |
| `models` | 961 | 8% | TCN, CueNet, heads, `LearnedEncoder`, `NeuralPitch` |
| `training` | 2,870 | 24% | extension protocols (`Recipe`/`Loss`/`Batch`), Trainer, callbacks, frame recipe |
| `semicrf` | 2,962 | 24% | the vertical slice (`decoders/semicrf` + `training/semicrf` merged) |
| `evaluation` | 705 | 6% | note + cues harness |
| `serving` | 284 | 2% | inference CLI |

Nothing dominates; compare today's `training` at 39%. `evaluation/` and `serving/` are
the composition roots — they do not need an `app/` wrapper, they need to stop importing
implementations.

**Two protocol sets, two homes.** `core` owns what *inference composes* — `Encoder`,
`Decoder`, `PitchEstimator`, `AudioTransform`, `Dataset`/`EvalItem`, the metrics, plus
the new `Reconstitutable` and the registries. `training` owns what an *extension
implements* — `Recipe`, `Loss`, `Batch`. No overlap, and the division answers "which
protocol goes where" without case-by-case argument.

### The 12 edges, grouped by the fix that removes them

**A — `Checkpoint` split + registries (5).** `Checkpoint` holds `recipe: Recipe` as a
*field*, so `load()` calls `RECIPES[name].from_config(...)`: you cannot read the file
without the full recipe table, i.e. without importing semicrf's loss, dataset and
loaders just to answer "what notes does this wav have". Split it:

- `core/checkpoint.py` — `CheckpointFile`: `version`, `recipe_name: str`,
  `recipe_config: dict`, `model_state`, `objective_state`. Pure data; `load` and the v1
  migration become dict normalisation with no torch module built.
- Reconstitution needs only 2 of `Recipe`'s 9 methods — `build_model()` and
  `decoder_from()` are the only ones `Checkpoint` ever calls. `core` declares that
  narrow half as a `Reconstitutable` Protocol; `Recipe` satisfies it structurally.
- `Transcriber.from_checkpoint(path)` looks up `BUILDERS[recipe_name]` and composes,
  never naming an objective. Same registry pattern retires
  `evaluation/cli.py`'s `DECODERS = {"heuristic": ..., "semicrf": ...}`.

Removes: `evaluation.cli → training.checkpoint`, `serving.cli → training.checkpoint`,
`evaluation.cli → decoders.semicrf`, `serving.cli → decoders.semicrf`,
`training.recipes → training.semicrf` (registration replaces import).

*Write stays in `training`:* `save()`'s guard validates state against
`recipe.build_model()`, and that guard exists for a bug that actually shipped.

**B — preprocessing was never training-specific (4).** The reviewer's third note asked
for a `training/preprocessing/` group. The better cut is that `labels`, `conditions`,
`condition_types` and `examples` belong in `core`: `conditions.py` says so in its own
docstring ("what the *evaluation* harness varies too, where objectives do not apply at
all"), `evaluation.cues` already imports `labels`, and `scripts/`+`tests/` reach for
`training.conditions` 16 times. A subpackage inside `training` would move the seam
rather than remove it — the chain is `transforms → cache → features → labels →
conditions`, and three of those five are already in `core`.

Removes: `evaluation.cues → training.labels`, `semicrf.recipe → training.conditions`,
`semicrf.tracks → training.conditions`, `semicrf.tracks → training.examples`.

**C — recipes contribute their own CLI flags (2).** `training/cli.py` imports
`ONSET_WEIGHT`/`PITCH_WEIGHT`/`VOICING_WEIGHT` from `semicrf.recipe` and `AUX_TERMS`
from `semicrf.terms`, *by name*. This is what stops semicrf being a template: extension
#2 would also require editing the framework's CLI. Until a recipe can contribute its own
argument group and config keys, "semicrf is the template for future extensions" is not
true whatever directory it lives in.

**D — encoder assembly belongs to `core/models` (1).** `SemiCRFRecipe.build_model()`
constructs a `FrameRecipe` and calls *its* `build_model()` ("Same encoder; only the
objective differs"). A plugin instantiating a sibling plugin. Building a `CueNet` from
`(conditions, pitch)` is a models concern; extract it and semicrf stops knowing `frame`
exists.

**Kept on purpose:** `semicrf → training.base` (`Recipe`, `Loss`, `config_field`) and
`semicrf → training.config` (`TrainConfig`). Plugin→framework is the direction the arrow
should point. After A–D those three are the *only* edges out of semicrf, which is the
reviewer's "entirely disconnected" in the sense that matters.

### The rule that makes the migration free: move code, never strings

`cache.content_key` is SHA-256 over stringified, NUL-joined parts, and the parts are
`AudioTransform.name`, `features.SIGNATURE` and resolved audio paths — **not** module
paths, class locations or package structure. Condition tags (`"clean"`, …) live in
checkpoint configs and MLflow params and resolve from anywhere.

Verified alongside: `FrameRecipe.config` and `SemiCRFRecipe.config` return primitives
only (`dur_edges_s` goes through `.cpu().tolist()`), and `Checkpoint.save` writes
`{version, recipe:{name, config}, model_state, objective_state}`. **No checkpoint on
disk pickles a module path**, so every existing checkpoint loads unchanged after the
move.

So: every file can move with zero cache invalidation, zero checkpoint breakage and zero
backfill, *provided no string changes*. A legacy layer becomes necessary only where a
rename is separately and deliberately chosen — never as fallout from relocation.

### `core/legacy.py`

Two migrations already exist and share a house style — `Checkpoint._from_v1` and
`serde.migrate_example_cache` — both versioned or idempotent, both logging what they
inferred, both documenting what lets them be deleted. Generalise into one module:

    CHECKPOINT_MIGRATIONS: dict[int, Callable[[dict], dict]]   # chained, v -> v+1
    CONDITION_ALIASES: dict[str, str]                          # old tag -> new tag
    RECIPE_ALIASES: dict[str, str]                             # old objective -> key

One rule per entry: the date it was added, and the condition that lets it be dropped.
This is cheap *because of* the registries in A — alias resolution is a lookup at the
registry boundary (`BUILDERS[RECIPE_ALIASES.get(name, name)]`), not a parallel compat
layer. First inhabitant is `migrate_example_cache`, which is already documented as a
droppable one-off and is imported only by `tests/training/test_serde.py`.

**Explicitly not in scope:** shim modules re-exporting old import paths. The Python API
has one consumer (this repo, three console scripts); shims would be compatibility with
ourselves, plus permanent ruff/mypy ignores. The artifacts that genuinely cannot be
regenerated are checkpoints, the feature/contour cache, MLflow history and the commands
pinned in `ablation-plan.md` — all of which the rule above already protects.

### Modernisation that rides along

- **Protocols over ABCs, selectively.** Under structural typing a plugin satisfies
  `Decoder` *without importing core's base class*, which is what "strict protocols
  between layers" actually buys. Convert the pure interfaces — `PitchEstimator`,
  `Encoder`, `FrameMetric`, `NoteMetric`. Keep ABCs where there is real shared
  behaviour: `Decoder.decode_batch`, `AudioTransform.apply`/`cache_path`, `Recipe`.
  Cost: no `isinstance` (`@runtime_checkable` checks method *names* only) — grep for
  type branching before converting each one.
- **PEP 695 generics.** `class Loss[B]`, `class Trainer[B]` instead of `Generic[B]` +
  `TypeVar`. Three sites, 3.12, free.
- **Rejected: pydantic for config.** `Recipe.from_config` already exists per recipe and
  is the correct seam; adding pydantic solves a problem it would first create (a second
  config idiom beside `TrainConfig`, plus a dependency).

### Also surfaced, not layering

- **The cues → `AlignedPairs` adapter exists twice** — `training/validation.py:156` and
  `evaluation/cues.py:45-62` both build it and call `score_cues`. The metric is shared
  (`core/metrics`); the adapter is not. Inputs genuinely differ (cached npz vs `EvalItem`
  audio), so this is one small constructor in `core`, not a harness merge. Worth doing
  for the reason `checkpoint.py` already records: "two implementations of this is what
  let them disagree."
- **`sagemaker` and `boto3` are declared in `pyproject.toml` and imported nowhere in
  `src/`.** Unrelated; flagged where it was found.

### Sequence

    B (moves) -> D (extract) -> A (registry) -> merge semicrf -> C (CLI flags)
    ^^^^^^^^^                                   ^^^^^^^^^^^^^^
    done e7f367c                                done e7f367c (brought forward: it is
                                                pure file movement and needed no
                                                registry work first)

B and D are pure moves with no design content, so they go first and shrink everything
downstream. A is the design work. C is last because it is the only step that changes
user-facing CLI behaviour.

**Remaining, in order.** All three are enforced rather than described: `PENDING` in
`tests/test_layers.py` names the layer edges and may only shrink, and
`test_semicrf_touches_only_the_training_contracts` pins D's single edge.

1. **D — encoder assembly into `models`** (1 edge). `SemiCRFRecipe.build_model()` builds
   a `FrameRecipe` to reuse it. Now that `models` exists this has an obvious home:
   a `build_cuenet(conditions, pitch=..., harmonics=...)` factory both recipes call.
   Smallest of the three, and it removes the only plugin-to-plugin edge.
2. **A — `CheckpointFile` split + registries** (5 edges). The design work. Note the
   correction above: the reconstitution protocol returns `Encoder`/`Decoder`, never
   `CueNet`.
3. **C — recipes contribute their own CLI flags** (2 edges). Changes user-facing CLI
   behaviour, so it goes last.

The acceptance test for all three together: **adding a new objective touches one
package.** Today it touches four files across three layers -- the new package,
`training/recipes.py`, `training/cli.py` (twice), and `evaluation/cli.py`'s hardcoded
`DECODERS`.

### Blocking the test-set eval: the note harness is single-channel (found 2026-08-05)

Unrelated to the layering, found while trying to score `tcn-v3-hcqt` on Slakh test.
`NoteEvaluation._estimate_notes` calls `transcribe(Sources(bass=path))` -- it never
populates `mix`, so a 2-channel checkpoint raises `this track has no 'mix' audio` on the
first item. `Sources.role` refusing to return `None` is correct and is what surfaced it;
the gap is that the harness predates bichannel models.

Consequence worth stating plainly: **no bichannel arm has ever been scored by the note
harness.** Every bichannel number on record comes from the training-time validator in
`training/validation.py`, which resolves conditions and channels properly. That is the
same "DONE is only against the pipeline it ran on" trap as before, and it means
bichannel results are not directly comparable to the mono test-set numbers in
`experiments.md`.

The fix is small -- `EvalItem.sources()` already exists and Slakh's `BassStem`
implements it, so the harness should use the item's own `Sources` rather than
synthesising a bass-only one. But it changes what the note harness *measures* for
multi-channel models, so it wants deciding rather than patching in passing.

**Not while a run is in flight.** The venv is an *editable* install, so a live
`stemify-train` resolves `stemify.*` from `src/` as it goes; relocating modules under it
risks a crash on any not-yet-imported module and on dataloader worker respawn. Costs
hours of GPU; waiting costs minutes.

### Rejected shapes

- **Three top-level packages** (`src/stemify`, `src/training`, `src/semicrf`, or
  `stemify_training`/`stemify_semicrf` as separate distributions). The reason to split
  distributions is dependency separation, and it does not hold here: `torch` is already
  in the bottom layer (`models/`, `encoders/`, `pitch/neural`, `device`), `demucs` is
  already a lazy import inside `transforms.py`, and `mlflow` — one file — is the only
  genuinely training-only dependency. A split would buy isolation from one dependency
  while making `scripts/`+`tests/` deep imports (16× `training.conditions`, 12×
  `training.checkpoint`, 11× `training.validation`, 6× `decoders.semicrf.gold`) into
  cross-package reaches into privates. Revisit only if something outside this repo
  installs one piece without the other. The layout above is the shape it would split
  along, so nothing is wasted.
- **Three top-level dirs in one wheel.** Enforces nothing — nothing stops `stemify/`
  importing `training/`. The DAG is enforced by a contract check (`import-linter`, or
  ~30 lines in `scripts/check.sh`), which works today on the current nesting.
- **`app/` as a composition root.** Puts the coupling somewhere instead of removing it;
  A removes it.
- **`models/` moving up with `training`.** `learned_encoder.py` and `neural_pitch.py`
  hold a `CueNet` to run *inference* -- both under `@torch.no_grad()` on a model in
  `.eval()` -- and `stemify-transcribe` needs them, so the layering would invert on its
  first real use. Autograd belongs to `Recipe`/`Loss`/`Trainer`; an `nn.Module` is not
  autograd. **But it does not stay in `core` either** -- see below.

### `models` as its own layer (done, 2026-08-05)

`core` held the nets because nothing above it could. The third option is better than
either: `models` between `core` and `training`, taking `LearnedEncoder` and
`NeuralPitch` with it since both exist only to wrap a `CueNet`. What makes it worth the
move is the property that falls out -- **`core` now imports no torch at all.** Exactly
seven files in `core` imported it and they were exactly the set that left.

That turns the boundary from a convention into a tripwire. `tests/test_layers.py`
asserts, in a subprocess walking every submodule, that importing `stemify.core` leaves
`torch` out of `sys.modules`; the same file holds the layer DAG as an allow-list with
the still-outstanding edges named in a `PENDING` set that may only shrink. A contract
nobody can breach by accident is worth more than a docstring asking them not to: the
failure being guarded is one convenient import, which reads as a one-line diff.

Two things the move flushed out, neither of which was visible before the test existed:

- `DemucsSeparator` probed `best_device()` in `__init__`, and `core.conditions` builds
  its registry at import time -- so merely importing the conditions table loaded torch,
  for every caller including the ones that never separate anything. The device is now
  resolved on first use. `best_device` itself stays in `core`: it already guarded its
  torch import inside the function, and its docstring already said why.
- **It corrects step A's protocol.** `Reconstitutable` was sketched as returning
  `CueNet`, which now lives in `models` -- so `core` naming it would be an upward edge.
  The contract must return core ABCs (`build_encoder(...) -> Encoder`,
  `decoder_from(...) -> Decoder`) instead, which is better protocol design regardless:
  the original leaked a concrete implementation class into a contract whose whole
  purpose is to avoid naming implementations.

### Second external review (2026-08-05), triaged

Eight points, from a flat listing of the tree rather than the code. Two worth acting on,
four worth watching, two that do not survive the measurements.

**Act on.**

- **"score" is overloaded** -- the best point of the eight. It means three different
  things: `semicrf/score.py` is a CRF energy over a segmentation, `core/metrics/*/
  scoring.py` is an evaluation metric, `validation.score_examples` is a validation
  result. Renaming the semi-CRF one (`segment_energy.py`) removes the collision at its
  source, since that is the meaning a reader is least likely to guess.
- **Recipe explosion risk** -- fair, and already the plan: steps A and C exist so that a
  new objective is a single-package change rather than a fifth edit to `training/cli.py`.

**Watch, do not act yet.**

- **`core` as a junk drawer.** Partially fair, but it has a *checkable* definition here
  -- torch-free, zero upward edges -- rather than "everything important". It is 4,406
  lines with real substructure already (`data/`, `decoders/`, `encoders/`, `metrics/`,
  `pitch/`) plus 13 flat modules. The suggested seven-way split (audio/cache/data/
  signal/domain/io/transcription) is too many buckets for 13 files, and it directly
  contradicts the same review's next point.
- **Depth.** Real, and partly self-inflicted: `core/` added a level to everything.
  Mitigated by the package `__init__`s -- `from stemify.core.metrics import score_notes`
  is the call site, not the five-segment path. In tension with the point above; you
  cannot have both fewer junk drawers and shallower paths.
- **Frame/note duplication.** `metrics/frame` and `metrics/note` are not duplicates --
  frame AUROC and note-level PRF against mir_eval matching are different measurements.
  The one real instance is already recorded above: the cues -> `AlignedPairs` adapter
  existing in both `validation.py` and `evaluation/cues.py`.
- **`models` mixes architectures and wrappers.** True -- `CueNet` is a system, `TCN` and
  the heads are building blocks, `LearnedEncoder`/`NeuralPitch` are pipeline stages. But
  that is 961 lines in 7 files; four subpackages for seven files buys nothing. Revisit
  if a second architecture lands.

**Does not survive measurement.**

- **"Some modules are getting large -- I suspect 500-1000 lines."** Measured: the
  largest file in `src/` is `validation.py` at 377. `trainer.py` 345, `transforms.py`
  ~225, `features.py` 171. The estimate is 2-3x high; no action.
- **Implementation-oriented names in `semicrf`.** `forward.py` holds the forward
  algorithm -- the standard name in the literature -- and the proposed
  `dynamic_programming.py` is *less* precise, since Viterbi is dynamic programming too
  and lives in its own file. `window.py` is the rolling alpha buffer; the proposed
  `beam_window.py` would be actively wrong, as nothing here does beam search. A terse
  correct name beats a descriptive incorrect one.
