# Datasets

Datasets live under `data/`, which is gitignored — nothing here is checked in, so
each machine downloads its own copy.

Two cross-cutting sections at the bottom are worth reading before trusting any
number produced from this data: [what each dataset fakes](#what-each-dataset-fakes)
and the [licensing summary](#licensing-summary).

## IDMT-SMT-Bass

~4800 isolated bass notes, 44.1 kHz / 24-bit mono WAV. Used to validate the
pitch-detection architecture on clean, unambiguous signal before moving to
continuous basslines.

### Download

Expects `wget` and `unzip`. ~1.6 GB compressed, ~1.7 GB unpacked.

```sh
mkdir -p data/raw && cd data/raw
wget "https://zenodo.org/records/7188892/files/IDMT-SMT-BASS.zip?download=1" -O IDMT-SMT-BASS.zip
unzip IDMT-SMT-BASS.zip -d idmt_smt_bass
rm IDMT-SMT-BASS.zip
```

Verify the download before unzipping if you like — the published MD5 is
`884d77260aaaa3df7557f77a6b3aef18` (`md5 -q IDMT-SMT-BASS.zip` on macOS,
`md5sum` on Linux).

The loader in `src/stemify/data/idmt_smt_bass.py` defaults to nothing and takes the root
explicitly, so the directory name is up to you — but `index("data/raw/idmt_smt_bass")`
is what the rest of the code assumes.

### License — unresolved, treat as non-commercial

The dataset states its license **two different ways**, and they conflict:

| Source | License |
|---|---|
| [Zenodo record 7188892](https://zenodo.org/records/7188892) (v1.0.0, Jan 2023) | CC BY-NC-ND 4.0 |
| `IDMT-SMT-Bass-Description.pdf` in the zip (v1.1, May 2011) | CC BY-SA 4.0 |

**Assume CC BY-NC-ND 4.0** — the restrictive reading, and the terms on the page
we actually download from. That bars commercial use *and* derivative works.

This is fine for research and architecture validation. Before Stemify does
anything commercial with a model trained on this data, get written clarification
from <datasets@idmt.fraunhofer.de>. The ND clause is the sharper risk than NC,
since trained weights are plausibly a derivative work.

### Citation

> Abeßer, Jakob. *IDMT-SMT-Bass Dataset*. Zenodo, 2023.
> DOI: [10.5281/zenodo.7188892](https://doi.org/10.5281/zenodo.7188892)

Underlying papers:

- Abeßer, J.; Lukashevich, H.; Schuller, G. "Feature-based extraction of plucking
  and expression styles of the electric bass guitar." *ICASSP*, 2010.
- Abeßer, J.; Dittmar, C.; Schuller, G. "Automatic Recognition and Parametrization
  of Frequency Modulation Techniques in Bass-Guitar Recordings." *42nd AES
  Conference on Semantic Audio*, 2011.

### What's actually in it

Layout, filename schemas, and the gotchas found by inspecting the unpacked data
are documented in the module docstring of `src/stemify/data/idmt_smt_bass.py`. The short
version:

- **Strings are numbered 1=E .. 4=G** (low-to-high), the reverse of standard tab
  convention. Convert before emitting tab.
- **Plucking and expression are never jointly varied.** `PS/` is all-`NO`
  expression; `ES/` is all-`FS` plucking. There is no 5×5 grid.
- **469 files are exact duplicates** between `ES/NO` and `PS/FS`. `index()` drops
  one copy by default; without that, identical audio lands on both sides of a
  train/test split.
- **Only ~3120 of ~4835 notes have label-derivable pitch.** Harmonics, dead notes,
  bends, and slides aren't pinned down by `(string, fret)`.
- Published counts disagree with reality (Zenodo says ~4300 files, the PDF says
  ~5200; there are 5306 on disk, 4837 unique). Trust the index, not the docs.

## MDB-bass-synth

71 real songs (from MedleyDB) with **perfect frame-level bass f0 annotations**. The
bridge off isolated notes: real recordings and real basslines, but still
frame-level, so it needs no note segmentation and no onset detection.

How the labels can be perfect: the bass stem was analysed, then *resynthesized*
from that analysis and mixed back in. The annotation is the synthesis control
signal, so it describes the audio exactly. The tradeoff is that the bass you hear
is synthetic even though the surrounding song is a real recording.

### Contents

Per track: the full mix (stereo, resynthesized bass in place), the isolated bass
stem (mono), and a CSV annotation.

The CSV is `timestamp,hz` at a `128/44100` s hop (~2.9 ms), first frame
zero-centred, **0 Hz = silence** — already the `mir_eval` convention used
throughout this codebase, so it needs no conversion.

Same labels apply to both the stem and the mix, which makes
`--source stem` vs `--source mix` a clean controlled test of whether the bass
survives a full arrangement.

### Download

**No scriptable fetch, no mirror.** Access is a request form at
<https://synthdatasets.weebly.com/mdb-bass-synth-download.html>, which asks for:

- name
- affiliation (university, company, institute)
- email
- **intended use** — "what you plan to use the dataset for"
- agreement to the terms

The link comes back by email; there is no Zenodo copy to `wget`.

A note on the intended-use field: the license is **CC BY-NC**, and this repo uses
the dataset strictly as a held-out evaluation set for architecture validation —
never as training data for a shipping model (see
[what each dataset fakes](#what-each-dataset-fakes) for the separate technical
reason). Say that. It is both true and consistent with the terms; describing
broader commercial intent would be neither.

Once the archive arrives, unpack under `data/raw/mdb_bass_synth` and verify the
layout:

```sh
uv run python -c "from stemify.data.mdb_bass_synth import index; ts=index('data/raw/mdb_bass_synth'); print(len(ts), ts[0])"
```

The loader pairs each `*.csv` with a `*.wav` sharing its basename. The published
archive's folder and file naming was inferred from the dataset description rather
than an unpacked copy, so if that print fails, fix `_find_audio()` in
[src/stemify/data/mdb_bass_synth.py](../src/stemify/data/mdb_bass_synth.py) — it
raises with a listing of what it actually found.

### License — CC BY-NC 4.0

Non-commercial, but **no ND clause**, unlike IDMT-SMT-Bass. Derivative works are
permitted, so trained weights are not themselves a licensing problem here — the
NC term is. See the licensing summary at the bottom of this file.

### Citation

> Salamon, J., Bittner, R. M., Bonada, J., Bosch, J. J., Gómez, E., and Bello, J. P.
> "An analysis/synthesis framework for automatic f0 annotation of multitrack
> datasets." *ISMIR*, Suzhou, China, 2017.

Also cite MedleyDB itself:

> Bittner, R. M., Salamon, J., Tierney, M., Mauch, M., Cannam, C., and Bello, J. P.
> "MedleyDB: A multitrack dataset for annotation-intensive MIR research." *ISMIR*,
> 155–160, Taipei, Taiwan, 2014.

## Slakh2100

1710 synthesized multitracks (redux release), 145 hours, every mix guaranteed to
contain bass. Audio was rendered from MIDI with 187 sampled instrument patches,
so the MIDI describes the audio exactly and labels are free.

This is the **training** set, and the only source here that is commercially
clean. Two things it uniquely provides:

- **Note-level ground truth** — MIDI carries onsets and offsets, not just a
  contour, so Slakh is what eventually unlocks note segmentation and note-level
  P/R/F. MDB-bass-synth can never do that.
- **Mix and isolated stem for identical material**, which makes "does the bass
  survive an arrangement" and "does source separation earn its place" controlled
  experiments rather than guesses.

### Download

One 104.3 GB tarball, no per-track fetch. MD5 `f4b71b6c45ac9b506f59788456b3f0c4`.
Unpacked as `.wav` it is ~500 GB, so **do not unpack it whole** — stream-filter
to keep only what the bass pipeline needs (stems, MIDI, mixes, metadata):

```sh
curl -L "https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz" \
  | tar -xz -C data/raw/slakh2100 --wildcards \
      '*/metadata.yaml' '*/mix.flac' '*/stems/*.flac' '*/MIDI/*.mid'
```

You still transfer 104 GB either way. Do this on an EC2 instance writing to S3
rather than a laptop — see [aws.md](aws.md).

Layout is `<root>/<split>/TrackXXXXX/`, splits `train`/`validation`/`test`, plus
an `omitted` directory holding tracks whose MIDI duplicates another's (the redux
release moves them aside so the same material cannot land on both sides of a
split). The loader skips `omitted` unless asked for it.

### Gotchas

- **Bass MIDI is not guaranteed monophonic.** It comes from Lakh — real
  transcriptions — which contain octave doubling and overlapping releases. The
  loader collapses overlaps to the lowest sounding pitch and exposes
  `polyphony_rate()`; `SlakhDataset` drops stems above 5% by default, since a
  doubled-octave transcription has no single right answer to score against.
- **Some tracks have more than one bass stem.** Each becomes its own item.
  Mix-based evaluation filters to single-bass tracks automatically, because with
  two basses sounding one reference contour cannot describe the mix.
- **Not every stem in `metadata.yaml` was rendered.** The `audio_rendered` and
  `midi_saved` flags are the dataset's own record of this; the loader trusts them.
- **The reference grid is our choice**, unlike MDB-bass-synth's annotated 2.9 ms.
  `REFERENCE_HOP_S` defaults to 10 ms, the mir_eval melody convention.

### License

**CC BY 4.0** — commercial use and derivative works both permitted. The only
dataset here without a non-commercial restriction.

### Citation

> Manilow, E., Wichern, G., Seetharaman, P., and Le Roux, J. "Cutting Music Source
> Separation Some Slakh: A Dataset to Study the Impact of Training Data Quality
> and Quantity." *WASPAA*, 2019.

## FiloBass

48 jazz standards played by 12 professional upright bassists — Ron Carter, Rufus Reid,
Ray Drummond and others — with manually verified, audio-aligned note transcriptions.
52,558 notes, ~4.9 h. The only **out-of-domain** set here: everything else is isolated
notes, resynthesized bass, or bass rendered from MIDI by a sample library.

### Download

Open access, a single 434 MB zip. `unzip` is not required — `python -m zipfile` works.

```sh
mkdir -p data/raw && cd data/raw
curl -fL -o FiloBass_v1.0.0.zip \
  "https://zenodo.org/records/10069709/files/FiloBass_v1.0.0.zip?download=1"
python3 -c "
import zipfile
z = zipfile.ZipFile('FiloBass_v1.0.0.zip')
keep = [n for n in z.namelist() if not n.startswith('__MACOSX') and '.DS_Store' not in n]
z.extractall('filobass', members=keep)"
mv filobass/'FiloBass ISMIR Publication'/* filobass/ && rmdir filobass/'FiloBass ISMIR Publication'
rm FiloBass_v1.0.0.zip
```

The archive nests everything under a directory whose name contains a space; the `mv`
flattens it, which is what `index()` expects. The macOS resource forks double the entry
count and are worth skipping.

### These stems are `separated`, not clean

**The single most important fact about this dataset for our purposes.** The bassists are
the Aebersold session players; nobody recorded them for the dataset. The releases carry
bass+drums on one stereo channel, and the paper takes that channel through **Demucs** —
the same separator our `separated` condition uses. So a FiloBass number compares against
a model's `separated` score, never its `clean` one; read as clean the transfer gap looks
~10 points worse than it is. The spectrum agrees: ≥80 dB down above 4 kHz.

### Contents

| directory | what |
|---|---|
| `audio_bass_stems/` | 48 × mp3, 44.1 kHz stereo, ~6 min (decodes natively via libsndfile ≥1.1, no ffmpeg) |
| **`midi_fully_aligned/`** | **the only usable MIDI** — note-level, aligned to the audio |
| `midi_downbeat_aligned/` | anchored at bar lines only; onsets drift by most of a beat |
| `midi_from_score/` | notation timing, none of the performance |
| `syncpoints/` | `[[bar_index, seconds], …]` |
| `notebooks_and_scripts/note_data.csv` | per-note chord/degree/measure, and the player |

One file per tune in every directory, sharing an exact stem. **MIDI is at sounding
pitch** — no `BASS_TRANSPOSE`, unlike Slakh.

### No mixture, and it is a paywall rather than a gap

Only stems ship, so a bichannel checkpoint runs here only with a silenced context
channel. The missing audio is `Bass_Drums.wav` — *the exact mixture these stems were
separated from* — which FiloSax's `Compile_Backing.py` rebuilds from **commercially
purchased** Aebersold play-alongs (jazzbooks.com, $17.50/volume, ~25–35 volumes,
MD5-verified so only the exact files work). Obtaining it would make this a true
`bichannel-separated` set.

`Piano_Drums.wav` is the wrong channel and holds no bass at all.

### License — contradictory, treat as non-commercial

| Source | License |
|---|---|
| [Zenodo record 10069709](https://zenodo.org/records/10069709) | CC BY 4.0 |
| [Project page](https://aim-qmul.github.io/FiloBass/) | Non-commercial research only; no redistribution |

Same shape as IDMT-SMT-Bass. **Assume the restrictive reading** — the corpus is built on
copyrighted jazz standards, which is very likely why the two disagree. Fine for research
and evaluation; get written clarification from Xavier Riley (j.x.riley@qmul.ac.uk) before
anything commercial touches it.

### Citation

> Riley, X.; Dixon, S. "FiloBass: A Dataset and Corpus Based Study of Jazz Basslines."
> *ISMIR*, 2023. DOI: [10.5281/zenodo.10069709](https://doi.org/10.5281/zenodo.10069709)

## Planned

Not yet downloaded — see the staged data plan.

- **IDMT-SMT-Bass-Single-Track** — 17 continuous basslines with onset/offset/pitch
  plus string/fret ground truth. Higher priority than its size suggests: it is
  the only source of genuinely real, continuous bass *audio*, which makes it the
  final check that synthetic timbre has not fooled us. Still 17 tracks, so a
  sanity check rather than a benchmark — see
  [what each dataset fakes](#what-each-dataset-fakes).
- **Re-rendering Slakh's own bass MIDI** through several soundfonts. Slakh's
  weakness is timbre: 187 fixed patches. Re-rendering multiplies timbre diversity
  on MIDI whose quality is already vetted, needs no new dataset, and can be
  validated against Slakh's own professional render of the same notes — paired
  ground truth for how much a cheaper pipeline degrades. Do this before building
  any larger synthesis corpus.
- **Lakh Clean MIDI + synthesis** — >17,000 MIDI files, CC BY 4.0, filenames
  giving artist and title. Slakh drew from `lmd_matched` (45,129 files) and kept
  2,100, a 4.7% yield, but only because it required piano *and* bass *and* guitar
  *and* drums for source-separation work. A bass-only filter is far weaker, so
  the yield here would be much higher. Two traps: Lakh contains many duplicate
  transcriptions of the same song, so dedupe and split by *artist*; and rendering
  both bass and accompaniment from the same sample libraries produces mixes that
  Demucs separates unrealistically well, biasing any separation artifacts learned
  from them.
- **Procedurally generated MIDI + synthesis** — unlimited self-labeling data. Its
  value is *distribution control*, not volume: real-song corpora will not
  oversample the low register or slap-thumb, where pyin's octave errors
  concentrate. Weakest option for musical realism, which matters more for a model
  using attention over musical context than for a frame-wise CNN.

Check each one's license independently. The IDMT-SMT-Bass situation says nothing
about theirs.

## What each dataset fakes

**Every dataset here is synthetic on at least one axis.** None of them is ground
truth about real bass in real music, and the useful discipline is knowing which
axis each one fakes rather than picking one to trust.

Three things can independently be real or synthetic: who played it, what the bass
actually sounds like, and what surrounds it in the mix.

| dataset | performance | bass timbre | surrounding mix |
|---|---|---|---|
| IDMT-SMT-Bass | real player | **real recording** | none — isolated notes |
| IDMT Single-Track | real player | **real recording** | none — isolated bassline |
| MDB-bass-synth | real player | resynthesized | **real recording** |
| Slakh2100 | MIDI | sampled library | sampled library |

### MDB-bass-synth is more real than "synth" suggests, and less real than "recording"

The bass you hear is computer-generated — that is the entire point, since it is
what makes the f0 annotation exact. But:

- the **performance** is a real bassist, so pitch trajectory, timing, vibrato and
  dynamics all come from a human take
- the synthesis is **wide-band harmonic modelling** estimating frequency,
  amplitude *and phase* of every harmonic, not a sine sum, so timbre and dynamics
  are largely preserved
- **every other stem in the mix is the untouched original recording**
- the authors validated that pitch algorithms score *statistically
  indistinguishable* on the resynthesized mixes versus the original manually
  annotated ones

That validation is what makes it usable as a reality check. It was established
for evaluation with 2017-era algorithms, though, and a modern neural model could
plausibly latch onto resynthesis artifacts in a way pyin cannot. **Use it for
evaluation; do not train on it.**

### The gap this leaves

Look at the last two columns of the table. **No dataset available to us has real
bass timbre inside a real mixture with reliable f0 labels.** Each satisfies
exactly one:

- MDB-bass-synth — real mixture, synthesized bass inside it
- IDMT — real bass audio, no mixture at all

So "does this work on real music?" has no single-dataset answer, and any claim
has to triangulate. Two consequences:

- **IDMT-SMT-Bass-Single-Track matters more than its size suggests.** 17 tracks
  is far too few for a trustworthy metric, but it is the only source of genuinely
  real, continuous bass audio with real annotations — and the only one carrying
  string/fret ground truth. Treat it as the final "did synthetic timbre fool us?"
  check, not as a benchmark.
- **Closing the gap properly means annotating real songs by hand**, or accepting
  that every reported number carries a synthetic caveat. Better to state the
  caveat than to let a clean-looking number imply it is not there.

### How this shapes dataset roles

| dataset | role |
|---|---|
| Slakh2100 | training — the only source with volume and a clean license |
| MDB-bass-synth | held-out evaluation — never training, per above |
| IDMT-SMT-Bass | unit-test scale sanity checks on clean isolated signal |
| IDMT Single-Track | final real-audio sanity check, not a benchmark |

## Licensing summary

| Dataset | License | Commercial use | Derivative works (weights) |
|---|---|---|---|
| IDMT-SMT-Bass | CC BY-NC-ND 4.0 (assumed — contradictory, see above) | ✗ | ✗ |
| MDB-bass-synth | CC BY-NC 4.0 | ✗ | ✓ |
| Slakh2100 (planned) | CC BY 4.0 | ✓ | ✓ |
| FiloBass | CC BY 4.0 vs non-commercial (contradictory, see above) | ✗ | ✗ |

**Everything currently on disk is non-commercial.** That is fine for research and
architecture validation, which is all this repo does today, but it means no model
trained on the current data can ship in a commercial product.

Slakh2100 is the only source on the roadmap that is commercially clean, which is a
second reason to reach it beyond scale. If a commercial path matters, the options
are: train the shipping model on Slakh and/or self-rendered DadaGP audio and use
the NC datasets only for evaluation and architecture selection, or get written
clarification from the rights holders. Decide which before, not after, a model
becomes load-bearing.
