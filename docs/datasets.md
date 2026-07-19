# Datasets

Datasets live under `data/`, which is gitignored — nothing here is checked in, so
each machine downloads its own copy.

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

**No scriptable fetch** — the download sits behind an email form at
<https://synthdatasets.weebly.com/mdb-bass-synth.html>. Request it, unpack under
`data/raw/mdb_bass_synth`, then verify the layout:

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

## Planned

Not yet downloaded — see the staged data plan.

- **IDMT-SMT-Bass-Single-Track** — 17 continuous basslines with onset/offset/pitch
  plus string/fret ground truth. The bridge dataset: exercises note segmentation
  and the full DP fingering pipeline end-to-end.
- **Slakh2100** — 2100 multi-track songs, aligned MIDI, isolated bass stems and
  full mixes. Synthetic and pitch-only (no tab). For scale and full-mix robustness.
- **DadaGP + synthesis** — bass tracks from the Guitar Pro corpus (real string/fret
  data) rendered through a bass VST, SynthTab-style, for large-scale (audio, tab)
  pairs.

Check each one's license independently. The IDMT-SMT-Bass situation says nothing
about theirs.

## Licensing summary

| Dataset | License | Commercial use | Derivative works (weights) |
|---|---|---|---|
| IDMT-SMT-Bass | CC BY-NC-ND 4.0 (assumed — contradictory, see above) | ✗ | ✗ |
| MDB-bass-synth | CC BY-NC 4.0 | ✗ | ✓ |
| Slakh2100 (planned) | CC BY 4.0 | ✓ | ✓ |

**Everything currently on disk is non-commercial.** That is fine for research and
architecture validation, which is all this repo does today, but it means no model
trained on the current data can ship in a commercial product.

Slakh2100 is the only source on the roadmap that is commercially clean, which is a
second reason to reach it beyond scale. If a commercial path matters, the options
are: train the shipping model on Slakh and/or self-rendered DadaGP audio and use
the NC datasets only for evaluation and architecture selection, or get written
clarification from the rights holders. Decide which before, not after, a model
becomes load-bearing.
