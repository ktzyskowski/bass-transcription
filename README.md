# stemify-ml

Bass tablature transcription: audio in, tab out.

Two stages. A frame-level pitch detector (CQT + CNN, monophonic — bass is
effectively single-note) produces a pitch sequence; a separate fingering stage
maps that sequence to `(string, fret)` via dynamic programming over a cost
function penalizing fret distance, string changes, and hand-position shifts. The
fingering stage is deterministic post-processing rather than learned, so it can
be evaluated on its own against ground-truth pitch.

## Setup

```sh
uv sync
```

Then fetch a dataset — see [docs/datasets.md](docs/datasets.md). Start with
IDMT-SMT-Bass:

```sh
mkdir -p data/raw && cd data/raw
wget "https://zenodo.org/records/7188892/files/IDMT-SMT-BASS.zip?download=1" -O IDMT-SMT-BASS.zip
unzip IDMT-SMT-BASS.zip -d idmt_smt_bass && rm IDMT-SMT-BASS.zip
```

## Layout

```text
src/data/     dataset loaders and feature extraction
docs/         dataset provenance, licensing, download instructions
data/         downloaded datasets (gitignored)
```

## Licensing

Datasets carry their own terms, and at least one of them (IDMT-SMT-Bass) is
non-commercial and unresolved. Read [docs/datasets.md](docs/datasets.md) before
assuming anything here can ship.