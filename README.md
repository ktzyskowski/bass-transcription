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
src/stemify/
  contour.py      Contour — the pitch-over-time type both sides speak
  data/
    base.py       Dataset / EvalItem — what a dataset must look like
    *.py          one loader per dataset, each owning its own references
  baselines/
    base.py       PitchBaseline — predict_f0 and/or predict_pitch
    *.py          off-the-shelf estimators to beat
  evaluation/
    run.py        Evaluation — binds dataset + baseline + metrics
    alignment.py  resampling onto the reference grid; FrameSet pooling
    report.py     Report — results as data, renderable and serialisable
    metrics/      Metric.calculate(true, pred) -> MetricResult
    harness.py    CLI
docs/           provenance, licensing, AWS runbook
data/           downloaded datasets (gitignored)
```

The evaluation pipeline is one sequence, and each stage belongs to exactly one
owner:

```text
dataset  -> reference contour    the Dataset (never the runner)
baseline -> estimate contour     the PitchBaseline
         -> align onto ref grid  alignment.align
         -> pool across items    FrameSet
         -> score                the Metrics
         -> Report               render() or write_json()
```

Adding a dataset is a `Dataset` subclass plus one line in `harness.DATASETS`.
Adding a metric is a `Metric` subclass. Neither touches the other.

```sh
uv run python -m stemify.evaluation.harness --baseline pyin --dataset idmt
uv run python -m stemify.evaluation.harness --baseline pesto --dataset slakh --source mix
```

## Licensing

Datasets carry their own terms, and at least one of them (IDMT-SMT-Bass) is
non-commercial and unresolved. Read [docs/datasets.md](docs/datasets.md) before
assuming anything here can ship.
