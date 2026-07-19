# Running on AWS

Staged deliberately: each step is useful on its own, and none of it requires the
next one to exist. Do not reach for SageMaker until repetition justifies it —
an EC2 box and an S3 bucket cover everything up to the first training run.

`boto3` and `sagemaker` are already dependencies, and `requires-python` is pinned
to 3.12 to match the SageMaker PyTorch DLC, so the interpreter that resolves deps
locally is the one that runs training in the container.

## Stage 1 — get Slakh into S3 (one time)

Slakh is a single 104 GB tarball. This is a network-bound job, not a compute one,
so pick an instance for bandwidth and use spot pricing; it is idle CPU the whole
time.

```sh
# c5n.2xlarge (up to 25 Gbps), spot, ~200 GB gp3 root volume.
# Transfer dominates: budget 1-3 hours depending on how Zenodo feels.
aws s3 mb s3://stemify-data

mkdir -p /data/slakh2100
curl -L "https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz" \
  | tar -xz -C /data/slakh2100 --wildcards \
      '*/metadata.yaml' '*/mix.flac' '*/stems/*.flac' '*/MIDI/*.mid'
```

**Prune before uploading.** The filter above keeps nearly the whole archive —
the tarball is already FLAC, so the 500 GB figure quoted around Slakh is for WAV
conversion, which we never do. The real saving is dropping the stems this project
will never read: a track has ~10 stems and the bass pipeline needs one of them
plus the mix.

```sh
uv run python -m stemify.data.prune_slakh /data/slakh2100   # keeps bass stems + mix + MIDI
aws s3 sync /data/slakh2100 s3://stemify-data/slakh2100/
```

That takes roughly 100 GB down to ~30 GB, which is about $0.70/month in S3
Standard. Terminate the instance afterwards; the EBS volume is transient.

Keep the *unpruned* set out of S3 rather than paying to store stems that only
matter if this project later needs other instruments.

## Stage 2 — run the harness on EC2

The evaluation is CPU-bound and embarrassingly parallel across tracks. pyin runs
at ~0.14 s per IDMT note; a full Slakh test split is meaningfully larger, so size
the box by core count and expect to want `--limit` while iterating.

```sh
git clone <repo> && cd stemify-ml
uv sync
aws s3 sync s3://stemify-data/slakh2100/ data/raw/slakh2100/
uv run python -m stemify.evaluation.harness --baseline pyin --dataset slakh --split test
```

Attach an instance role with `s3:GetObject` on the bucket rather than putting
keys on the box.

For the three-condition comparison (clean stem, separated, raw mix) this is also
where Demucs runs — that one *is* GPU-bound, so a `g5.xlarge` for the separation
pass and a cheaper CPU box for scoring is the sensible split. Cache separated
audio back to S3; Demucs is deterministic, so it never needs recomputing for the
same input.

## Stage 3 — SageMaker, once there is a model to train

Worth the ceremony only when runs become repetitive. Two things to get right:

**Data access mode matters more than instance type.** `File` mode copies the
entire channel to the instance before training starts — with 30 GB that is dead
time on every run. Use `FastFile` mode, which streams from S3 on demand and lets
training start immediately.

```python
from sagemaker.inputs import TrainingInput

TrainingInput(
    "s3://stemify-data/slakh2100/",
    input_mode="FastFile",
)
```

**Keep data out of the image.** The container carries code and dependencies;
data arrives through a channel. Baking a dataset into an image makes it
enormous, slow to push, and impossible to version independently of the code.

Point the estimator at `src/stemify` as `source_dir` so the installed package
layout is identical to local, and pin the framework version rather than floating
it — a DLC upgrade that changes the torch version underneath a training run is a
genuinely confusing failure to debug.

## Cost notes

Rough and worth re-checking, but the shape is right:

| item | approximate |
|---|---|
| Slakh transfer + prune (c5n.2xlarge spot, ~3 h) | a few dollars, once |
| S3 storage, pruned Slakh (~30 GB) | ~$0.70/month |
| Demucs pass over the corpus (g5.xlarge) | hours of GPU, once, then cached |
| Evaluation runs (CPU spot) | cents per run |

The dominant recurring cost is GPU training time, not data. Ingest and storage
are close to noise, which is a good reason to keep the raw material in S3 rather
than re-downloading 104 GB whenever a laptop fills up.
