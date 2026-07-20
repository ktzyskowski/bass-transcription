"""Audio in, audio out -- preprocessing that sits between a dataset and a baseline.

The motivating case is source separation. In production a bass pitch model will
not receive a clean stem; it will receive whatever a separator extracted from a
full mix. Training and evaluating only on clean stems bakes in a train/deploy
mismatch, so the harness needs a place to put "run this through Demucs first".

A transform produces a **file**, not an array, for two reasons: baselines take
paths (``librosa.load`` wants a filename), and separation is expensive enough
that recomputing it per run would dominate everything else. ``apply`` therefore
caches, and because a separator can take minutes per track, the write is atomic
-- an interrupted run must never leave a truncated file that later looks cached.

**What gets cached is exactly what the transform produced.** No mono mixdown, no
resampling, no bit-depth reduction, even though all three would shrink the cache
considerably. Those are the consumer's decisions -- baselines already load with
``mono=True`` and their own sample rate -- and baking them in here would make the
cache lossy in ways a later consumer cannot undo. FLAC keeps the size honest
without making that choice.

The cache is keyed on the transform's identity plus the *source path*, not the
source's contents. Hashing 100 MB of audio per lookup would cost more than it
saves, and the datasets here are immutable. The tradeoff: moving a dataset
directory invalidates its cache.
"""

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

DEFAULT_CACHE = Path("data/cache/transforms")


def _best_device() -> str:
    """The fastest accelerator torch can reach here.

    Deliberately not delegated to the library: Demucs defaults to ``"cpu"`` and
    never probes for an accelerator, so anything relying on its default runs on
    CPU without saying so.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AudioTransform(ABC):
    """A cached audio-to-audio operation."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE) -> None:
        self.cache_dir = Path(cache_dir)

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identity **including configuration**.

        This is the cache key and the label that appears in reports, so it must
        change whenever the output would. A separator that folds its model name
        in here gets correct invalidation for free when the model changes; one
        that returns a bare ``"demucs"`` will silently serve stale audio from a
        different model.
        """

    @abstractmethod
    def _run(self, source: Path, dest: Path) -> None:
        """Do the work, writing to ``dest``. Its parent already exists."""

    def cache_path(self, source: Path) -> Path:
        digest = hashlib.sha256(
            f"{self.name}\0{Path(source).resolve()}".encode()
        ).hexdigest()[:16]
        # Source stem in the filename keeps the cache browsable by eye; the
        # digest is what actually disambiguates.
        return self.cache_dir / self.name / f"{Path(source).stem}-{digest}.flac"

    def apply(self, source: str | Path) -> Path:
        """The transformed audio, computing it only if not already cached."""
        source = Path(source)
        dest = self.cache_path(source)
        if dest.exists():
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        # Write aside and rename: rename is atomic within a filesystem, so an
        # interrupted separation leaves no half-written file masquerading as a
        # cache hit. Without this a Ctrl-C mid-track silently poisons the cache.
        # The temp name carries this process's pid so two processes separating
        # the same source concurrently (a pre-warm pass alongside an eval run)
        # write to distinct temp files -- last rename wins, both are valid, and
        # neither sees a half-written partial from the other.
        partial = dest.with_name(f"{dest.name}.{os.getpid()}.partial")
        try:
            self._run(source, partial)
            partial.replace(dest)
        finally:
            partial.unlink(missing_ok=True)
        return dest


class Identity(AudioTransform):
    """Passes audio through untouched.

    Exists so "no separation" is a transform like any other rather than a
    special case: the three-condition comparison (clean stem / separated / raw
    mix) then differs only in which transform is configured, and every result
    carries a transform label instead of some carrying ``None``.
    """

    @property
    def name(self) -> str:
        return "identity"

    def _run(self, source: Path, dest: Path) -> None:  # pragma: no cover
        raise AssertionError("Identity.apply short-circuits; _run is unreachable")

    def apply(self, source: str | Path) -> Path:
        return Path(source)


class DemucsSeparator(AudioTransform):
    """Extracts one stem from a mixture using Demucs.

    Two things about Demucs that shape how this is used:

    **It is deterministic.** The same input always yields the same output, so
    this is not augmentation -- it contributes one sample per track, not
    variety. Variety has to come from varying the music.

    **It was trained on real music.** Feeding it synthetic or procedurally
    generated mixes produces a different artifact profile than it produces on
    real recordings, so a model trained against those artifacts is one domain
    gap further from production than it appears.

    The model is loaded once and reused; instantiating it per file would
    dominate the runtime on short clips.
    """

    def __init__(
        self,
        model: str = "htdemucs",
        stem: str = "bass",
        device: str | None = None,
        cache_dir: str | Path = DEFAULT_CACHE,
    ) -> None:
        super().__init__(cache_dir)
        self.model = model
        self.stem = stem
        # Demucs itself defaults to "cpu" -- it does *not* auto-select an
        # accelerator -- so leaving this unset silently gives up a large
        # speedup. On an M-series Mac, MPS measured 14x realtime against CPU's
        # 4.4x on a 171 s track.
        self.device = device or _best_device()
        self._separator = None

    @property
    def name(self) -> str:
        # Model and stem change the output, so both belong in the key. The
        # *device* deliberately does not, even though CPU and MPS produce
        # measurably different waveforms (~20 dB SNR between them on a real
        # track, spread throughout rather than an offset). Metrics computed
        # from either agree to within 0.1 percentage points -- the divergence
        # lives in low-level detail of the separation residual, which does not
        # move where pitch is or whether it is detectable. Keying on device
        # would mean re-separating an entire corpus after switching hardware,
        # for a difference the metrics cannot see.
        #
        # Caveat: that equivalence is established for *pitch evaluation*. If
        # separated audio is ever used as neural training data, re-check it
        # rather than assuming it carries over.
        return f"demucs-{self.model}-{self.stem}"

    def _load(self):
        if self._separator is None:
            try:
                from demucs.api import Separator
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "demucs is not installed. Add it with `uv add demucs`."
                ) from exc
            self._separator = Separator(model=self.model, device=self.device)
        return self._separator

    def _run(self, source: Path, dest: Path) -> None:
        import soundfile as sf

        separator = self._load()
        _origin, stems = separator.separate_audio_file(str(source))
        if self.stem not in stems:
            raise KeyError(
                f"{self.model} produced stems {sorted(stems)}, not {self.stem!r}"
            )
        audio = stems[self.stem].detach().cpu().numpy()
        # Demucs returns (channels, samples); soundfile wants (samples, channels).
        sf.write(dest, np.asarray(audio).T, separator.samplerate, format="FLAC")