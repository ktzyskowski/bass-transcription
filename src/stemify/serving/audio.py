"""Any audio file -> ``Sources``, the thing a ``Transcriber`` consumes.

The training corpus hands out stems with known paths; a song does not. What is missing
between the two is one step -- run the mixture through separation and name the roles --
and this is that step.

**The domain gap this exists to measure.** ``DemucsSeparator`` notes that demucs was
trained on real music, so feeding it synthetic mixes yields a different artifact profile
than real recordings do. Every model here has only ever seen demucs-on-Slakh, which is
synthesised. Running a real song through this path is the first time the encoder sees
the artifacts it would actually be deployed against, and a large drop would be evidence
about the corpus rather than about the decoder.
"""

from pathlib import Path

from stemify.sources import Sources
from stemify.transforms import DEFAULT_CACHE, DemucsSeparator, Identity


def sources_from(
    audio: str | Path,
    *,
    separate: bool = True,
    cache_dir: str | Path = DEFAULT_CACHE,
    device: str | None = None,
) -> Sources:
    """``Sources`` for a file, separating it first unless it is already a bass stem.

    ``separate=False`` is for an isolated bass recording -- a DI, or a stem someone
    already extracted. The distinction matters and cannot be guessed from the file:
    separating an already-isolated bass runs demucs over a signal it will mostly pass
    through, costing minutes and adding artifacts the model was never trained on.

    ``mix`` is carried through whenever there was one, because a two-channel checkpoint
    reads it. A bass-only input leaves it ``None``, which ``Sources`` documents as the
    isolated-upload case and a one-channel encoder never looks at.
    """
    path = Path(audio)
    if not path.exists():
        raise FileNotFoundError(f"no audio at {path}")
    if not separate:
        return Sources(bass=Identity().apply(path))
    stem = DemucsSeparator(cache_dir=cache_dir, device=device).apply(path)
    return Sources(bass=stem, mix=path)
