"""The memoised pitch contour.

The peer of ``audio.cache``, and here for the same reason: knowing that pyin is
expensive and that ``estimator.name`` identifies the producer is this layer's business,
not the store's.

Worth stating why memoising pays *more* here than for the CQT. A pyin contour is a
function of the audio and the estimator's fixed parameters and of nothing else -- no
checkpoint, no condition -- so it is computed once per file **ever** and reused across
every checkpoint and every future run. That is what makes a large validation set
affordable. A learned pitch head is checkpoint-*dependent* and so is recomputed per
checkpoint; it does not come through here, which is the point of the split rather than
a gap in it.
"""

from pathlib import Path

from stemify.transcription.pitch.base import PitchEstimator
from stemify.music.contour import Contour
from stemify.persistence.serde import load_contour, save_contour
from stemify.persistence.store import Store


def ensure_contour(
    store: Store, estimator: PitchEstimator, audio: str | Path
) -> Contour:
    """An estimator's contour for one resolved audio file, memoised to disk.

    Each condition's audio caches independently and never collides, since the resolved
    path is in the key -- a stem and the separated version of the same track are two
    different files and therefore two different entries.
    """
    path = store.location("contours", estimator.name, audio)
    if path.exists():
        return load_contour(path)
    contour = estimator.estimate(Path(audio))
    save_contour(contour, path)  # atomic_write creates the parent
    return contour
