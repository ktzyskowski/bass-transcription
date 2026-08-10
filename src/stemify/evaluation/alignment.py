"""Put an estimate contour on the reference's time grid, voicing-aware.

Every pitch source emits its own grid: pyin hops 512 samples at 22.05 kHz (~23 ms),
MDB-bass-synth annotates every 128 samples at 44.1 kHz (~2.9 ms). Nothing can be
compared until they agree.

The reference grid wins. That is the mir_eval convention and the right one -- the
reference decides where the question "what pitch is sounding?" is asked, and the
estimate answers wherever it is asked. Resampling the reference down to a coarse
estimate grid would instead let a source improve its score by predicting less often.

Resampling is voicing-aware (``mir_eval.melody.resample_melody_series``):
interpolating Hz straight through an unvoiced gap would invent a glide between two
unrelated notes, so voicing is carried separately and reapplied after.

This is the pitch-channel aligner for ``CuesEvaluation`` -- the encoder's contour
(``cues.contour``) is the estimate, the item's reference contour the grid.
"""

import mir_eval
import numpy as np

from stemify.music.contour import Contour


def align(reference: Contour, estimate: Contour) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``estimate`` onto the reference grid.

    Returns ``(true, pred)``: two equal-length Hz arrays with 0.0 for unvoiced,
    which is exactly the contract ``FrameMetric.compute`` expects.
    """
    if len(estimate) == 0:
        return reference.freqs, np.zeros_like(reference.freqs)

    est_freqs, est_voicing = mir_eval.melody.resample_melody_series(
        times=estimate.times,
        frequencies=np.abs(estimate.freqs),
        voicing=estimate.voiced.astype(float),
        times_new=reference.times,
    )
    # resample_melody_series returns a soft voicing weight; anything that is not
    # confidently voiced after interpolation is treated as a gap.
    pred = np.where(est_voicing > 0.5, est_freqs, 0.0)
    return reference.freqs, pred
