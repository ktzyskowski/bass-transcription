"""Spotify Basic Pitch baseline.

Basic Pitch is the baseline worth carrying across *every* stage: unlike a bare
f0 tracker it emits note events (onset, offset, pitch), which is the
representation the continuous-bassline stages actually need. On isolated notes
we collapse its events down to the single dominant pitch.

Two caveats that matter for bass specifically:

  - It is a *polyphonic* model. On clean monophonic input a dedicated f0 tracker
    (pyin, CREPE) can beat it, so read this as "the off-the-shelf transcriber,"
    not "the strongest possible pitch estimator."
  - It octave-errors in the low register, where the fundamental is weak relative
    to its harmonics -- the same effect seen in the raw CQT peaks. The
    evaluation reports octave-collapsed accuracy precisely so this is legible.

Dependency note: basic-pitch pulls in a heavy ML runtime (TensorFlow / ONNX /
CoreML, depending on the extra) and is NOT yet in pyproject. Verify a backend
wheel exists for this project's Python before adding it. The import is deferred
so this module still loads without the dependency present.
"""

from pathlib import Path

from stemify.baselines.base import PitchBaseline


class BasicPitchBaseline(PitchBaseline):
    name = "basic-pitch"

    def __init__(
        self,
        minimum_frequency: float = 30.0,
        maximum_frequency: float = 900.0,
    ) -> None:
        # Constrain the model to the plausible bass band. This does not stop
        # octave-up errors within the band -- it just rejects sub/ultrasonic
        # spurious detections.
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency

    def predict_pitch(self, path: str | Path) -> float | None:
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import predict
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "basic-pitch is not installed. Add it with `uv add basic-pitch` "
                "(confirm a backend wheel exists for this Python first), then "
                "retry."
            ) from exc

        # NOTE: predict() reloads the model per call. Fine for a one-off run;
        # for the full ~3100-note sweep, switch to the reusable Model API so the
        # graph loads once. Left simple until the dependency is confirmed.
        _model_output, _midi_data, note_events = predict(
            str(path),
            ICASSP_2022_MODEL_PATH,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
        )
        if not note_events:
            return None
        # Each event is (start_s, end_s, pitch_midi, amplitude, pitch_bends).
        # Pick the one carrying the most energy over time -- the actual note,
        # not a transient blip.
        start, end, pitch, amplitude, *_ = max(
            note_events, key=lambda e: (e[1] - e[0]) * e[3]
        )
        return float(pitch)
