"""Scoring a model against a dataset, at two altitudes of the note pipeline.

    from stemify.evaluation import CuesEvaluation, NoteEvaluation
    cues  = CuesEvaluation(SlakhDataset("data/raw/slakh2100"), ClassicalEncoder()).run()
    notes = NoteEvaluation(SlakhDataset("data/raw/slakh2100"),
                           Transcriber(ClassicalEncoder(), GreedyDecoder())).run()

Two harnesses, same shape (dataset + object -> a ``Report``), different altitude:

    cues.py   CuesEvaluation -> CuesReport   scores an Encoder's per-frame cues
    note.py   NoteEvaluation -> NoteReport   scores a whole Transcriber's notes

They are the two things worth scoring in ``Sources -> Encoder -> Frames -> Decoder ->
notes``: the encoder's evidence and the end-to-end notes. Run together they localise
a regression to the encoder or the decoder. Both share the ``Report`` surface
(render, to_dict, write_json), so a caller can hold either behind one type.

Supporting pieces stay in their modules:

    report.py      Report -- the shared render/to_dict/write_json surface
    alignment.py   voicing-aware resampling of a contour onto the reference grid
    ../metrics/    Metric.compute(true, pred) -> MetricResult
    ../segmenter.py  Transcriber -- Sources -> notes, the object the note harness scores
    cli.py         argument parsing

Nothing here imports from the package root, so these re-exports cannot cycle.
"""

from stemify.evaluation.cues import CuesEvaluation, CuesReport
from stemify.evaluation.note import NoteEvaluation, NoteReport
from stemify.evaluation.report import Report

__all__ = [
    "CuesEvaluation",
    "CuesReport",
    "NoteEvaluation",
    "NoteReport",
    "Report",
]
