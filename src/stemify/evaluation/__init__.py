"""Scoring a pitch baseline against a dataset, frame-level and note-level.

    from stemify.evaluation import FrameEvaluation, NoteEvaluation
    frame = FrameEvaluation(SlakhDataset("data/raw/slakh2100"), PyinBaseline()).run()
    notes = NoteEvaluation(SlakhDataset("data/raw/slakh2100"), PyinBaseline()).run()

Two parallel harnesses, same shape (dataset + baseline -> a ``Report``), different
question:

    frame.py  FrameEvaluation -> FrameReport   "what pitch at each instant"
    note.py   NoteEvaluation  -> NoteReport     "did you find the right notes"

They are separate rather than unified because the data flow genuinely differs --
frame pooling and per-frame metrics on one side, note-interval matching and
count pooling on the other. What they share is the ``Report`` surface (render,
to_dict, write_json), so a caller can hold either behind one type.

Both are re-exported here so callers never reach into a submodule. Supporting
pieces stay in their modules:

    report.py      Report -- the shared render/to_dict/write_json surface
    alignment.py   resampling onto the reference grid, and FrameSet pooling
    metrics/       Metric.calculate(true, pred) -> MetricResult   (frame-level)
    ../segmentation.py  Segmenter -- contour -> notes, for the note harness
    cli.py         argument parsing

Nothing here imports from the package root, so these re-exports cannot cycle.
"""

from stemify.evaluation.frame import FrameEvaluation, FrameReport
from stemify.evaluation.note import NoteEvaluation, NoteReport
from stemify.evaluation.report import Report

__all__ = [
    "FrameEvaluation",
    "FrameReport",
    "NoteEvaluation",
    "NoteReport",
    "Report",
]