"""Transcribing a real recording, and making the result checkable by a person.

Everything else in this project scores against Slakh, which is **synthesised**, with
reference notes to compare to. A song has neither. So this package's output is not a
metric -- it is a listen and a worksheet, and its job is to make a bass player's
disagreement fast to express.

  - ``audio`` -- any file to ``Sources``, separating a mixture on the way.
  - ``export`` -- notes to a preview mix and a CSV.
  - ``cli`` -- ``stemify-transcribe``.
"""

from stemify.serving.audio import sources_from
from stemify.serving.export import notes_csv, preview_wav

__all__ = ["notes_csv", "preview_wav", "sources_from"]
