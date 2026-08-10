"""Atomic file writes: never leave a truncated file that reads back as a cache hit.

Every disk cache here is a memoised expensive computation (separation, the CQT, a
pyin contour), and an interrupted write -- a Ctrl-C mid-track, a killed worker --
must not poison the cache with a half-written file the next run trusts. The guard is
write-aside-and-rename: write to a temp path, then rename onto the destination, which
is atomic within a filesystem, so a reader only ever sees a whole file or none.

A tiny top-level leaf (stdlib only) so both ``transforms`` (audio caches) and the
feature/contour caches share it without an import cycle.
"""

import os
from collections.abc import Callable
from pathlib import Path


def atomic_write(dest: Path, write_fn: Callable[[Path], None]) -> None:
    """Write ``dest`` via a temp file + atomic rename. ``write_fn`` is handed the temp
    path and must write the whole payload to it; on success it is renamed onto
    ``dest``, on failure it is cleaned up and ``dest`` is untouched.

    The temp name carries this process's pid, so two processes computing the same
    entry concurrently (a pre-warm pass alongside a run) write distinct temp files --
    last rename wins, both are valid, neither sees the other's partial.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(f"{dest.name}.{os.getpid()}.partial")
    try:
        write_fn(partial)
        partial.replace(dest)
    finally:
        partial.unlink(missing_ok=True)
