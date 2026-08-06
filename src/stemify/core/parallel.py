"""A single place for the process-pool boilerplate, so callers express *what* to
parallelise, not *how*.

Deliberately a function, not a class: the work is a stateless input->output map, so
there is no pool lifecycle or configuration to hold between calls. A class would
earn its place only if we wanted a long-lived pool reused across many maps (to
amortise spawn cost) -- we do not; each parallel pass here runs once.

The one constraint the spawn start method imposes: ``fn`` and every item must be
picklable. Spawn re-imports the module in each worker rather than forking, so
``fn`` has to be a module-level function -- a lambda or a local closure will not
pickle, and neither will an object holding a torch/MPS handle. Pass plain data
(paths, ints, strings) and a top-level worker.
"""

import multiprocessing
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    workers: int,
    chunksize: int = 1,
    on_result: Callable[[int, R], None] | None = None,
) -> list[R]:
    """Apply ``fn`` to each item across ``workers`` subprocesses; serial when
    ``workers <= 1`` (a pool has real per-process overhead not worth paying for a
    handful of items). Uses the spawn context so a worker starts from a clean
    interpreter rather than inheriting the parent's torch/MPS state.

    ``fn`` must be a module-level callable and every item picklable -- see the
    module docstring. Results come back in input order.

    ``on_result(index, result)`` is called in the *caller's* process as each result
    arrives (1-based, input order), for streaming progress -- so it may be a lambda
    or closure, unlike ``fn``. Results stream via ``imap`` so a long run reports
    incrementally rather than only at the end.
    """
    items = list(items)

    def drain(results: Iterable[R]) -> list[R]:
        collected: list[R] = []
        for index, result in enumerate(results, start=1):
            if on_result is not None:
                on_result(index, result)
            collected.append(result)
        return collected

    if workers <= 1 or len(items) <= 1:
        return drain(fn(item) for item in items)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        return drain(pool.imap(fn, items, chunksize=chunksize))
