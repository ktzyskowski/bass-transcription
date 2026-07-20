"""Scoring a pitch baseline against a dataset.

    from stemify.evaluation import Evaluation
    report = Evaluation(SlakhDataset("data/raw/slakh2100"), PyinBaseline()).run()

``Evaluation`` and ``Report`` are re-exported here so callers never reach into a
submodule: a notebook, a test, or a SageMaker entry point imports them straight
from the package.

The module is named after the class it holds. ``stemify.evaluation.evaluation``
does repeat itself, but that path is written exactly once -- in the re-export
below -- because Python code imports the class, not the module. The alternative
was naming the module after a verb (``run.py``), which reads fine in an import
line and tells you nothing about what is inside. The stdlib makes the same
trade in ``datetime.datetime``.

Submodules, where the work lives:

    evaluation.py  Evaluation -- binds dataset + baseline + metrics
    alignment.py   resampling onto the reference grid, and FrameSet pooling
    report.py      results as data
    metrics/       Metric.calculate(true, pred) -> MetricResult
    cli.py         argument parsing

Nothing here imports from the package root, so these re-exports cannot create a
cycle. Keep it that way: a submodule should import ``stemify.evaluation.report``,
never ``stemify.evaluation``.
"""

from stemify.evaluation.evaluation import Evaluation
from stemify.evaluation.report import Report

__all__ = ["Evaluation", "Report"]