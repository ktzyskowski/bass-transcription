"""A run's results as data, not as printed output.

The previous harness printed and returned nothing, which meant a result existed
only in a terminal scrollback. That is fine while iterating by hand and useless
the moment you want to compare two baselines, track RPA across commits, or read
back what a SageMaker job produced. A ``Report`` is serialisable, so a run can be
written to S3 next to the model it evaluated.

Rendering stays here rather than in the metrics: a metric knows how to format
*itself*, a report knows how to lay out a whole run.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from stemify.evaluation.metrics.base import MetricResult


def _jsonable(value: Any) -> Any:
    """Convert numpy types to something ``json`` will accept."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


@dataclass(frozen=True)
class Report:
    """What one baseline scored on one dataset, overall and by group."""

    baseline: str
    dataset: str
    n_items: int
    n_frames: int
    overall: list[MetricResult]
    breakdowns: dict[str, dict[str, list[MetricResult]]] = field(default_factory=dict)
    per_item: list[tuple[str, dict[str, float]]] = field(default_factory=list)

    def scalar(self, metric: str, key: str) -> float | None:
        """One number by name, for comparing runs without parsing text."""
        for result in self.overall:
            if result.name == metric:
                return result.scalars.get(key)
        return None

    def to_dict(self) -> dict:
        """Serialisable form. Arrays are kept -- a confusion matrix is small and
        is the part you would want to plot later from a stored run."""
        return {
            "baseline": self.baseline,
            "dataset": self.dataset,
            "n_items": self.n_items,
            "n_frames": self.n_frames,
            "overall": {
                r.name: {
                    "scalars": {k: _jsonable(v) for k, v in r.scalars.items()},
                    "arrays": {k: _jsonable(v) for k, v in r.arrays.items()},
                }
                for r in self.overall
            },
            "breakdowns": {
                key: {
                    value: {
                        r.name: {k: _jsonable(v) for k, v in r.scalars.items()}
                        for r in results
                    }
                    for value, results in buckets.items()
                }
                for key, buckets in self.breakdowns.items()
            },
            "per_item": [
                {"name": name, **{k: _jsonable(v) for k, v in scalars.items()}}
                for name, scalars in self.per_item
            ],
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def render(self, worst: int = 10) -> str:
        lines = [
            f"\n=== {self.baseline} on {self.dataset} "
            f"({self.n_items:,} items, {self.n_frames:,} frames) ==="
        ]
        for result in self.overall:
            lines += [f"\n{result.name}", result.render()]

        for key, buckets in self.breakdowns.items():
            lines.append(f"\nby {key}")
            lines.append(f"  {key:<14} {'n':>9} {'RPA':>8} {'RCA':>8} {'gap':>8}")
            for value, results in buckets.items():
                scalars = next(
                    (r.scalars for r in results if r.name == "pitch-accuracy"), None
                )
                if scalars is None:
                    continue
                lines.append(
                    f"  {value[:14]:<14} {scalars['n_voiced']:>9,.0f} "
                    f"{scalars['rpa']:>8.1%} {scalars['rca']:>8.1%} "
                    f"{scalars['octave_gap']:>8.1%}"
                )

        # A good aggregate can hide a handful of items the baseline fails
        # outright, which is a different problem from being uniformly mediocre.
        if len(self.per_item) > 1 and worst:
            ranked = sorted(self.per_item, key=lambda pair: pair[1]["rpa"])[:worst]
            lines.append(f"\nworst {len(ranked)} items by RPA")
            lines.append(f"  {'RPA':>8} {'oct-gap':>8}  item")
            for name, scalars in ranked:
                lines.append(
                    f"  {scalars['rpa']:>8.1%} {scalars['octave_gap']:>8.1%}  {name[:50]}"
                )
        return "\n".join(lines)
