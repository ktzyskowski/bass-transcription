"""Evaluate a pitch baseline on the IDMT-SMT-Bass isolated notes.

    uv run python -m stemify.evaluation.pitch --baseline pyin
    uv run python -m stemify.evaluation.pitch --baseline pyin --limit 300 --csv out.csv

Only the notes whose pitch the labels actually pin down are scored (NO/VI/VIF/VIS
-- see Note.midi_pitch); harmonics, dead notes, bends and slides are excluded
because there is no single ground-truth pitch to compare against.

Because weak bass fundamentals make octave errors the dominant failure mode,
accuracy is reported both exactly and octave-collapsed (chroma), with the
octave-error rate broken out on its own.
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from stemify.baselines.base import PitchBaseline
from stemify.baselines.basic_pitch import BasicPitchBaseline
from stemify.baselines.pyin import PyinBaseline
from stemify.data.idmt_smt_bass import Note, index

BASELINES: dict[str, type[PitchBaseline]] = {
    "pyin": PyinBaseline,
    "basic-pitch": BasicPitchBaseline,
}


@dataclass
class Prediction:
    note: Note
    pred: float | None

    @property
    def label(self) -> int:
        # Guaranteed non-None: the runner only scores the pitched subset.
        assert self.note.midi_pitch is not None
        return self.note.midi_pitch

    @property
    def predicted(self) -> bool:
        return self.pred is not None

    @property
    def exact(self) -> bool:
        return self.pred is not None and round(self.pred) == self.label

    @property
    def chroma_match(self) -> bool:
        """Right pitch class, any octave -- i.e. exact or a pure octave error."""
        return self.pred is not None and (round(self.pred) - self.label) % 12 == 0

    @property
    def cents_error(self) -> float | None:
        """Tuning error once octaves are folded out, for chroma-correct notes."""
        if not self.chroma_match:
            return None
        err = self.pred - self.label
        err -= 12 * round(err / 12)  # fold octaves to the nearest
        return abs(err) * 100.0


def run(baseline: PitchBaseline, notes: list[Note]) -> list[Prediction]:
    preds: list[Prediction] = []
    for i, note in enumerate(notes, 1):
        preds.append(Prediction(note, baseline.predict_pitch(note.path)))
        if i % 100 == 0 or i == len(notes):
            print(f"  {i}/{len(notes)}", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    return preds


def _rates(preds: list[Prediction]) -> dict[str, float]:
    scored = [p for p in preds if p.predicted]
    if not scored:
        return {"n": len(preds), "coverage": 0.0}
    cents = [p.cents_error for p in scored if p.cents_error is not None]
    cents.sort()
    return {
        "n": len(preds),
        "coverage": len(scored) / len(preds),
        "exact": sum(p.exact for p in scored) / len(scored),
        "chroma": sum(p.chroma_match for p in scored) / len(scored),
        "octave_err": sum(p.chroma_match and not p.exact for p in scored)
        / len(scored),
        "median_cents": cents[len(cents) // 2] if cents else float("nan"),
    }


def _print_group(title: str, key, preds: list[Prediction]) -> None:
    print(f"\n{title}")
    print(f"  {'':<10} {'n':>5} {'exact':>7} {'chroma':>7} {'oct-err':>7}")
    for value in sorted({key(p) for p in preds}, key=str):
        rows = [p for p in preds if key(p) == value]
        r = _rates(rows)
        print(
            f"  {str(value):<10} {r['n']:>5} "
            f"{r.get('exact', 0):>7.1%} {r.get('chroma', 0):>7.1%} "
            f"{r.get('octave_err', 0):>7.1%}"
        )


def summarize(baseline_name: str, preds: list[Prediction]) -> None:
    r = _rates(preds)
    print(f"\n=== {baseline_name} on {r['n']} pitched notes ===")
    print(f"  coverage (returned a pitch)  {r['coverage']:>7.1%}")
    print(f"  exact accuracy               {r.get('exact', 0):>7.1%}")
    print(f"  chroma accuracy (any octave) {r.get('chroma', 0):>7.1%}")
    print(f"  of which octave errors       {r.get('octave_err', 0):>7.1%}")
    print(f"  median cents error (in-tune) {r.get('median_cents', float('nan')):>7.1f}")
    # String 1 is the low E, where the fundamental is weakest -- expect the
    # worst octave-error rate there.
    _print_group("by string (1=E .. 4=G)", lambda p: p.note.string, preds)
    _print_group("by plucking", lambda p: p.note.plucking, preds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", choices=BASELINES, required=True)
    ap.add_argument("--root", default="data/raw/idmt_smt_bass")
    ap.add_argument("--limit", type=int, help="score only the first N notes")
    ap.add_argument("--csv", type=Path, help="also write per-note predictions here")
    args = ap.parse_args()

    notes = [n for n in index(args.root) if n.midi_pitch is not None]
    if args.limit:
        notes = notes[: args.limit]

    baseline = BASELINES[args.baseline]()
    preds = run(baseline, notes)
    summarize(baseline.name, preds)

    if args.csv:
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "string", "fret", "plucking", "label", "pred"])
            for p in preds:
                w.writerow(
                    [p.note.path, p.note.string, p.note.fret, p.note.plucking,
                     p.label, "" if p.pred is None else f"{p.pred:.3f}"]
                )
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
