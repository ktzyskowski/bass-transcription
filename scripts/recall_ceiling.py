"""Which reference notes never make it into the candidate set?

Written when candidate recall appeared pinned near 0.90 -- unmoved by 37 vs 1015
training tracks, by unweighted vs ``pos_weight=5``, and by decode threshold 0.05 vs
0.7. Three independent axes leaving it fixed says the missing notes are structural
rather than a matter of tuning, so the useful question is *what* the structure is.

The answer turned out to be mostly the pitch source rather than the encoder, and the
learned pitch head then moved the ceiling this script was built to explain (see
docs/experiments.md). The decomposition below is still how you would ask the question
again -- of whatever the current ceiling is -- so read the numbers it prints, not any
number quoted here.

The question is decision-relevant, not idle. A note verifier (rung 3) only ever
*deletes* candidates, so every reference note absent from ``group()``'s output is a
hard cap on what one could ever be worth. Whether it is worth building depends on
whether the missing notes are:

  - cheap    -- a decoder constant (MIN_NOTE_S, MIN_ONSET_GAP_S) throws them away, and
                a one-line change recovers them
  - upstream -- pyin tracked no pitch, or demucs deleted the evidence, in which case
                nothing downstream of the encoder can help either
  - real     -- the encoder genuinely never fired, which is the emissions problem the
                verifier was never going to fix

Matching is pitch-blind (``pitch_tolerance`` relaxed to 1e6 cents -> onset-only), so a
note that was found at the wrong pitch counts as found. This isolates *placement*
failure, which is what recall means here; pitch is rung 4's problem and is measured
separately by ``pitch_ceiling.py``.

Every reference note is written to a CSV -- matched ones too. The FN population alone
would only describe what bass notes look like in general; the comparison against
matched notes is what makes a bucket a finding.
"""

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from stemify.cache import Cache
from stemify.decoders.greedy import (
    MIN_NOTE_S,
    MIN_ONSET_GAP_S,
    ONSET_THRESHOLD,
    VOICING_THRESHOLD,
    GreedyDecoder,
    onset_times_from_probs,
)
from stemify.encoders.learned import LearnedEncoder
from stemify.training.checkpoint import load_cuenet
from stemify.features import BINS_PER_SEMITONE, midi_to_bin
from stemify.metrics.note.base import ONSET_TOLERANCE_S, matched_pairs
from stemify.notes import NoteEvent
from stemify.serde import features_npy
from stemify.training.conditions import CONDITIONS, canonical
from stemify.training.validation import CheckpointValidator, ValItem

ROOT = "data/raw/slakh2100"
CACHE_DIR = "data/cache"
IGNORE_PITCH = 1e6  # cents: any pitch matches, so matching is onset-only
# The CQT is dB relative to the track's own peak and hard-clipped at -80 (TOP_DB), so
# -80 is literal silence. -60 leaves margin above the clip: a note 60 dB under the
# loudest thing in its own track is not audible in it, whatever the MIDI says.
FLOOR_DB = -60.0


@dataclass
class RefNote:
    """One reference note, with everything that might explain whether it was found.

    Flat and CSV-shaped on purpose: the model pass is the expensive part (minutes),
    the slicing is not, so this is dumped once and re-read as often as a new
    hypothesis needs testing.
    """

    track: int
    index: int
    found: bool

    # -- what the note is ----------------------------------------------------------
    onset: float
    duration: float
    pitch: float
    gap_prev: float       # seconds of silence before it (negative = overlaps its predecessor)
    ioi_prev: float       # onset-to-onset from the previous note; inf for the first
    ioi_next: float       # onset-to-onset to the next note; inf for the last
    interval_prev: float  # semitones from the previous note's pitch (0.0 = a repeat)
    overlaps: int         # how many other reference notes sound during this one

    # -- what the pipeline had to work with -----------------------------------------
    onset_prob: float     # peak of the onset curve within the match tolerance
    peak_dt: float        # seconds to the nearest *picked* peak; inf if none fired
    voiced_frac: float    # fraction of the note's frames the voicing head calls voiced
    pyin_frac: float      # fraction of the note's frames pyin tracked any pitch in
    energy_db: float      # CQT level at this pitch during the note, dB below track peak
    clean_db: float       # the same, measured in the *clean stem* -- see below

    # -- what came out ---------------------------------------------------------------
    est_dt: float         # seconds to the nearest estimated note onset; inf if none
    absorbed: bool        # an estimate covers this onset but started well before it

    @property
    def separation_loss(self) -> float:
        """How much level this note lost to separation, dB. The column that decides
        whether a silent note is demucs's fault or the label's: energy in the clean
        stem but not in the separated audio means separation deleted it (recoverable
        in principle), no energy in either means the MIDI note never audibly sounded
        (recoverable by nobody, and it should not be)."""
        return self.clean_db - self.energy_db


def _span(times: np.ndarray, lo: float, hi: float) -> slice:
    """The frames of a grid falling in [lo, hi), never empty at a grid's resolution."""
    i, j = np.searchsorted(times, [lo, hi])
    return slice(int(i), max(int(j), int(i) + 1))


def _energy_db(features: np.ndarray, frame_times: np.ndarray, note: NoteEvent) -> float:
    """Peak CQT level at the note's fundamental while it sounds, in dB below the
    track's own peak (the CQT is ``ref=np.max``-normalised, so 0 is the loudest thing
    in the track and -80 is the floor).

    Channel 0 only -- that is the audio being transcribed. Under the ``separated``
    condition this answers the question no amount of downstream work can: did demucs
    leave anything there at all? A note sitting at the -80 floor was deleted before the
    model ever saw it.
    """
    if features.ndim == 3:
        features = features[0]
    bin_c = midi_to_bin(note.midi_pitch)
    lo = max(0, int(round(bin_c)) - BINS_PER_SEMITONE)
    hi = min(features.shape[0], int(round(bin_c)) + BINS_PER_SEMITONE + 1)
    if lo >= hi:
        return float("nan")
    window = features[lo:hi, _span(frame_times, note.onset, note.offset)]
    return float(window.max()) if window.size else float("nan")


def profile_track(
    ex: ValItem,
    cues,
    est: list[NoteEvent],
    track: int,
    threshold: float,
    clean: np.ndarray | None = None,
) -> list[RefNote]:
    """Every reference note in one track, matched or not, with its explanatory features."""
    ref = ex.notes
    matched = {r for r, _ in matched_pairs(ref, est, pitch_tolerance=IGNORE_PITCH)}

    peaks = onset_times_from_probs(cues.onset, cues.frame_times, threshold, MIN_ONSET_GAP_S)
    est_onsets = np.array([n.onset for n in est]) if est else np.zeros(0)
    features = ex.load_features()
    contour = ex.contour

    rows = []
    for i, n in enumerate(ref):
        prev = ref[i - 1] if i else None
        nxt = ref[i + 1] if i + 1 < len(ref) else None

        frames = _span(cues.frame_times, n.onset, n.offset)
        near = _span(cues.frame_times, n.onset - ONSET_TOLERANCE_S, n.onset + ONSET_TOLERANCE_S)
        c_frames = _span(contour.times, n.onset, n.offset)

        voicing = cues.voicing
        rows.append(
            RefNote(
                track=track,
                index=i,
                found=i in matched,
                onset=n.onset,
                duration=n.offset - n.onset,
                pitch=n.midi_pitch,
                gap_prev=n.onset - prev.offset if prev else float("inf"),
                ioi_prev=n.onset - prev.onset if prev else float("inf"),
                ioi_next=nxt.onset - n.onset if nxt else float("inf"),
                interval_prev=n.midi_pitch - prev.midi_pitch if prev else float("nan"),
                overlaps=sum(
                    1 for o in ref if o is not n and o.onset < n.offset and o.offset > n.onset
                ),
                onset_prob=float(cues.onset[near].max()),
                peak_dt=float(np.abs(peaks - n.onset).min()) if peaks.size else float("inf"),
                voiced_frac=(
                    float((voicing[frames] > VOICING_THRESHOLD).mean())
                    if voicing is not None else float("nan")
                ),
                pyin_frac=float((contour.freqs[c_frames] > 0).mean()),
                energy_db=_energy_db(features.values, cues.frame_times, n),
                clean_db=(
                    _energy_db(clean, cues.frame_times, n) if clean is not None
                    else float("nan")
                ),
                est_dt=(
                    float(np.abs(est_onsets - n.onset).min()) if est_onsets.size
                    else float("inf")
                ),
                # An estimate that spans this onset but began more than a tolerance
                # earlier is the merge failure: the note is inside a candidate, not
                # missing from the audio, and only a *split* recovers it.
                absorbed=any(
                    e.onset < n.onset - ONSET_TOLERANCE_S and e.offset > n.onset for e in est
                ),
            )
        )
    return rows


def clean_features(limit: int, cache: Cache) -> dict[str, Path]:
    """Each val track's *clean stem* CQT, by track name.

    Joined by name rather than position: ``clean`` and the separated conditions filter
    to the same single-bass tracks and so should enumerate identically, but "should"
    is how track *i*'s notes end up scored against track *j*'s audio, and that failure
    looks like a real finding rather than a bug.
    """
    condition = CONDITIONS["clean"]
    dataset = condition.dataset(ROOT, "validation", limit)
    paths = condition.channel_paths(dataset, cache)
    return {item.name: features_npy(p[0]) for item, p in zip(dataset, paths, strict=True)}


def collect(tag: str, ckpt: Path, limit: int, threshold: float) -> list[RefNote]:
    cache = Cache(CACHE_DIR)
    condition = CONDITIONS[canonical(tag)]
    dataset = condition.dataset(ROOT, "validation", limit)
    names = [item.name for item in dataset]

    validator = CheckpointValidator(dataset, condition, cache=cache)
    clean = clean_features(limit, cache)
    missing = [n for n in names if n not in clean]
    if missing:
        raise ValueError(f"no clean stem for {len(missing)} track(s): {missing[:5]}")

    encoder = LearnedEncoder(load_cuenet(ckpt))
    decoder = GreedyDecoder(onset_threshold=threshold)

    rows: list[RefNote] = []
    for t, (name, ex) in enumerate(zip(names, validator.examples, strict=True)):
        cues = encoder.encode_features(ex.contour, ex.load_features())
        rows += profile_track(
            ex, cues, decoder.decode(cues), t, threshold, np.load(clean[name])
        )
    return rows


# ---------------------------------------------------------------------------------
# Reporting. Every table is a *miss rate per bucket* against the share of notes in
# that bucket -- a bucket with a high miss rate but 12 notes in it is noise, and the
# count column is what stops it being read as a finding.
# ---------------------------------------------------------------------------------

def audibility_table(rows: list[RefNote]) -> str:
    """Split the quiet notes by *why* they are quiet -- the fork that decides whether
    the recall ceiling is a bug or the labels.

    A reference note with no energy in the clean stem either was never played audibly
    or is a MIDI artifact; it is not a miss any model should be asked to fix, and it
    caps recall for everyone. A note loud in the stem but gone from the separated
    audio is separation damage, which *is* addressable -- by a better separator, or by
    the mix channel the model already has.
    """
    audible = np.array([r.clean_db for r in rows]) > FLOOR_DB
    survived = np.array([r.energy_db for r in rows]) > FLOOR_DB
    found = np.array([r.found for r in rows])

    out = ["  audibility".ljust(42) + f"{'notes':>8} {'share':>7} {'miss':>7}"]
    for label, sel in [
        ("audible in stem, survived separation", audible & survived),
        ("audible in stem, LOST to separation", audible & ~survived),
        ("inaudible in the clean stem too", ~audible),
    ]:
        n = int(sel.sum())
        if n:
            out.append(
                f"    {label:<38}{n:>8} {n / len(rows):>6.1%} {1 - found[sel].mean():>6.1%}"
            )
    # The ceiling if the un-played notes were simply not in the reference.
    real = audible
    out += [
        "",
        f"    recall over all reference notes    {found.mean():>8.4f}",
        f"    recall over audible notes only     {found[real].mean():>8.4f}"
        f"   <- the ceiling that is actually ours to raise",
    ]
    return "\n".join(out)


def bucket_table(rows: list[RefNote], field: str, edges: list[float], unit: str = "") -> str:
    vals = np.array([getattr(r, field) for r in rows], dtype=float)
    found = np.array([r.found for r in rows])
    ok = ~np.isnan(vals)

    out = [f"  {field}{unit}".ljust(24) + f"{'notes':>8} {'share':>7} {'miss':>7} {'lift':>6}"]
    base = 1.0 - found[ok].mean()
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = ok & (vals >= lo) & (vals < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        miss = 1.0 - found[sel].mean()
        label = f"{lo:g} .. {hi:g}" if np.isfinite(hi) else f">= {lo:g}"
        out.append(
            f"    {label:<20}{n:>8} {n / ok.sum():>6.1%} {miss:>6.1%} {miss / base:>5.1f}x"
        )
    return "\n".join(out)


def mechanism_table(rows: list[RefNote], threshold: float) -> str:
    """Why each missed note was missed, assigned to the earliest stage that explains it.

    Ordered by pipeline position, and *exclusive*: a note whose pitch pyin never
    tracked is counted there even if the onset curve was also flat, because fixing the
    later stage would not have recovered it. The marginal (non-exclusive) rates follow,
    since the ordering is a choice and the overlaps are worth seeing.
    """
    missed = [r for r in rows if not r.found]
    tests = [
        ("pyin tracked no pitch", lambda r: r.pyin_frac < 0.5),
        ("voicing head says rest", lambda r: r.voiced_frac < 0.5),
        ("no onset peak fired", lambda r: r.peak_dt > ONSET_TOLERANCE_S),
        ("absorbed into a neighbour", lambda r: r.absorbed),
        ("estimate there, lost to matching", lambda r: r.est_dt <= ONSET_TOLERANCE_S),
        ("shorter than MIN_NOTE_S", lambda r: r.duration < MIN_NOTE_S),
    ]

    out = [f"  {len(missed)} missed of {len(rows)} reference notes",
           "",
           "  exclusive (earliest stage that explains it)".ljust(42)
           + f"{'notes':>7} {'of miss':>9}"]
    claimed = set()
    for label, test in tests:
        hit = [i for i, r in enumerate(missed) if i not in claimed and test(r)]
        claimed |= set(hit)
        out.append(f"    {label:<38}{len(hit):>7} {len(hit) / len(missed):>8.1%}")
    left = len(missed) - len(claimed)
    out.append(f"    {'unexplained':<38}{left:>7} {left / len(missed):>8.1%}")

    out += ["", "  marginal (each test over all missed notes)".ljust(42) + f"{'notes':>7} {'of miss':>9}"]
    for label, test in tests:
        n = sum(1 for r in missed if test(r))
        out.append(f"    {label:<38}{n:>7} {n / len(missed):>8.1%}")
    out.append(f"    {'sub-threshold peak (prob < %.2f)' % threshold:<38}"
               f"{sum(1 for r in missed if r.onset_prob < threshold):>7}"
               f" {sum(1 for r in missed if r.onset_prob < threshold) / len(missed):>8.1%}")
    return "\n".join(out)


def report(rows: list[RefNote], threshold: float) -> None:
    found = np.array([r.found for r in rows])
    print(f"\ncandidate recall (pitch-blind)  {found.mean():.4f}"
          f"   {int(found.sum())}/{len(rows)} notes, {rows[-1].track + 1} tracks\n")
    print(mechanism_table(rows, threshold))
    print()
    print(audibility_table(rows))

    print("\n  -- distributions: miss rate by bucket "
          "(lift = miss rate / overall miss rate) --\n")
    inf = float("inf")
    for field, edges, unit in [
        ("duration", [0, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0, inf], " (s)"),
        ("gap_prev", [-inf, 0, 0.01, 0.05, 0.1, 0.3, inf], " (s of rest before)"),
        ("ioi_prev", [0, 0.1, 0.15, 0.2, 0.3, 0.5, inf], " (s since prev onset)"),
        ("pitch", [0, 33, 38, 43, 48, 55, 128], " (midi)"),
        ("interval_prev", [-inf, -12, -5, -0.5, 0.5, 5, 12, inf], " (semitones)"),
        ("energy_db", [-80, -60, -50, -40, -30, -20, 0], " (dB below track peak)"),
        ("clean_db", [-80, -60, -50, -40, -30, -20, 0], " (same, in the clean stem)"),
        ("separation_loss", [-inf, 0, 5, 10, 20, 40, inf], " (dB lost to separation)"),
        ("overlaps", [0, 1, 2, 3, inf], " (simultaneous notes)"),
        ("onset_prob", [0, 0.05, 0.2, 0.5, 0.8, 1.01], ""),
        ("pyin_frac", [0, 0.01, 0.25, 0.5, 0.75, 1.01], " (frames pyin tracked)"),
        ("voiced_frac", [0, 0.01, 0.25, 0.5, 0.75, 1.01], " (frames called voiced)"),
    ]:
        print(bucket_table(rows, field, edges, unit), "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--condition", default="separated")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--threshold", type=float, default=ONSET_THRESHOLD)
    p.add_argument("--out", type=Path, help="write the per-note CSV here")
    args = p.parse_args()

    rows = collect(args.condition, args.checkpoint, args.limit, args.threshold)
    if args.out:
        with args.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[f.name for f in fields(RefNote)])
            w.writeheader()
            w.writerows(asdict(r) for r in rows)
        print(f"wrote {len(rows)} rows to {args.out}")
    report(rows, args.threshold)


if __name__ == "__main__":
    main()
