"""SemiCRFDecoder -- gold construction, a differentiable NLL that trains, and a
Viterbi decode that recovers the notes after a few fit steps. No audio, no model:
synthetic onset/voicing curves and a pyin-style contour drive the whole path.
"""

import numpy as np
import torch

from stemify.contour import Contour
from stemify.cues import Cues
from stemify.decoders.semicrf import (
    NOTE,
    REST,
    SemiCRFDecoder,
    _emissions,
    _gold_segments,
)
from stemify.notes import NoteEvent

DT = 0.0116  # feature hop, seconds
T = 240
TIMES = np.arange(T) * DT  # ~2.8 s grid


def _curve(spans: list[tuple[float, float]], hi: float, lo: float) -> np.ndarray:
    """A per-frame curve at ``hi`` inside any ``(start, end)`` span, else ``lo``."""
    y = np.full(T, lo, dtype=np.float32)
    for a, b in spans:
        y[(TIMES >= a) & (TIMES < b)] = hi
    return y


def _two_note_cues(
    note_spans: list[tuple[float, float]], pitches_hz: list[float]
) -> Cues:
    """Cues for held notes: an onset spike at each start, voicing high inside the
    notes, and a contour holding each note's pitch (unvoiced between)."""
    onset = np.full(T, 0.01, dtype=np.float32)
    for a, _ in note_spans:
        onset[round(a / DT)] = 0.99
    voicing = _curve(note_spans, 0.99, 0.01)
    freqs = np.zeros(T, dtype=np.float32)
    for (a, b), hz in zip(note_spans, pitches_hz):
        freqs[(TIMES >= a) & (TIMES < b)] = hz
    return Cues(
        contour=Contour(times=TIMES, freqs=freqs),
        onset=onset, frame_times=TIMES, voicing=voicing,
    )


def _notes(spans: list[tuple[float, float]], midis: list[float]) -> list[NoteEvent]:
    return [NoteEvent(onset=a, offset=b, midi_pitch=m) for (a, b), m in zip(spans, midis)]


def test_gold_rest_across_audible_gap():
    # 0.9 -> 1.2 is a 0.3 s gap (> GAP_REST_S): a real rest between the two notes.
    dec = SemiCRFDecoder()
    e = _emissions(_two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0]), dec.stride)
    dmax, _ = dec._grid(e.frame_times)
    gold = _gold_segments(e.frame_times, _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50]), dmax)
    labels = [lab for lab, _, _ in gold]
    assert labels.count(NOTE) == 2
    assert REST in labels  # the audible gap became a rest


def test_gold_closes_a_note_off_artifact_gap():
    # 0.5 -> 0.55 is a 0.05 s gap (< GAP_REST_S): a note-off artifact, so the notes
    # are made contiguous (note -> note) with no interior rest.
    dec = SemiCRFDecoder()
    e = _emissions(_two_note_cues([(0.2, 0.5), (0.55, 0.9)], [110.0, 110.0]), dec.stride)
    dmax, _ = dec._grid(e.frame_times)
    gold = _gold_segments(e.frame_times, _notes([(0.2, 0.5), (0.55, 0.9)], [45, 45]), dmax)
    # No rest between the notes: everything from the first note to the last is NOTE.
    note_idx = [k for k, (lab, _, _) in enumerate(gold) if lab == NOTE]
    between = gold[note_idx[0] : note_idx[-1] + 1]
    assert all(lab == NOTE for lab, _, _ in between)


def test_nll_is_differentiable_and_trains_down():
    dec = SemiCRFDecoder()
    cues = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    notes = _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50])
    opt = torch.optim.Adam(dec.parameters(), lr=0.2)

    first = dec.nll(cues, notes)
    assert first.requires_grad
    for _ in range(60):
        opt.zero_grad()
        loss = dec.nll(cues, notes)
        loss.backward()
        # a learned potential receives gradient -> the encoder would too, end to end
        assert dec.w_on.grad is not None
        opt.step()
    assert dec.nll(cues, notes).item() < first.item()


def test_decode_recovers_two_notes_after_fitting():
    dec = SemiCRFDecoder()
    cues = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    notes = _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50])
    opt = torch.optim.Adam(dec.parameters(), lr=0.2)
    for _ in range(120):
        opt.zero_grad()
        dec.nll(cues, notes).backward()
        opt.step()

    decoded = dec.decode(cues)
    assert len(decoded) == 2
    onsets = sorted(n.onset for n in decoded)
    assert abs(onsets[0] - 0.2) < 0.1 and abs(onsets[1] - 1.2) < 0.1
    # pitch read from the contour, not invented: ~A2 (110 Hz) then ~D3 (146 Hz)
    pitches = sorted(n.midi_pitch for n in decoded)
    assert abs(pitches[0] - 45) < 1.5 and abs(pitches[1] - 50) < 1.5


def test_decode_empty_contour_is_no_notes():
    empty = Cues(
        contour=Contour.empty(),
        onset=np.zeros(0, dtype=np.float32),
        frame_times=np.zeros(0),
        voicing=None,
    )
    assert SemiCRFDecoder().decode(empty) == []


def test_decode_without_voicing_head_falls_back_to_contour():
    # No voicing curve: voicing derives from the contour's own voiced mask.
    cues = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    cues = Cues(contour=cues.contour, onset=cues.onset, frame_times=cues.frame_times)
    dec = SemiCRFDecoder()
    notes = _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50])
    opt = torch.optim.Adam(dec.parameters(), lr=0.2)
    for _ in range(120):
        opt.zero_grad()
        dec.nll(cues, notes).backward()
        opt.step()
    assert len(dec.decode(cues)) == 2
