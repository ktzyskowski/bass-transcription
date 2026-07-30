"""SemiCRFDecoder -- gold construction, a differentiable NLL that trains, and a
Viterbi decode that recovers the notes after a few fit steps. No audio, no model:
synthetic onset/voicing curves and a pyin-style contour drive the whole path.

The last section sweeps the same paths over ``cpu`` and ``cuda`` (see the ``device``
fixture) and asserts the two agree numerically. Running is not enough: a device fix
that quietly changes the arithmetic would still "work" everywhere, so parity against
CPU is what actually pins the behaviour.
"""

import numpy as np
import pytest
import torch

from stemify.contour import Contour
from stemify.cues import Cues
from stemify.decoders.semicrf import SemiCRFDecoder, TensorCues
from stemify.decoders.semicrf.constants import NOTE, REST, START
from stemify.decoders.semicrf.forward import batched_log_partition, chunk_log_partition
from stemify.decoders.semicrf.score import segmentation_score, segmentation_scores
from stemify.decoders.semicrf.emissions import Emissions, build_emissions, pad_emissions
from stemify.decoders.semicrf.gold import gold_segments
from stemify.notes import NoteEvent

SPANS = [(0.2, 0.9), (1.2, 1.9)]
HZ = [110.0, 146.0]
MIDI = [45, 50]

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
    e = build_emissions(_two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0]), dec.stride)
    dmax, _ = dec.pot.grid(e.frame_times)
    gold = gold_segments(e.frame_times, _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50]), dmax)
    labels = [lab for lab, _, _ in gold]
    assert labels.count(NOTE) == 2
    assert REST in labels  # the audible gap became a rest


def test_gold_closes_a_note_off_artifact_gap():
    # 0.5 -> 0.55 is a 0.05 s gap (< GAP_REST_S): a note-off artifact, so the notes
    # are made contiguous (note -> note) with no interior rest.
    dec = SemiCRFDecoder()
    e = build_emissions(_two_note_cues([(0.2, 0.5), (0.55, 0.9)], [110.0, 110.0]), dec.stride)
    dmax, _ = dec.pot.grid(e.frame_times)
    gold = gold_segments(e.frame_times, _notes([(0.2, 0.5), (0.55, 0.9)], [45, 45]), dmax)
    # No rest between the notes: everything from the first note to the last is NOTE.
    note_idx = [k for k, (lab, _, _) in enumerate(gold) if lab == NOTE]
    between = gold[note_idx[0] : note_idx[-1] + 1]
    assert all(lab == NOTE for lab, _, _ in between)


def _score_by_loop(pot, e, gold, dur_bucket):
    """The pre-vectorisation reference: one segment at a time, straight from the
    definition. Kept in the test rather than the library so the fast path is pinned
    against an independent implementation, not against itself."""
    score = e.onset_logit.new_zeros(())
    prev = START
    for label, i, j in gold:
        idx = torch.tensor([i], device=e.device)
        if label == NOTE:
            unary = pot.note_potentials(e, idx, j, dur_bucket)[0]
        else:
            unary = pot.rest_potentials(e, idx, j)[0]
        trans = pot.init[label] if prev == START else pot.trans[prev, label]
        score = score + unary + trans
        prev = label
    return score


def _fitted_potentials(dec):
    """Move the potentials off their symmetric init, so a bug that happens to cancel
    at w=1/bias=0 cannot pass."""
    with torch.no_grad():
        dec.pot.w_on.fill_(0.7)
        dec.pot.w_v.fill_(1.3)
        dec.pot.w_u.fill_(0.4)
        dec.pot.w_pitch.fill_(-0.9)
        dec.pot.b_note.fill_(0.25)
        dec.pot.b_rest.fill_(-0.4)
        dec.pot.dur.copy_(torch.linspace(-1.0, 1.0, dec.pot.dur.numel()))
        dec.pot.trans.copy_(torch.tensor([[0.3, -0.2], [-0.5, 0.1]]))
        dec.pot.init.copy_(torch.tensor([0.6, -0.3]))
    return dec


def test_gold_score_matches_the_per_segment_loop():
    # The vectorised gold_score evaluates both label branches and selects with
    # `where`; the loop it replaced branched. They must agree exactly.
    dec = _fitted_potentials(SemiCRFDecoder())
    cues = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    e = build_emissions(cues, dec.stride)
    dmax, dur_bucket = dec.pot.grid(e.frame_times)
    gold = gold_segments(e.frame_times, _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50]), dmax)
    assert len(gold) > 2  # the test is vacuous if gold is trivial

    fast = segmentation_score(dec.pot, e, gold, dur_bucket)
    ref = _score_by_loop(dec.pot, e, gold, dur_bucket)
    assert torch.allclose(fast, ref, atol=1e-5), f"{fast.item()} != {ref.item()}"


def test_gold_score_gradient_matches_the_per_segment_loop():
    # gold_score is half the NLL, so its gradient is what trains the potentials.
    dec = _fitted_potentials(SemiCRFDecoder())
    cues = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    e = build_emissions(cues, dec.stride)
    dmax, dur_bucket = dec.pot.grid(e.frame_times)
    gold = gold_segments(e.frame_times, _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50]), dmax)

    g_fast = torch.autograd.grad(
        segmentation_score(dec.pot, e, gold, dur_bucket), list(dec.pot.parameters()),
        retain_graph=True, allow_unused=True,
    )
    g_ref = torch.autograd.grad(
        _score_by_loop(dec.pot, e, gold, dur_bucket), list(dec.pot.parameters()),
        allow_unused=True,
    )
    for a, b in zip(g_fast, g_ref, strict=True):
        if a is None or b is None:
            assert a is None and b is None
        else:
            assert torch.allclose(a, b, atol=1e-5)


def test_gold_scores_batched_equals_per_track():
    # The batched form flattens every track's segments into one indexed set and folds
    # them back with index_add; a mis-grouped `first` mask would mis-assign the START
    # transition, which only shows up with more than one track.
    dec = _fitted_potentials(SemiCRFDecoder())
    a = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    na = _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50])
    short = Cues(
        contour=Contour(times=TIMES[:120], freqs=a.contour.freqs[:120]),
        onset=a.onset[:120], frame_times=TIMES[:120], voicing=a.voicing[:120],
    )
    ns = _notes([(0.2, 0.9)], [45])

    es = [build_emissions(c, dec.stride) for c in (a, short)]
    dmax, dur_bucket = dec.pot.grid(es[0].frame_times)
    golds = [gold_segments(e.frame_times, n, dmax) for e, n in zip(es, [na, ns])]

    batched = segmentation_scores(dec.pot, pad_emissions(es), golds, dur_bucket)
    single = torch.stack([
        segmentation_score(dec.pot, e, g, dur_bucket) for e, g in zip(es, golds)
    ])
    assert torch.allclose(batched, single, atol=1e-4)


def test_gold_scores_handles_a_track_with_no_segments():
    # An empty note list yields empty gold; index_add must leave that row at zero
    # rather than dropping the row and shifting every later track's score.
    dec = SemiCRFDecoder()
    cues = _two_note_cues([(0.2, 0.9)], [110.0])
    e = build_emissions(cues, dec.stride)
    dmax, dur_bucket = dec.pot.grid(e.frame_times)
    gold = gold_segments(e.frame_times, _notes([(0.2, 0.9)], [45]), dmax)

    scores = segmentation_scores(dec.pot, pad_emissions([e, e]), [[], gold], dur_bucket)
    assert scores[0].item() == 0.0
    assert torch.allclose(scores[1], segmentation_score(dec.pot, e, gold, dur_bucket), atol=1e-4)


def _truncate(e, n):
    """The first ``n`` DP frames of an Emissions, so chunks tile it exactly."""
    return Emissions(
        e.onset_logit[:n], e.Vc[: n + 1], e.Uc[: n + 1], e.Jc[: n + 1],
        e.frame_times[:n], e.contour,
    )


def _long_emissions(dec, seconds=25.0, seed=0):
    """A synthetic track long enough that ``dmax`` (115 DP frames, 4 s) fits several
    times over -- the module-level fixture is only 80 DP frames, shorter than one
    ``dmax``, so it cannot exercise a carry at all."""
    n = round(seconds / DT)
    times = np.arange(n) * DT
    rng = np.random.default_rng(seed)
    onset = rng.uniform(0.01, 0.99, n).astype(np.float32)
    voicing = rng.uniform(0.01, 0.99, n).astype(np.float32)
    freqs = np.where(voicing > 0.5, 110.0, 0.0).astype(np.float32)
    cues = Cues(
        contour=Contour(times=times, freqs=freqs),
        onset=onset, frame_times=times, voicing=voicing,
    )
    return build_emissions(cues, dec.stride)


def _chunks(e, size, dmax):
    """Split one track's Emissions into fixed-size chunks, each prefixed with up to
    ``dmax`` frames of left context.

    The context is not optional: the alpha carry covers a boundary-crossing span's
    *history*, but the span still reads cue values (``onset[i]``, ``Vc[j]-Vc[i]``) from
    before the chunk. Cumsums are re-based at each chunk's own frame 0, so only
    differences survive and the absolute offset never matters.

    Yields ``(chunk, context)`` -- ``context`` is short only for chunks whose overlap
    already reaches global frame 0."""
    out = []
    for s in range(0, e.T - size + 1, size):
        lo = max(0, s - dmax)
        out.append((
            Emissions(
                e.onset_logit[lo : s + size],
                e.Vc[lo : s + size + 1] - e.Vc[lo],
                e.Uc[lo : s + size + 1] - e.Uc[lo],
                e.Jc[lo : s + size + 1] - e.Jc[lo],
                e.frame_times[lo : s + size],
                e.contour,
            ),
            s - lo,
        ))
    return out


def _run_chunked(dec, e, size, dmax, dur_bucket):
    """Scan a track chunk by chunk, carrying alpha. Returns the cumulative log-mass."""
    carry, A = None, None
    for chunk, ctx in _chunks(e, size, dmax):
        A, carry = chunk_log_partition(
            dec.pot, pad_emissions([chunk]), dmax, dur_bucket, carry, ctx
        )
    return A


@pytest.mark.parametrize("n_chunks", [2, 5])
def test_chunked_carry_equals_the_whole_track_log_partition(n_chunks):
    # The property the whole truncated-BPTT design rests on: carrying the alpha window
    # across chunk boundaries reproduces the whole-track log Z exactly. If this drifts,
    # chunked training optimises a different objective than it reports.
    dec = _fitted_potentials(SemiCRFDecoder())
    e = _long_emissions(dec)
    dmax, dur_bucket = dec.pot.grid(e.frame_times)

    size = e.T // n_chunks
    trimmed = _truncate(e, size * n_chunks)  # chunks must tile it exactly
    whole = batched_log_partition(dec.pot, pad_emissions([trimmed]), dmax, dur_bucket)
    got = _run_chunked(dec, trimmed, size, dmax, dur_bucket)
    assert torch.allclose(got, whole, atol=1e-3), f"{got.item()} != {whole.item()}"


def test_chunked_carry_holds_for_chunks_shorter_than_dmax():
    # dmax spans several chunks here, so a span's candidate starts reach back through
    # more than one boundary -- where an off-by-one in the window or a short context
    # shows up and a single-boundary test would not.
    dec = _fitted_potentials(SemiCRFDecoder())
    e = _long_emissions(dec)
    dmax, dur_bucket = dec.pot.grid(e.frame_times)

    size = max(1, dmax // 3)
    n = e.T // size
    assert n > 3 and size < dmax, "the point is several chunks inside one dmax"
    trimmed = _truncate(e, size * n)
    whole = batched_log_partition(dec.pot, pad_emissions([trimmed]), dmax, dur_bucket)
    got = _run_chunked(dec, trimmed, size, dmax, dur_bucket)
    assert torch.allclose(got, whole, atol=1e-3), f"{got.item()} != {whole.item()}"


def test_dropping_the_left_context_changes_the_answer():
    # Guards the invariant the docstring states: the carry alone is not enough. Without
    # the overlap the scan clamps boundary-crossing starts to the chunk's own frame 0
    # and scores them against the wrong cues -- silently, so it needs an explicit test.
    dec = _fitted_potentials(SemiCRFDecoder())
    e = _long_emissions(dec)
    dmax, dur_bucket = dec.pot.grid(e.frame_times)
    size = e.T // 4
    trimmed = _truncate(e, size * 4)
    whole = batched_log_partition(dec.pot, pad_emissions([trimmed]), dmax, dur_bucket)

    carry, A = None, None
    for chunk, _ in _chunks(trimmed, size, 0):  # dmax=0 -> no overlap at all
        A, carry = chunk_log_partition(
            dec.pot, pad_emissions([chunk]), dmax, dur_bucket, carry, 0
        )
    assert not torch.allclose(A, whole, atol=1e-3), (
        "dropping the left context still matched, so the parity tests above prove "
        "nothing about the context"
    )


def test_detaching_the_carry_still_reaches_every_chunk():
    # Truncated BPTT: with the carry detached, a single end-of-track loss would only
    # reach the last chunk. The telescoping per-chunk loss must put gradient on every
    # chunk's own cues.
    dec = _fitted_potentials(SemiCRFDecoder())
    e = _long_emissions(dec)
    dmax, dur_bucket = dec.pot.grid(e.frame_times)
    size = e.T // 4
    trimmed = _truncate(e, size * 4)

    grads, carry, prev_A = [], None, 0.0
    for chunk, ctx in _chunks(trimmed, size, dmax):
        onset = chunk.onset_logit.detach().clone().requires_grad_(True)
        live = Emissions(
            onset, chunk.Vc, chunk.Uc, chunk.Jc, chunk.frame_times, chunk.contour
        )
        A, carry = chunk_log_partition(
            dec.pot, pad_emissions([live]), dmax, dur_bucket, carry, ctx
        )
        (A - prev_A).sum().backward()
        carry, prev_A = carry.detach(), A.detach()
        grads.append(onset.grad)

    assert len(grads) == 4
    for k, g in enumerate(grads):
        assert g is not None, f"chunk {k} got no gradient"
        assert g.abs().sum() > 0, f"chunk {k} gradient is all zero"


def test_telescoping_chunk_losses_sum_to_the_whole_track_log_partition():
    # The training loss is sum_k (A_k - A_{k-1}); with A_0 = 0 that must telescope to
    # the whole-track log Z, or the reported NLL is not the NLL being optimised.
    dec = _fitted_potentials(SemiCRFDecoder())
    e = _long_emissions(dec)
    dmax, dur_bucket = dec.pot.grid(e.frame_times)
    size = e.T // 4
    trimmed = _truncate(e, size * 4)
    whole = batched_log_partition(dec.pot, pad_emissions([trimmed]), dmax, dur_bucket)

    total, carry, prev_A = 0.0, None, 0.0
    for chunk, ctx in _chunks(trimmed, size, dmax):
        A, carry = chunk_log_partition(
            dec.pot, pad_emissions([chunk]), dmax, dur_bucket, carry, ctx
        )
        total = total + (A - prev_A)
        carry, prev_A = carry.detach(), A.detach()
    assert torch.allclose(total, whole, atol=1e-3), f"{total.item()} != {whole.item()}"


def test_chunk_log_partition_rejects_ragged_lengths():
    # A chunk reads its answer off the end of the window, so a short track would be
    # read at the wrong position rather than failing loudly.
    dec = SemiCRFDecoder()
    a = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    ea = build_emissions(a, dec.stride)
    short = Cues(
        contour=Contour(times=TIMES[:120], freqs=a.contour.freqs[:120]),
        onset=a.onset[:120], frame_times=TIMES[:120], voicing=a.voicing[:120],
    )
    dmax, dur_bucket = dec.pot.grid(ea.frame_times)
    batch = pad_emissions([ea, build_emissions(short, dec.stride)])
    with pytest.raises(ValueError, match="uniform lengths"):
        chunk_log_partition(dec.pot, batch, dmax, dur_bucket)


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
        assert dec.pot.w_on.grad is not None
        opt.step()
    assert dec.nll(cues, notes).item() < first.item()


def test_nll_grad_reaches_the_encoder_through_tensor_cues():
    # A stand-in "encoder": a learnable bias added to the onset logits. If the CRF
    # loss puts gradient on it through TensorCues, the same path reaches a real
    # backbone's heads -- this is what makes end-to-end training possible.
    dec = SemiCRFDecoder()
    cues = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    notes = _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50])
    onset_logit = torch.logit(torch.from_numpy(cues.onset).clamp(1e-6, 1 - 1e-6))
    voicing_logit = torch.logit(torch.from_numpy(cues.voicing).clamp(1e-6, 1 - 1e-6))
    bias = torch.zeros(1, requires_grad=True)  # the "encoder parameter"

    tcues = TensorCues.from_logits(
        cues.contour, onset_logit + bias, cues.frame_times, voicing_logit
    )
    dec.nll(tcues, notes).backward()
    assert bias.grad is not None and bias.grad.abs().item() > 0


def test_numpy_cues_nll_does_not_reach_the_inputs():
    # The inference path: a numpy Cues is a constant, so the loss is differentiable
    # only in the CRF's own potentials -- never the (already-computed) cues.
    dec = SemiCRFDecoder()
    cues = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    e = build_emissions(cues, dec.stride)
    assert not e.onset_logit.requires_grad


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


def test_batched_nll_equals_single_track_padded():
    # The batched scan over a padded minibatch must equal the per-track scan, track
    # for track -- padding (different lengths) must not leak into a track's log Z.
    dec = SemiCRFDecoder()
    a = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    na = _notes([(0.2, 0.9), (1.2, 1.9)], [45, 50])
    # a shorter track, so the batch pads
    short = Cues(
        contour=Contour(times=TIMES[:120], freqs=a.contour.freqs[:120]),
        onset=a.onset[:120], frame_times=TIMES[:120], voicing=a.voicing[:120],
    )
    ns = _notes([(0.2, 0.9)], [45])

    batched = dec.nll_batch([a, short], [na, ns])
    single = torch.stack([dec.nll(a, na), dec.nll(short, ns)])
    assert torch.allclose(batched, single, atol=1e-4)


def test_batched_nll_grad_survives_padding():
    # The padded-batch build scatters graph tensors into a zeros buffer; check grad
    # still reaches an encoder stand-in through the batched scan.
    dec = SemiCRFDecoder()
    a = _two_note_cues([(0.2, 0.9), (1.2, 1.9)], [110.0, 146.0])
    bias = torch.zeros(1, requires_grad=True)
    onset_logit = torch.logit(torch.from_numpy(a.onset).clamp(1e-6, 1 - 1e-6)) + bias
    voicing_logit = torch.logit(torch.from_numpy(a.voicing).clamp(1e-6, 1 - 1e-6))
    tcues = TensorCues.from_logits(a.contour, onset_logit, a.frame_times, voicing_logit)

    short = _two_note_cues([(0.3, 0.8)], [98.0])
    dec.nll_batch(
        [tcues, short], [_notes([(0.2, 0.9), (1.2, 1.9)], [45, 50]), _notes([(0.3, 0.8)], [43])]
    ).mean().backward()
    assert bias.grad is not None and bias.grad.abs().item() > 0


# --- device placement -------------------------------------------------------------
# The semi-CRF had never run off CPU: both entry points that build a NeuralEncoder
# hardcode device="cpu", and there is no semi-CRF training entry point yet, so nothing
# ever called .to() on the potentials. Two fields were stranded on the host as a
# result -- the dur_edges_s buffer (read via .numpy(), which raises once moved) and Jc
# (built from the numpy contour). These tests are what keep that from returning.


def _tensor_cues(cues: Cues, device, bias: torch.Tensor | None = None) -> TensorCues:
    """``cues`` as the grad-bearing TensorCues an encoder on ``device`` would emit.
    ``bias`` stands in for an encoder parameter, so grad has somewhere to land."""
    onset_logit = torch.logit(torch.from_numpy(cues.onset).clamp(1e-6, 1 - 1e-6)).to(device)
    voicing_logit = torch.logit(torch.from_numpy(cues.voicing).clamp(1e-6, 1 - 1e-6)).to(device)
    if bias is not None:
        onset_logit = onset_logit + bias
    return TensorCues.from_logits(
        cues.contour, onset_logit, cues.frame_times, voicing_logit
    )


def _fitted(cues: Cues, notes: list[NoteEvent], steps: int = 120) -> SemiCRFDecoder:
    dec = SemiCRFDecoder()
    opt = torch.optim.Adam(dec.parameters(), lr=0.2)
    for _ in range(steps):
        opt.zero_grad()
        dec.nll(cues, notes).backward()
        opt.step()
    return dec


def test_numpy_cues_decode_on_device(device):
    # The offline-eval path: numpy Cues have no device of their own, so the module's
    # placement is the authority and the emissions must follow the parameters.
    dec = SemiCRFDecoder().to(device)
    cues, notes = _two_note_cues(SPANS, HZ), _notes(SPANS, MIDI)
    nll = dec.nll(cues, notes)
    assert nll.device.type == device.type
    assert isinstance(dec.decode(cues), list)


def test_tensor_cues_grad_reaches_encoder_on_device(device):
    # The training path: heads on `device` feeding a decoder on `device`, end to end.
    dec = SemiCRFDecoder().to(device)
    bias = torch.zeros(1, device=device, requires_grad=True)
    tcues = _tensor_cues(_two_note_cues(SPANS, HZ), device, bias)
    dec.nll(tcues, _notes(SPANS, MIDI)).backward()
    assert bias.grad is not None and bias.grad.abs().item() > 0
    assert dec.pot.w_on.grad is not None


def test_nll_batch_on_device(device):
    # The batched scan is what SemiCRFLoss actually calls, so it needs its own sweep.
    dec = SemiCRFDecoder().to(device)
    full = _two_note_cues(SPANS, HZ)
    short = _two_note_cues([(0.3, 0.8)], [98.0])
    out = dec.nll_batch(
        [_tensor_cues(full, device), _tensor_cues(short, device)],
        [_notes(SPANS, MIDI), _notes([(0.3, 0.8)], [43])],
    )
    assert out.shape == (2,) and out.device.type == device.type


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
def test_cuda_nll_matches_cpu():
    # Parity, not just liveness: the same potentials must score the same track the
    # same on either backend, or a "device fix" has changed the model.
    cues, notes = _two_note_cues(SPANS, HZ), _notes(SPANS, MIDI)
    cpu = SemiCRFDecoder()
    gpu = SemiCRFDecoder()
    gpu.load_state_dict(cpu.state_dict())
    assert torch.allclose(cpu.nll(cues, notes), gpu.to("cuda").nll(cues, notes).cpu(),
                          atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
def test_cuda_decode_matches_cpu():
    # Viterbi picks discrete boundaries, so a fitted model must decode to the *same*
    # notes on both backends -- onsets are read off the frame grid, not interpolated.
    cues, notes = _two_note_cues(SPANS, HZ), _notes(SPANS, MIDI)
    cpu = _fitted(cues, notes)
    gpu = SemiCRFDecoder()
    gpu.load_state_dict(cpu.state_dict())
    gpu = gpu.to("cuda")

    on_cpu, on_gpu = cpu.decode(cues), gpu.decode(cues)
    assert len(on_cpu) == len(on_gpu) == 2
    for a, b in zip(on_cpu, on_gpu):
        assert a.onset == pytest.approx(b.onset)
        assert a.offset == pytest.approx(b.offset)
        assert a.midi_pitch == pytest.approx(b.midi_pitch, abs=1e-4)


def test_emissions_rejects_fields_on_different_devices():
    # The invariant's guard. A "meta" tensor is device-distinct from CPU without
    # needing an accelerator, so this runs everywhere -- including the laptop where
    # the original bug was invisible.
    t, cum = torch.zeros(4), torch.zeros(5)
    with pytest.raises(ValueError, match="span multiple devices"):
        Emissions(t, cum, cum, torch.zeros(5, device="meta"), np.zeros(4), Contour.empty())


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
