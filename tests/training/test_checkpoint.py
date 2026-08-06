"""``Checkpoint``: the one way a trained system comes back off disk.

The bug this type exists to make impossible: the saved file and the in-memory snapshot
named the same content differently (``state_dict``/``loss_state`` against
``model``/``loss``), each reader translated by hand, and the eval CLI's hand translation
dropped the semi-CRF's potentials. A default-constructed decoder has ``dur=zeros`` and
still emits notes, so the failure produced numbers instead of an error.

The load-bearing test here is the last one: score in process, save, reload through the
deployment path, score again, assert equal. It does not care *how* a future divergence
arises -- a renamed key, an unsaved field, a decoder rebuilt from defaults -- only that
the thing on disk transcribes like the thing in memory.
"""

import logging

import numpy as np
import pytest
import torch

from stemify.core.contour import Contour
from stemify.core.cues import Cues
from stemify.semicrf import SemiCRFDecoder
from stemify.core.notes import NoteEvent
from stemify.training.checkpoint import VERSION, Checkpoint
from stemify.training.frame.recipe import FrameRecipe
from stemify.semicrf.fit.recipe import SemiCRFRecipe

DT = 0.0116


def _trained_potentials() -> dict[str, torch.Tensor]:
    """Potentials visibly off their defaults, so a lost restore cannot pass."""
    decoder = SemiCRFDecoder()
    with torch.no_grad():
        decoder.scorer.w_on.fill_(0.31)
        decoder.scorer.b_note.fill_(-0.77)
        decoder.scorer.dur.copy_(torch.linspace(-1.0, 1.0, decoder.scorer.dur.numel()))
    return {k: v.detach().clone() for k, v in decoder.state_dict().items()}


def _semicrf_checkpoint() -> Checkpoint:
    recipe = SemiCRFRecipe(["clean"])
    return Checkpoint(
        recipe=recipe,
        model_state=recipe.build_model().state_dict(),
        objective_state=_trained_potentials(),
    )


def test_save_load_round_trips_the_recipe_and_both_states(tmp_path):
    original = _semicrf_checkpoint()
    path = tmp_path / "ck.pt"
    original.save(path)
    back = Checkpoint.load(path)

    assert back.recipe.name == original.recipe.name
    assert back.recipe.config == original.recipe.config
    for key, want in original.objective_state.items():
        assert torch.allclose(back.objective_state[key], want)


def test_the_decoder_comes_back_trained_not_default(tmp_path):
    # The actual defect. `dur` all-zeros is "no duration preference", which is what a
    # fresh decoder has and the opposite of what this objective exists to learn.
    path = tmp_path / "ck.pt"
    _semicrf_checkpoint().save(path)
    dur = Checkpoint.load(path).decoder().scorer.dur
    assert not torch.allclose(dur, torch.zeros_like(dur))


def test_frame_checkpoint_scores_through_the_heuristic(tmp_path):
    # The recipe owns which decoder scores it, so no caller passes a flag that could
    # disagree with how the model was trained.
    recipe = FrameRecipe(["clean"], pitch=True)
    path = tmp_path / "frame.pt"
    Checkpoint(recipe, recipe.build_model().state_dict(), {}).save(path)
    back = Checkpoint.load(path)
    assert back.decoder().name == "heuristic"
    assert back.recipe.heads == ["onset", "voicing", "pitch"]


def test_a_version_1_checkpoint_loads_and_says_what_it_inferred(tmp_path, caplog):
    # Files written before this type existed carry derived facts and no recipe. They
    # must still load -- but every guess has to be visible, because a wrong one is
    # exactly the silent mis-scoring this type was introduced to end.
    recipe = SemiCRFRecipe(["clean"])
    path = tmp_path / "v1.pt"
    torch.save(
        {
            "state_dict": recipe.build_model().state_dict(),
            "loss_state": _trained_potentials(),
            "heads": ["onset", "voicing"],
            "in_channels": 1,
            "conditions": ["clean"],
        },
        path,
    )
    with caplog.at_level(logging.WARNING):
        back = Checkpoint.load(path)
    assert back.recipe.name == "semicrf"          # inferred from having potentials
    assert "version-1" in caplog.text and "inferring" in caplog.text
    dur = back.decoder().scorer.dur
    assert not torch.allclose(dur, torch.zeros_like(dur))


def test_saved_file_records_its_version(tmp_path):
    path = tmp_path / "ck.pt"
    _semicrf_checkpoint().save(path)
    assert torch.load(path, map_location="cpu", weights_only=False)["version"] == VERSION


def _cues(n: int = 120) -> Cues:
    ft = np.arange(n) * DT
    rng = np.random.default_rng(0)
    return Cues(
        contour=Contour(times=ft, freqs=np.full(n, 110.0, dtype=np.float32)),
        onset=rng.random(n).astype(np.float32),
        frame_times=ft,
        voicing=np.full(n, 0.9, dtype=np.float32),
    )


def _notes(decoder, cues) -> list[NoteEvent]:
    return decoder.decode(cues)


def test_a_reloaded_checkpoint_decodes_identically_to_the_one_in_memory(tmp_path):
    # THE invariant. Whatever a future change does -- renames a key, forgets a field,
    # rebuilds geometry from defaults -- if the thing on disk stops transcribing like
    # the thing in memory, this fails. It is mechanism-agnostic on purpose: both bugs
    # this type was written for would have tripped it, and neither had a unit test that
    # could, because each half worked in isolation.
    checkpoint = _semicrf_checkpoint()
    cues = _cues()

    before = _notes(checkpoint.decoder(), cues)
    path = tmp_path / "ck.pt"
    checkpoint.save(path)
    after = _notes(Checkpoint.load(path).decoder(), cues)

    assert len(before) == len(after), f"{len(before)} notes in memory, {len(after)} from disk"
    for a, b in zip(before, after, strict=True):
        assert a.onset == pytest.approx(b.onset)
        assert a.offset == pytest.approx(b.offset)
        assert a.midi_pitch == pytest.approx(b.midi_pitch)


def test_swapping_the_two_state_dicts_is_refused_at_save(tmp_path):
    # model_state and objective_state are both dict[str, Tensor], so the type system
    # cannot tell them apart and a swap writes a plausible file. Without this the
    # mistake surfaces at load -- after the training run that made it has ended.
    recipe = SemiCRFRecipe(["clean"])
    swapped = Checkpoint(
        recipe=recipe,
        model_state=_trained_potentials(),               # the objective's, not the model's
        objective_state=recipe.build_model().state_dict(),
    )
    with pytest.raises(ValueError, match="wrong order"):
        swapped.save(tmp_path / "swapped.pt")


def test_a_truncated_model_state_is_refused_at_save(tmp_path):
    # Same guard, different cause: a partial state dict (a filtered save, a renamed
    # layer) would load non-strictly somewhere else and leave branches at their init.
    recipe = SemiCRFRecipe(["clean"])
    full = recipe.build_model().state_dict()
    partial = {k: v for i, (k, v) in enumerate(full.items()) if i % 2 == 0}
    with pytest.raises(ValueError, match="does not match"):
        Checkpoint(recipe, partial, {}).save(tmp_path / "partial.pt")
