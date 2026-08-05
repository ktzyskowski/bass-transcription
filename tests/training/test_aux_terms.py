"""Auxiliary terms declare what they are, so nothing has to guess from a head's name.

The kind of a term used to be a string in a table, read by an ``if`` in the loss and
restated as ``h != "pitch"`` in the CLI. Two places encoding one fact, and neither able
to answer the question that actually matters -- *does this term accept a class weight?*
So ``--pos-weight pitch=5`` was caught by a hardcoded comparison while
``--pos-weight typo=5``, and a weight naming a head the run scores with weight 0, were
both stored in the checkpoint and never read.

Three different mistakes with one symptom: nothing happens, and no loss curve shows it.
"""

import pytest
import torch

from stemify.training.semicrf.loss import SemiCRFLoss
from stemify.training.semicrf.terms import (
    AUX_TERMS,
    BinaryTerm,
    CategoricalTerm,
)
from stemify.decoders.semicrf import SemiCRFDecoder


def _loss(**kwargs):
    return SemiCRFLoss(SemiCRFDecoder(), **kwargs)


# --------------------------------------------------------------------------- #
# What a term says about itself                                               #
# --------------------------------------------------------------------------- #
def test_only_the_binary_term_takes_a_class_weight():
    """A categorical CE over 216 bins has no positive class: every voiced frame has
    exactly one correct bin, so the imbalance is between bins, not between present and
    absent."""
    assert BinaryTerm().takes_pos_weight is True
    assert CategoricalTerm().takes_pos_weight is False


def test_every_declared_head_has_a_term():
    assert set(AUX_TERMS) == {"pitch", "onset", "voicing"}
    assert isinstance(AUX_TERMS["pitch"], CategoricalTerm)
    assert all(isinstance(AUX_TERMS[h], BinaryTerm) for h in ("onset", "voicing"))


def test_the_binary_term_forwards_its_weight():
    """Not decoration -- if the weight were dropped the term still runs, still falls,
    and trains unweighted while the run's logged params claim otherwise."""
    logits = torch.zeros(2, 8)
    target = torch.zeros(2, 8)
    target[:, :2] = 1.0
    plain = BinaryTerm()(logits, target)
    weighted = BinaryTerm()(logits, target, pos_weight=5.0)
    assert float(weighted) > float(plain)


def test_the_categorical_term_ignores_a_weight_rather_than_crashing():
    """It is refused upstream, so reaching here means a caller bypassed validation;
    the term must still be a total function of its arguments."""
    logits = torch.randn(2, 4, 216)
    target = torch.full((2, 4), 40.0)
    assert torch.allclose(
        CategoricalTerm()(logits, target),
        CategoricalTerm()(logits, target, pos_weight=5.0),
    )


# --------------------------------------------------------------------------- #
# The three refusals the string table could not express                       #
# --------------------------------------------------------------------------- #
def test_a_class_weight_on_an_unknown_head_is_refused():
    """Previously stored and never read. `_parse_pos_weight` checks against `HEADS`,
    which is the *frame* path's head list -- it does not know which of them this
    objective scores."""
    with pytest.raises(ValueError, match="no auxiliary term"):
        _loss(aux_heads=["onset"], pos_weight={"nonsense": 5.0})


def test_a_class_weight_on_a_categorical_term_is_refused():
    """The case the CLI caught by comparing against the literal 'pitch'. Asked of the
    term instead, so a *second* categorical head would be caught too."""
    with pytest.raises(ValueError, match="no positive class"):
        _loss(aux_heads=["pitch"], pos_weight={"pitch": 5.0})


def test_a_class_weight_on_a_head_this_run_does_not_score_is_refused():
    """The subtlest of the three: a real head, a real binary term, a weight that means
    something -- for a term this run never computes. It would ride into the checkpoint
    and describe a training that did not happen."""
    with pytest.raises(ValueError, match="does not score"):
        _loss(aux_heads=["voicing"], pos_weight={"onset": 5.0})


def test_a_class_weight_on_a_scored_binary_head_is_accepted():
    """The refusals must not have swallowed the working case."""
    objective = _loss(aux_heads=["onset", "voicing"], pos_weight={"onset": 5.0})
    assert objective.pos_weight == {"onset": 5.0}
