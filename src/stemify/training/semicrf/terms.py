"""What an auxiliary per-frame term *is*, so that nothing has to guess.

The structured NLL under-supervises the heads it reads -- measured, onset AP 0.20-0.53
and voicing AUROC 0.51 against frame BCE's 0.994/0.995 for the same head -- so each
gets a per-frame term alongside. They are not interchangeable: pitch is a categorical
cross-entropy over CQT bins scored on voiced frames, the binary heads are BCE over every
frame.

That difference used to be a **string in a table** (``"categorical"`` / ``"binary"``)
read by an ``if`` in the loss, with the same knowledge restated as ``h != "pitch"`` in
the CLI's validation. Two places encoding one fact, and neither able to answer the
question that matters -- *does this term accept a class weight?* -- so
``--pos-weight pitch=5`` was refused by a hardcoded comparison while
``--pos-weight typo=5`` and a weight naming a head with weight 0 both passed silently
and did nothing.

A term declares it instead. Adding a head is a line in ``AUX_TERMS``, and every check
follows from what the term says about itself.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch

from stemify.models.heads import frame_bce, pitch_ce


class AuxTerm(ABC):
    """One per-frame auxiliary loss, over a head's logits and its targets."""

    takes_pos_weight: bool = False
    """Whether a positive-class weight means anything to this term.

    Declared rather than inferred from the head's name. It is what lets a class weight
    naming a term that has no positive class be *refused* instead of stored and never
    read -- the failure mode this package's docstrings keep returning to, because a
    setting that silently does nothing shows up in no loss curve."""

    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        pos_weight: float | None = None,
    ) -> torch.Tensor:
        """The term's value. ``target`` may carry NaN past a track's end; both
        implementations mask on it rather than propagating."""


class BinaryTerm(AuxTerm):
    """Per-frame BCE -- the onset and voicing heads.

    The *same* loss the frame path runs, which is the point: the semi-CRF's heads were
    reaching a fraction of the quality the identical architecture reached under frame
    BCE, so the fix is that loss, not a new one.
    """

    takes_pos_weight = True

    def __call__(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        pos_weight: float | None = None,
    ) -> torch.Tensor:
        return frame_bce(logits, target, pos_weight)


class CategoricalTerm(AuxTerm):
    """Cross-entropy over pitch bins -- the frequency head.

    No positive class to weight: every voiced frame has exactly one correct bin out of
    216, so imbalance is between *bins*, not between present and absent. Scaling the
    whole term is ``--pitch-weight``, which is a different knob answering a different
    question.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        pos_weight: float | None = None,
    ) -> torch.Tensor:
        return pitch_ce(logits, target)


AUX_TERMS: Mapping[str, AuxTerm] = {
    "pitch": CategoricalTerm(),
    "onset": BinaryTerm(),
    "voicing": BinaryTerm(),
}
"""The head each auxiliary term scores. A head absent here has no term and is refused.

Stateless singletons: a term is an algebra over logits and targets, not a configuration.
"""
