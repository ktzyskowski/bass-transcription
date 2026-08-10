"""How the standard encoder is assembled: backbone, heads, and what decides each.

One function, and it exists because *two* recipes need the same network. The frame
objective and the semi-CRF differ in what they optimise, not in what they optimise --
same TCN, same heads, same front-end. Previously ``SemiCRFRecipe.build_model`` obtained
that by constructing a whole ``FrameRecipe`` and calling its ``build_model``, which is
the only reason an extension ever imported the framework's other extension. Assembly is
a fact about ``models``, so it lives here and neither recipe knows the other exists.
"""

from collections.abc import Sequence

from stemify.models.framenet import FrameNet
from stemify.models.heads import onset_head, pitch_head, voicing_head
from stemify.models.networks import TCN


def build_framenet(
    in_channels: int, harmonics: Sequence[int] = (), pitch: bool = True
) -> FrameNet:
    """The encoder every objective trains: a TCN backbone with onset/voicing heads,
    and a ``FrequencyHead`` when the run predicts pitch.

    ``in_channels`` is the condition's channel count -- one for a bass stem, two when a
    mixture rides alongside it. ``harmonics`` stacks each note's partials as extra
    front-end channels.
    """
    backbone = TCN(in_channels=in_channels, harmonics=harmonics)
    return FrameNet(
        backbone,
        {
            "onset": onset_head(backbone.hidden),
            "voicing": voicing_head(backbone.hidden),
        },
        # A FrequencyHead takes no `hidden`: it taps the front-end before the flatten,
        # reading (B, C, F, T) with the frequency axis still intact. The head's width is
        # the front-end's *cropped* axis, not `N_BINS` -- harmonic stacking spends the
        # top of the fundamental axis on the channels below it.
        frequency_heads=(
            {"pitch": pitch_head(n_bins=backbone.frontend.n_bins)} if pitch else None
        ),
    )
