"""``Sources``: a track's audio by role, and what happens when a role is absent.

Small, but the absent-role behaviour is load-bearing. ``role`` is called while
assembling a model input channel, and a ``None`` returned there would become a
zero-filled channel -- an encoder reads a missing mixture as a *silent* one, which
is a perfectly coherent input it will happily train on.
"""

from pathlib import Path

import pytest

from stemify.core.sources import Sources


def test_role_returns_the_audio_as_a_path():
    s = Sources(bass="stem.flac", mix="mix.flac")
    assert s.role("bass") == Path("stem.flac")
    assert s.role("mix") == Path("mix.flac")


def test_an_absent_role_raises_rather_than_yielding_none():
    with pytest.raises(ValueError, match="no 'mix' audio"):
        Sources(bass="stem.flac").role("mix")


def test_an_unknown_role_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="unknown source role"):
        Sources(bass="stem.flac").role("drums")


def test_a_method_name_is_not_a_role():
    # getattr alone would happily return the bound method and the caller would
    # featurise a <bound method> -- the lookup is restricted to the dataclass fields.
    with pytest.raises(ValueError, match="unknown source role"):
        Sources(bass="stem.flac").role("role")
