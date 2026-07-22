"""EMA smoothing: seeds on the first sample, then decays toward new values, and
rejects a beta outside [0, 1)."""

import pytest

from stemify.training.ema import EMA


def test_first_update_seeds_value():
    ema = EMA(beta=0.9)
    assert ema.value is None
    assert ema.update(5.0) == 5.0  # no zero-biased warmup: first sample is the value
    assert ema.value == 5.0


def test_tracks_toward_new_samples_but_lags():
    ema = EMA(beta=0.5)
    ema.update(0.0)
    v = ema.update(10.0)          # 0.5*0 + 0.5*10
    assert v == pytest.approx(5.0)
    assert ema.update(10.0) == pytest.approx(7.5)  # lags a step behind a step change


def test_higher_beta_smooths_harder():
    smooth, jumpy = EMA(beta=0.99), EMA(beta=0.5)
    for e in (smooth, jumpy):
        e.update(0.0)
    smooth.update(1.0)
    jumpy.update(1.0)
    assert smooth.value < jumpy.value  # more retention -> moves less toward the spike


@pytest.mark.parametrize("beta", [-0.1, 1.0, 1.5])
def test_rejects_out_of_range_beta(beta):
    with pytest.raises(ValueError):
        EMA(beta=beta)
