"""Unit tests for the low-level indicators."""

from __future__ import annotations

import numpy as np

from robinhood_agent import indicators as ind


def test_sma_matches_manual():
    x = np.array([1.0, 2, 3, 4, 5])
    out = ind.sma(x, 3)
    assert np.isnan(out[0]) and np.isnan(out[1])
    np.testing.assert_allclose(out[2:], [2.0, 3.0, 4.0])


def test_ema_tracks_and_is_finite():
    x = np.arange(1, 21, dtype=float)
    out = ind.ema(x, 5)
    assert np.all(np.isfinite(out))
    assert out[-1] < x[-1]  # EMA lags a rising series


def test_rsi_bounds_and_extremes():
    up = np.cumsum(np.ones(60)) + 100  # strictly increasing
    down = 200 - np.cumsum(np.ones(60))  # strictly decreasing
    r_up = ind.rsi(up, 14)
    r_down = ind.rsi(down, 14)
    assert ind.last_finite(r_up) > 99  # all gains -> ~100
    assert ind.last_finite(r_down) < 1  # all losses -> ~0
    finite = r_up[~np.isnan(r_up)]
    assert finite.min() >= 0 and finite.max() <= 100


def test_atr_positive():
    n = 60
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + 1.0
    low = close - 1.0
    a = ind.atr(high, low, close, 14)
    assert ind.last_finite(a) > 0


def test_obv_rises_on_up_series():
    close = np.array([10.0, 11, 12, 13, 14])
    vol = np.array([100.0, 100, 100, 100, 100])
    o = ind.obv(close, vol)
    assert o[-1] == 400.0  # every step up adds volume


def test_mfi_bounds():
    n = 40
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + 0.5
    low = close - 0.5
    vol = np.full(n, 1000.0)
    m = ind.mfi(high, low, close, vol, 14)
    finite = m[~np.isnan(m)]
    assert finite.min() >= 0 and finite.max() <= 100


def test_roc_value():
    x = np.array([100.0, 0, 0, 0, 0, 110.0])
    out = ind.roc(x, 5)
    assert abs(ind.last_finite(out) - 10.0) < 1e-6  # +10% over 5 bars


def test_swing_detection_finds_pivots():
    # A clean zig-zag: peak at index 3, trough at index 7.
    series = np.array([1.0, 2, 3, 5, 3, 2, 1, 0.5, 1.5, 2.5, 3.5])
    highs = ind.swing_high_indices(series, 2, 2)
    lows = ind.swing_low_indices(series, 2, 2)
    assert 3 in highs
    assert 7 in lows


def test_linreg_slope_sign():
    up = np.arange(20, dtype=float)
    down = np.arange(20, 0, -1, dtype=float)
    assert ind.linreg_slope(up, 10) > 0
    assert ind.linreg_slope(down, 10) < 0


def test_last_finite_handles_all_nan():
    assert np.isnan(ind.last_finite(np.array([np.nan, np.nan])))
