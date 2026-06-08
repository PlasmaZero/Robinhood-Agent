"""Unit tests for the five factor signals."""

from __future__ import annotations

import numpy as np

from robinhood_agent import signals as sg
from robinhood_agent import synthetic as syn
from robinhood_agent.market_data import make_bars
from robinhood_agent.signals import SignalParams


def _bars_from_close(close: np.ndarray, volume: float = 5_000_000.0):
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    vol = np.full(len(close), volume)
    return make_bars(open_, high, low, close, vol)


def _all_in_bounds(s: sg.SignalResult):
    assert -1.0 <= s.score <= 1.0
    assert 0.0 <= s.confidence <= 1.0


def test_momentum_bullish_on_uptrend_bearish_on_downtrend():
    p = SignalParams()
    up = sg.momentum_signal(syn.trending_series(drift=+0.005), p)
    down = sg.momentum_signal(syn.trending_series(drift=-0.005, seed=3), p)
    assert up.score > 0.3 and up.confidence > 0.3
    assert down.score < -0.3 and down.confidence > 0.3
    _all_in_bounds(up)
    _all_in_bounds(down)


def test_orderflow_follows_volume():
    p = SignalParams()
    up = sg.orderflow_signal(syn.trending_series(drift=+0.005), p)
    down = sg.orderflow_signal(syn.trending_series(drift=-0.005, seed=3), p)
    assert up.score > 0
    assert down.score < 0


def test_liquidity_gate_high_for_liquid_low_for_thin():
    p = SignalParams()
    liquid = sg.liquidity_signal(syn.trending_series(base_volume=5_000_000.0), p)
    thin = sg.liquidity_signal(syn.thin_uptrend(), p)
    assert liquid.confidence > 0.9  # deep, tradeable
    assert thin.confidence < 0.1  # microcap suppressed


def test_support_resistance_breakout_and_breakdown():
    p = SignalParams()
    up = sg.support_resistance_signal(syn.trending_series(drift=+0.005), p)
    down = sg.support_resistance_signal(syn.trending_series(drift=-0.006, noise=0.003, seed=1), p)
    assert up.score > 0  # new highs read bullish, not "at resistance"
    assert down.score < 0  # new lows read bearish


def test_fibonacci_bullish_in_golden_zone_pullback():
    # Rally 100 -> 200 over 100 bars, then retrace to the 38.2% level (161.8).
    rally = np.linspace(100.0, 200.0, 100)
    pull = np.linspace(200.0, 161.8, 40)
    close = np.concatenate([rally, pull])
    bars = _bars_from_close(close)
    s = sg.fibonacci_signal(bars, SignalParams())
    assert s.score > 0.4  # buy-the-dip in an uptrend
    assert s.detail["uptrend"] is True
    assert 0.3 < s.detail["retracement"] < 0.7


def test_signals_neutral_on_insufficient_history():
    # Fewer bars than any factor's minimum window (liquidity's is the smallest).
    short = _bars_from_close(np.linspace(100, 110, 15))
    for fn in (
        sg.momentum_signal,
        sg.support_resistance_signal,
        sg.fibonacci_signal,
        sg.orderflow_signal,
        sg.liquidity_signal,
    ):
        s = fn(short, SignalParams())
        assert s.score == 0.0 and s.confidence == 0.0


def test_compute_all_signals_returns_five_in_bounds():
    results = sg.compute_all_signals(syn.trending_series(drift=+0.004))
    assert [r.name for r in results] == [
        "momentum",
        "support_resistance",
        "fibonacci",
        "orderflow",
        "liquidity",
    ]
    for r in results:
        _all_in_bounds(r)
        assert np.isfinite(r.evidence)
