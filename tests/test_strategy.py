"""Tests for the decision aggregation and the 65% threshold contract."""

from __future__ import annotations

import numpy as np

from robinhood_agent import synthetic as syn
from robinhood_agent.market_data import make_bars
from robinhood_agent.strategy import (
    BUY,
    HOLD,
    SELL,
    StrategyParams,
    decide,
    suggested_notional,
)


def test_strong_uptrend_triggers_buy_above_threshold():
    d = decide("UP", syn.trending_series(drift=+0.005))
    assert d.action == BUY
    assert d.confidence >= 0.65
    assert d.direction == 1
    # Risk levels are well-formed for a long.
    assert d.stop_price < d.price < d.target_price
    assert d.risk_reward >= 1.5


def test_clean_breakdown_triggers_sell_above_threshold():
    d = decide("DN", syn.trending_series(drift=-0.006, noise=0.003, seed=1))
    assert d.action == SELL
    assert d.confidence >= 0.65
    assert d.direction == -1


def test_choppy_market_holds_below_threshold():
    for seed in (11, 4, 23):
        d = decide("CHOP", syn.choppy_series(seed=seed))
        assert d.action == HOLD
        assert d.confidence < 0.65


def test_weak_trend_holds():
    d = decide("WEAK", syn.trending_series(drift=+0.0015, noise=0.012, seed=21))
    assert d.action == HOLD
    assert d.confidence < 0.65


def test_thin_liquidity_suppresses_even_a_clean_trend():
    d = decide("THIN", syn.thin_uptrend())
    assert d.liquidity_gate < 0.1
    assert d.action == HOLD  # gate multiplies confidence toward zero


def test_insufficient_history_holds():
    close = np.linspace(100, 110, 30)
    bars = make_bars(close, close * 1.01, close * 0.99, close, np.full(30, 1e6))
    d = decide("SHORT", bars)
    assert d.action == HOLD
    assert "bars" in d.rationale


def test_confidence_always_in_unit_interval():
    for gen in (
        syn.trending_series(drift=+0.005),
        syn.trending_series(drift=-0.005, seed=2),
        syn.choppy_series(),
        syn.pullback_in_uptrend(),
    ):
        d = decide("X", gen)
        assert 0.0 <= d.confidence <= 1.0
        assert d.contributions  # every factor contributed


def test_threshold_is_configurable():
    bars = syn.trending_series(drift=+0.0025, noise=0.006, seed=8)
    strict = decide("X", bars, StrategyParams(threshold=0.95))
    loose = decide("X", bars, StrategyParams(threshold=0.20))
    assert strict.confidence == loose.confidence  # same evidence
    assert strict.action == HOLD  # ... but the bar to act differs
    assert loose.action in (BUY, SELL)


def test_suggested_notional_scales_with_confidence():
    base = decide("UP", syn.trending_series(drift=+0.005))
    max_order, floor = 500.0, 0.5

    at_floor = base.__class__(**{**base.__dict__, "confidence": base.threshold})
    at_top = base.__class__(**{**base.__dict__, "confidence": 1.0})
    hold = base.__class__(**{**base.__dict__, "action": HOLD})

    n_floor = suggested_notional(at_floor, max_order, floor)
    n_top = suggested_notional(at_top, max_order, floor)
    assert abs(n_floor - max_order * floor) < 1e-6
    assert abs(n_top - max_order) < 1e-6
    assert n_floor < suggested_notional(base, max_order, floor) <= n_top
    assert suggested_notional(hold, max_order, floor) == 0.0
