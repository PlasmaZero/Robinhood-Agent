"""Tests for the backtester — correctness of mechanics, not profitability.

The headline test is `test_no_lookahead_bias`: the equity path up to bar K must
be identical whether or not bars after K exist. If a future bar could change a
past decision, that test fails — which is the bug that makes backtests lie.
"""

from __future__ import annotations

import numpy as np

from robinhood_agent import synthetic as syn
from robinhood_agent.backtest import load_long_csv, run_backtest
from robinhood_agent.market_data import Bars, make_bars


def _concat(*bars_list: Bars) -> Bars:
    return Bars(
        open=np.concatenate([b.open for b in bars_list]),
        high=np.concatenate([b.high for b in bars_list]),
        low=np.concatenate([b.low for b in bars_list]),
        close=np.concatenate([b.close for b in bars_list]),
        volume=np.concatenate([b.volume for b in bars_list]),
        symbol="SYN",
    )


def _multi_regime_series() -> Bars:
    # Up, down, up, down, up — guarantees multiple entries and exits.
    return _concat(
        syn.trending_series(150, drift=+0.006, seed=1),
        syn.trending_series(120, drift=-0.006, noise=0.003, seed=2),
        syn.trending_series(150, drift=+0.006, seed=3),
        syn.trending_series(120, drift=-0.006, noise=0.003, seed=4),
        syn.trending_series(150, drift=+0.006, seed=5),
    )


def test_no_lookahead_bias():
    bars = _multi_regime_series()
    full = run_backtest("SYN", bars)

    k = 500  # truncation point well inside the series
    trunc = Bars(
        open=bars.open[:k], high=bars.high[:k], low=bars.low[:k],
        close=bars.close[:k], volume=bars.volume[:k], symbol="SYN",
    )
    partial = run_backtest("SYN", trunc)

    # Compare the overlapping equity path, minus a small buffer so the truncated
    # run's forced end-of-data liquidation doesn't pollute the comparison.
    m = len(partial.equity_curve) - 5
    assert m > 50
    np.testing.assert_allclose(full.equity_curve[:m], partial.equity_curve[:m], rtol=1e-9)


def test_profits_in_clean_uptrend():
    bars = syn.trending_series(250, drift=+0.005, seed=7)
    r = run_backtest("UP", bars)
    assert r.total_return > 0
    assert len(r.trades) >= 1


def test_stop_loss_caps_losses_on_a_crash():
    # Rally to trigger an entry, then a sharp sustained drop.
    up = syn.trending_series(180, drift=+0.006, seed=1)
    crash_close = up.close[-1] * np.linspace(1.0, 0.5, 60)
    crash = make_bars(
        crash_close, crash_close * 1.005, crash_close * 0.99, crash_close,
        np.full(60, 5_000_000.0),
    )
    bars = _concat(up, crash)
    r = run_backtest("CRASH", bars)
    assert any(t.exit_reason == "stop" for t in r.trades)
    # No single trade should lose more than ~20% with an ATR stop (gap risk aside).
    worst = min((t.return_pct for t in r.trades), default=0.0)
    assert worst > -0.25


def test_slippage_reduces_return():
    bars = syn.trending_series(250, drift=+0.005, seed=7)
    no_slip = run_backtest("UP", bars, slippage_bps=0.0)
    with_slip = run_backtest("UP", bars, slippage_bps=20.0)
    assert with_slip.total_return <= no_slip.total_return


def test_buy_hold_return_matches_prices():
    bars = syn.trending_series(250, drift=+0.004, seed=2)
    r = run_backtest("UP", bars)
    # Buy-hold is measured from the first decision bar to the last close.
    start = r.n_bars  # equity_curve length == n - start
    expected = bars.close[-1] / bars.close[len(bars) - start] - 1.0
    assert abs(r.buy_hold_return - expected) < 1e-9


def test_trailing_stop_holds_longer_and_removes_target_cap():
    # Mechanics only: a trailing stop holds longer than a fixed target and never
    # exits via "target". Whether it is MORE PROFITABLE is an empirical question
    # for real data, not an invariant (giving back open profit on pullbacks can
    # underperform locking in + re-entering) — see BACKTEST.md.
    bars = syn.trending_series(300, drift=+0.005, seed=7)
    target = run_backtest("UP", bars, exit_mode="target")
    trailing = run_backtest("UP", bars, exit_mode="trailing")

    def avg_hold(r):
        return np.mean([t.bars_held for t in r.trades]) if r.trades else 0.0

    assert avg_hold(trailing) > avg_hold(target)
    assert all(t.exit_reason in ("trail", "signal", "eod") for t in trailing.trades)
    assert not any(t.exit_reason == "target" for t in trailing.trades)


def test_no_trades_leaves_equity_flat():
    # Too few bars to ever trade -> equity stays at start.
    close = np.linspace(100, 101, 40)
    bars = make_bars(close, close * 1.001, close * 0.999, close, np.full(40, 1e6))
    r = run_backtest("FLAT", bars)
    assert r.trades == []
    assert r.end_equity == r.start_equity


def test_load_long_csv(tmp_path):
    p = tmp_path / "data.csv"
    rows = ["date,open,high,low,close,volume,Name"]
    for i in range(130):
        px = 100 + i
        rows.append(f"2020-01-01,{px},{px+1},{px-1},{px},1000000,FOO")
    p.write_text("\n".join(rows))
    data = load_long_csv(str(p))
    assert "FOO" in data
    assert len(data["FOO"]) == 130
