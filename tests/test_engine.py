"""Tests for the engine's planning, gating, and execution flow (no network)."""

from __future__ import annotations

import asyncio

import pytest

from robinhood_agent import synthetic as syn
from robinhood_agent.config import Config
from robinhood_agent.engine import TradingEngine
from robinhood_agent.execution import Executor, Position, Quote
from robinhood_agent.strategy import BUY, decide


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeProvider:
    def __init__(self):
        self._bars = {
            "UP": syn.trending_series(drift=+0.005),
            "DN": syn.trending_series(drift=-0.006, noise=0.003, seed=1),
            "CHOP": syn.choppy_series(),
        }

    def get_bars(self, symbol, interval="1d", lookback=300):
        return self._bars[symbol.upper()]


class FakeExecutor:
    """In-memory Executor that records reviews and placements."""

    def __init__(self, buying_power=10_000.0, positions=None, quotes=None):
        self._bp = buying_power
        self._positions = positions or {}
        self._quotes = quotes or {}
        self.reviewed: list[dict] = []
        self.placed: list[dict] = []

    async def get_buying_power(self, account):
        return self._bp

    async def get_positions(self, account):
        return dict(self._positions)

    async def get_quotes(self, symbols):
        return {s: self._quotes[s] for s in symbols if s in self._quotes}

    async def review_order(self, **order):
        self.reviewed.append(order)
        return {"ok": True, "estimated_cost": "100.00"}

    async def place_order(self, **order):
        self.placed.append(order)
        return {"id": "order-123", "state": "queued"}


def _config(**overrides):
    base = dict(
        anthropic_api_key="x",
        robinhood_mcp_token="y",
        account_number="ACC-1",
        max_order_value_usd=500.0,
        max_daily_spend_usd=2000.0,
    )
    base.update(overrides)
    return Config(**base)


def _quotes_for(provider):
    q = {}
    for sym, bars in provider._bars.items():
        px = bars.last_price
        q[sym] = Quote(sym, price=px, bid=px * 0.999, ask=px * 1.001)
    return q


# --------------------------------------------------------------------------- #
# Protocol / planning (pure)
# --------------------------------------------------------------------------- #
def test_fake_executor_satisfies_protocol():
    assert isinstance(FakeExecutor(), Executor)


def test_plan_buy_respects_allowlist():
    eng = TradingEngine(_config(allowed_symbols=["AAPL"]), FakeProvider())
    d = decide("UP", FakeProvider()._bars["UP"])
    order, msg = eng.plan_buy(d, Quote("UP", 100, ask=100.1), 10_000, 2000)
    assert order is None and "allowlist" in msg


def test_plan_buy_caps_by_daily_budget_and_buying_power():
    eng = TradingEngine(_config(), FakeProvider())
    d = decide("UP", FakeProvider()._bars["UP"])
    # Tiny remaining budget -> rejected.
    order, msg = eng.plan_buy(d, Quote("UP", 100, ask=100.1), 10_000, 0.5)
    assert order is None and "daily cap" in msg
    # Tiny buying power -> rejected.
    order, msg = eng.plan_buy(d, Quote("UP", 100, ask=100.1), 0.5, 2000)
    assert order is None and "buying power" in msg


def test_plan_buy_sizes_within_cap():
    eng = TradingEngine(_config(max_order_value_usd=500.0, size_floor_frac=0.5), FakeProvider())
    d = decide("UP", FakeProvider()._bars["UP"])
    order, msg = eng.plan_buy(d, Quote("UP", 100, ask=100.0), 10_000, 2000)
    assert order is not None and msg == "ok"
    assert order.side == "buy" and order.order_type == "limit"
    assert order.quantity >= 1
    assert order.notional <= 500.0 + 1e-6
    assert order.limit_price >= 100.0  # marketable: at/above ask


def test_plan_sell_requires_position():
    eng = TradingEngine(_config(), FakeProvider())
    d = decide("DN", FakeProvider()._bars["DN"])
    none_order, msg = eng.plan_sell(d, None, Quote("DN", 50, bid=49.9))
    assert none_order is None and "no long position" in msg
    order, msg = eng.plan_sell(d, Position("DN", 10), Quote("DN", 50, bid=49.9))
    assert order is not None and order.side == "sell" and order.quantity == 10


# --------------------------------------------------------------------------- #
# End-to-end run
# --------------------------------------------------------------------------- #
def test_dry_run_reviews_but_never_places():
    provider = FakeProvider()
    execu = FakeExecutor(quotes=_quotes_for(provider))
    eng = TradingEngine(_config(allow_trading=False), provider, execu)
    records = asyncio.run(eng.run(["UP", "CHOP"]))
    by_sym = {r.decision.symbol: r for r in records}
    assert by_sym["UP"].status == "reviewed"  # BUY reviewed, not placed
    assert by_sym["CHOP"].status == "skipped"
    assert len(execu.reviewed) == 1
    assert execu.placed == []


def test_live_run_places_buy():
    provider = FakeProvider()
    execu = FakeExecutor(quotes=_quotes_for(provider))
    eng = TradingEngine(_config(allow_trading=True), provider, execu)
    records = asyncio.run(eng.run(["UP"]))
    assert records[0].status == "placed"
    assert len(execu.placed) == 1
    assert execu.placed[0]["side"] == "buy"
    assert "ref_id" in execu.placed[0]  # idempotency key on placement


def test_live_run_sells_when_holding():
    provider = FakeProvider()
    execu = FakeExecutor(
        quotes=_quotes_for(provider),
        positions={"DN": Position("DN", 10, average_cost=80.0)},
    )
    eng = TradingEngine(_config(allow_trading=True), provider, execu)
    records = asyncio.run(eng.run(["DN"]))
    assert records[0].decision.action == "SELL"
    assert records[0].status == "placed"
    assert execu.placed[0]["side"] == "sell"


def test_buy_skipped_when_already_holding():
    provider = FakeProvider()
    execu = FakeExecutor(
        quotes=_quotes_for(provider),
        positions={"UP": Position("UP", 5, average_cost=100.0)},
    )
    eng = TradingEngine(_config(allow_trading=True), provider, execu)
    records = asyncio.run(eng.run(["UP"]))
    assert records[0].status == "skipped"
    assert "already holding" in records[0].message
    assert execu.placed == []


def test_daily_cap_limits_total_buys():
    provider = FakeProvider()
    # Two buyable names but a daily cap that only affords one order.
    provider._bars["UP2"] = syn.trending_series(drift=+0.005, seed=2)
    execu = FakeExecutor(quotes=_quotes_for(provider))
    execu._quotes["UP2"] = Quote(
        "UP2", price=provider._bars["UP2"].last_price,
        bid=provider._bars["UP2"].last_price, ask=provider._bars["UP2"].last_price,
    )
    # Market orders use the dollar notional directly, so the daily cap is the
    # exact binding constraint (no per-share rounding in between).
    eng = TradingEngine(
        _config(
            allow_trading=True,
            order_type="market",
            max_order_value_usd=500.0,
            max_daily_spend_usd=400.0,
        ),
        provider,
        execu,
    )
    records = asyncio.run(eng.run(["UP", "UP2"]))
    placed = [r for r in records if r.status == "placed"]
    skipped_cap = [r for r in records if r.status == "skipped" and "cap" in r.message]
    assert len(placed) == 1
    assert len(skipped_cap) == 1
    assert sum(r.order.notional for r in placed) <= 400.0 + 1e-6


def test_run_requires_account():
    eng = TradingEngine(_config(account_number=""), FakeProvider(), FakeExecutor())
    with pytest.raises(RuntimeError, match="ACCOUNT"):
        asyncio.run(eng.run(["UP"]))


def test_scan_handles_data_errors_gracefully():
    class BadProvider:
        def get_bars(self, *a, **k):
            raise ValueError("boom")

    eng = TradingEngine(_config(), BadProvider())
    decisions = eng.scan(["AAA"])
    assert decisions[0].action == "HOLD"
    assert "data error" in decisions[0].rationale
