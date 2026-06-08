"""Orchestration: data → decision → (review → place), with risk gating.

:class:`TradingEngine` ties the pieces together:

* ``scan``  — fetch history for each symbol and produce a :class:`Decision`.
  Pure analysis; needs only a data provider (works fully offline in ``--demo``).
* ``run``   — additionally read the account, size orders within the caps, review
  every order, and (only when ``allow_trading`` is set) place them.

The order-sizing/gating logic (:meth:`plan_buy`, :meth:`plan_sell`) is kept pure
so it is unit-tested without any brokerage connection.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .execution import Executor, Position, Quote
from .market_data import Bars, DataProvider
from .strategy import BUY, HOLD, SELL, Decision, StrategyParams, decide, suggested_notional

MIN_NOTIONAL = 1.0


@dataclass
class PlannedOrder:
    """A concrete, ready-to-submit order derived from a decision."""

    symbol: str
    side: str  # buy / sell
    order_type: str  # market / limit
    quantity: float | None = None
    dollar_amount: float | None = None
    limit_price: float | None = None
    time_in_force: str = "gfd"
    market_hours: str = "regular_hours"
    notional: float = 0.0
    ref_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_args(self, account: str, *, include_ref: bool) -> dict[str, Any]:
        args: dict[str, Any] = {
            "account_number": account,
            "symbol": self.symbol,
            "side": self.side,
            "type": self.order_type,
            "time_in_force": self.time_in_force,
            "market_hours": self.market_hours,
        }
        if self.quantity is not None:
            args["quantity"] = f"{self.quantity:g}"
        if self.dollar_amount is not None:
            args["dollar_amount"] = f"{self.dollar_amount:.2f}"
        if self.limit_price is not None:
            args["limit_price"] = f"{self.limit_price:.2f}"
        if include_ref:
            args["ref_id"] = self.ref_id
        return args


@dataclass
class ExecutionRecord:
    """What happened for one symbol on a run."""

    decision: Decision
    status: str  # placed / reviewed / skipped / error
    message: str = ""
    order: PlannedOrder | None = None
    review: dict[str, Any] | None = None
    placement: dict[str, Any] | None = None


class TradingEngine:
    def __init__(
        self,
        config: Config,
        provider: DataProvider,
        executor: Executor | None = None,
        strategy_params: StrategyParams | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.executor = executor
        self.params = strategy_params or config.build_strategy_params()

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def analyze(self, symbol: str, bars: Bars, price: float | None = None) -> Decision:
        return decide(symbol, bars, self.params, price=price)

    def scan(self, symbols: list[str], prices: dict[str, float] | None = None) -> list[Decision]:
        prices = prices or {}
        decisions: list[Decision] = []
        for symbol in symbols:
            sym = symbol.upper()
            try:
                bars = self.provider.get_bars(sym, self.config.data_interval, self.config.lookback_bars)
                decisions.append(self.analyze(sym, bars, price=prices.get(sym)))
            except Exception as exc:  # data hiccup on one symbol shouldn't kill the scan
                decisions.append(_error_decision(sym, f"data error: {exc}"))
        return decisions

    # ------------------------------------------------------------------ #
    # Order planning (pure)
    # ------------------------------------------------------------------ #
    def plan_buy(
        self,
        decision: Decision,
        quote: Quote,
        buying_power: float | None,
        remaining_daily: float,
    ) -> tuple[PlannedOrder | None, str]:
        cfg = self.config
        if cfg.allowed_symbols and decision.symbol not in cfg.allowed_symbols:
            return None, "symbol not in allowlist"

        notional = suggested_notional(decision, cfg.max_order_value_usd, cfg.size_floor_frac)
        notional = min(notional, remaining_daily)
        if buying_power is not None:
            notional = min(notional, buying_power)
        notional = round(notional, 2)
        if notional < MIN_NOTIONAL:
            reason = "daily cap reached" if remaining_daily < MIN_NOTIONAL else "insufficient buying power"
            return None, f"size below ${MIN_NOTIONAL:.0f} ({reason})"

        if cfg.order_type == "market":
            order = PlannedOrder(
                symbol=decision.symbol,
                side="buy",
                order_type="market",
                dollar_amount=notional,
                time_in_force=cfg.time_in_force,
                market_hours=cfg.market_hours,
                notional=notional,
            )
            return order, "ok"

        ask = quote.ask or quote.price
        limit_price = round(ask * (1.0 + cfg.slippage_bps / 1e4), 2)
        qty = int(notional // limit_price)
        if qty < 1:
            return None, f"size ${notional:.2f} < 1 share @ ${limit_price:.2f}"
        order = PlannedOrder(
            symbol=decision.symbol,
            side="buy",
            order_type="limit",
            quantity=qty,
            limit_price=limit_price,
            time_in_force=cfg.time_in_force,
            market_hours=cfg.market_hours,
            notional=round(qty * limit_price, 2),
        )
        return order, "ok"

    def plan_sell(
        self, decision: Decision, position: Position | None, quote: Quote
    ) -> tuple[PlannedOrder | None, str]:
        cfg = self.config
        if position is None or position.quantity <= 0:
            return None, "SELL signal but no long position (shorting disabled)"

        if cfg.order_type == "market":
            order = PlannedOrder(
                symbol=decision.symbol,
                side="sell",
                order_type="market",
                quantity=position.quantity,  # fractional ok for market in regular hours
                time_in_force=cfg.time_in_force,
                market_hours=cfg.market_hours,
                notional=round(position.quantity * quote.price, 2),
            )
            return order, "ok"

        bid = quote.bid or quote.price
        limit_price = round(bid * (1.0 - cfg.slippage_bps / 1e4), 2)
        qty = int(position.quantity)  # whole shares for a limit exit
        if qty < 1:
            return None, "only a fractional position; use ORDER_TYPE=market to exit"
        order = PlannedOrder(
            symbol=decision.symbol,
            side="sell",
            order_type="limit",
            quantity=qty,
            limit_price=limit_price,
            time_in_force=cfg.time_in_force,
            market_hours=cfg.market_hours,
            notional=round(qty * limit_price, 2),
        )
        return order, "ok"

    # ------------------------------------------------------------------ #
    # Live run
    # ------------------------------------------------------------------ #
    async def run(self, symbols: list[str]) -> list[ExecutionRecord]:
        if self.executor is None:
            raise RuntimeError("run() needs an executor; use scan() for analysis-only")
        account = self.config.account_number
        if not account:
            raise RuntimeError("ROBINHOOD_ACCOUNT_NUMBER is required to trade")

        syms = [s.upper() for s in symbols]
        quotes = await self.executor.get_quotes(syms)
        positions = await self.executor.get_positions(account)
        buying_power = await self.executor.get_buying_power(account)

        prices = {s: q.price for s, q in quotes.items()}
        decisions = self.scan(syms, prices=prices)

        records: list[ExecutionRecord] = []
        remaining_daily = self.config.max_daily_spend_usd
        for d in decisions:
            quote = quotes.get(d.symbol, Quote(d.symbol, d.price))
            if d.action == HOLD:
                records.append(ExecutionRecord(d, "skipped", d.rationale))
                continue

            if d.action == BUY:
                if d.symbol in positions and positions[d.symbol].quantity > 0:
                    records.append(ExecutionRecord(d, "skipped", "already holding (no pyramiding)"))
                    continue
                if buying_power is None and self.config.allow_trading:
                    records.append(ExecutionRecord(d, "skipped", "buying power unknown; refusing to place"))
                    continue
                order, msg = self.plan_buy(d, quote, buying_power, remaining_daily)
            else:  # SELL
                order, msg = self.plan_sell(d, positions.get(d.symbol), quote)

            if order is None:
                records.append(ExecutionRecord(d, "skipped", msg))
                continue

            records.append(await self._review_and_place(account, d, order))
            if order.side == "buy" and records[-1].status in ("placed", "reviewed"):
                remaining_daily -= order.notional
                if buying_power is not None:
                    buying_power -= order.notional

        return records

    async def _review_and_place(
        self, account: str, decision: Decision, order: PlannedOrder
    ) -> ExecutionRecord:
        assert self.executor is not None
        try:
            review = await self.executor.review_order(**order.to_args(account, include_ref=False))
        except Exception as exc:
            return ExecutionRecord(decision, "error", f"review failed: {exc}", order=order)

        if not self.config.allow_trading:
            return ExecutionRecord(
                decision, "reviewed", "dry-run: reviewed, not placed", order=order, review=review
            )
        try:
            placement = await self.executor.place_order(**order.to_args(account, include_ref=True))
        except Exception as exc:
            return ExecutionRecord(
                decision, "error", f"place failed: {exc}", order=order, review=review
            )
        return ExecutionRecord(decision, "placed", "order placed", order=order, review=review, placement=placement)


def _error_decision(symbol: str, message: str) -> Decision:
    return Decision(
        symbol=symbol,
        action=HOLD,
        direction=0,
        confidence=0.0,
        threshold=0.65,
        price=float("nan"),
        weighted_score=0.0,
        agreement=0.0,
        liquidity_gate=0.0,
        atr=float("nan"),
        stop_price=None,
        target_price=None,
        risk_reward=None,
        signals=[],
        contributions={},
        rationale=message,
    )
