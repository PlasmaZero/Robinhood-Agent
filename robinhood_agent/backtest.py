"""Event-driven, bias-free backtester for the TA engine.

The whole point of this module is to measure the strategy *honestly*:

* **No lookahead.** The decision for acting on bar ``t+1`` is computed using only
  data through the close of bar ``t``. Fills happen at bar ``t+1``'s *open* with
  slippage — you can never trade on information you wouldn't have had.
* **Realistic exits.** Stops/targets are checked intrabar against each bar's
  high/low, with gap handling at the open (a gap straight through your stop fills
  at the open, not the stop).
* **Costs.** Slippage (and optional commission) are charged on every fill.

It reports the metrics that matter — return, win rate, profit factor, expectancy,
Sharpe, max drawdown, exposure — and always against a **buy-and-hold benchmark**
over the same window, so "profitable" is judged relative to just owning the stock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .market_data import Bars
from .strategy import BUY, SELL, StrategyParams, decide, suggested_notional

# A rolling window passed to decide(): comfortably exceeds every factor's
# lookback (sr/fib 120, slow_ma 50) plus warm-up, so it matches full-history
# decisions while keeping each step O(window) instead of O(t).
DECISION_WINDOW = 220


@dataclass
class Trade:
    symbol: str
    entry_index: int
    entry_price: float
    exit_index: int
    exit_price: float
    quantity: float
    exit_reason: str  # stop / target / signal / eod
    pnl: float
    return_pct: float

    @property
    def bars_held(self) -> int:
        return self.exit_index - self.entry_index


@dataclass
class BacktestResult:
    symbol: str
    trades: list[Trade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    start_equity: float = 0.0
    end_equity: float = 0.0
    buy_hold_return: float = 0.0
    buy_hold_max_dd: float = 0.0
    buy_hold_sharpe: float = 0.0
    n_bars: int = 0
    bars_in_market: int = 0

    @property
    def total_return(self) -> float:
        return self.end_equity / self.start_equity - 1.0 if self.start_equity else 0.0

    @property
    def exposure(self) -> float:
        return self.bars_in_market / self.n_bars if self.n_bars else 0.0

    @property
    def max_drawdown(self) -> float:
        return _max_drawdown(self.equity_curve)

    @property
    def sharpe(self) -> float:
        return _sharpe(self.equity_curve)


def run_backtest(
    symbol: str,
    bars: Bars,
    params: StrategyParams | None = None,
    *,
    start_equity: float = 100_000.0,
    slippage_bps: float = 5.0,
    commission_per_trade: float = 0.0,
    size_floor_frac: float = 0.5,
    max_position_frac: float = 0.95,
    exit_mode: str = "target",  # "target" (fixed 2:1) or "trailing" (let winners run)
    trail_atr_mult: float = 3.0,
) -> BacktestResult:
    sp = params or StrategyParams()
    o, h, l, c = bars.open, bars.high, bars.low, bars.close
    n = len(bars)
    start = max(sp.signal_params.min_bars, DECISION_WINDOW // 4)
    if n <= start + 2:
        return BacktestResult(symbol=symbol, start_equity=start_equity, end_equity=start_equity)

    slip = slippage_bps / 1e4
    cash = start_equity
    qty = 0.0
    entry_price = 0.0
    entry_index = 0
    stop = target = 0.0
    atr_entry = 0.0
    highest = 0.0  # high-water mark since entry, for the trailing stop
    pending: tuple[str, object] | None = None  # ('buy'|'sell', Decision)

    trades: list[Trade] = []
    equity_curve = np.empty(n - start)
    bars_in_market = 0

    for t in range(start, n):
        # 1) Execute any order scheduled on the previous bar, at THIS bar's open.
        if pending is not None:
            side, dec = pending
            pending = None
            if side == "buy" and qty == 0:
                fill = o[t] * (1.0 + slip)
                frac = min(suggested_notional(dec, 1.0, size_floor_frac), max_position_frac)  # type: ignore[arg-type]
                shares = math.floor((cash * frac) / fill)
                if shares >= 1:
                    qty = float(shares)
                    entry_price = fill
                    entry_index = t
                    cash -= qty * fill + commission_per_trade
                    stop = dec.stop_price if dec.stop_price else fill * 0.95  # type: ignore[attr-defined]
                    target = dec.target_price if dec.target_price else fill * 1.10  # type: ignore[attr-defined]
                    atr_entry = dec.atr if (dec.atr and dec.atr > 0) else fill * 0.02  # type: ignore[attr-defined]
                    highest = fill
            elif side == "sell" and qty > 0:
                fill = o[t] * (1.0 - slip)
                _close(trades, symbol, entry_index, entry_price, t, fill, qty, "signal", commission_per_trade)
                cash += qty * fill - commission_per_trade
                qty = 0.0

        # 2) While in a position, check exits intrabar (gap-aware).
        if qty > 0:
            exit_price = None
            reason = ""
            if exit_mode == "trailing":
                # Ratchet a stop up under the high-water mark; no profit cap, so
                # winners can run the length of the trend.
                highest = max(highest, h[t])
                eff_stop = max(stop, highest - trail_atr_mult * atr_entry)
                if o[t] <= eff_stop:  # gapped through the trail
                    exit_price, reason = o[t] * (1.0 - slip), "trail"
                elif l[t] <= eff_stop:
                    exit_price, reason = eff_stop * (1.0 - slip), "trail"
            else:
                if o[t] <= stop:  # gapped down through the stop
                    exit_price, reason = o[t] * (1.0 - slip), "stop"
                elif o[t] >= target:  # gapped up through the target
                    exit_price, reason = o[t] * (1.0 - slip), "target"
                elif l[t] <= stop:  # stop checked before target (pessimistic)
                    exit_price, reason = stop * (1.0 - slip), "stop"
                elif h[t] >= target:
                    exit_price, reason = target * (1.0 - slip), "target"
            if exit_price is not None:
                _close(trades, symbol, entry_index, entry_price, t, exit_price, qty, reason, commission_per_trade)
                cash += qty * exit_price - commission_per_trade
                qty = 0.0

        # 3) Decide using ONLY data through the close of bar t.
        lo = max(0, t - DECISION_WINDOW + 1)
        window = Bars(
            open=o[lo : t + 1], high=h[lo : t + 1], low=l[lo : t + 1],
            close=c[lo : t + 1], volume=bars.volume[lo : t + 1], symbol=symbol,
        )
        dec = decide(symbol, window, sp, price=float(c[t]))
        if qty == 0 and dec.action == BUY:
            pending = ("buy", dec)
        elif qty > 0 and dec.action == SELL:
            pending = ("sell", dec)

        if qty > 0:
            bars_in_market += 1
        equity_curve[t - start] = cash + qty * c[t]

    # Flatten any open position at the final close.
    if qty > 0:
        fill = c[-1] * (1.0 - slip)
        _close(trades, symbol, entry_index, entry_price, n - 1, fill, qty, "eod", commission_per_trade)
        cash += qty * fill - commission_per_trade
        qty = 0.0
        equity_curve[-1] = cash

    bh_curve = start_equity * c[start:] / c[start]  # own the stock over the same window
    return BacktestResult(
        symbol=symbol,
        trades=trades,
        equity_curve=equity_curve,
        start_equity=start_equity,
        end_equity=float(equity_curve[-1]),
        buy_hold_return=float(c[-1] / c[start] - 1.0),
        buy_hold_max_dd=_max_drawdown(bh_curve),
        buy_hold_sharpe=_sharpe(bh_curve),
        n_bars=n - start,
        bars_in_market=bars_in_market,
    )


def _close(trades, symbol, ei, ep, xi, xp, qty, reason, commission):
    pnl = qty * (xp - ep) - 2 * commission
    trades.append(
        Trade(
            symbol=symbol,
            entry_index=ei,
            entry_price=ep,
            exit_index=xi,
            exit_price=xp,
            quantity=qty,
            exit_reason=reason,
            pnl=pnl,
            return_pct=xp / ep - 1.0,
        )
    )


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak))


def _sharpe(equity: np.ndarray, periods_per_year: int = 252) -> float:
    if equity.size < 3:
        return 0.0
    rets = np.diff(equity) / equity[:-1]
    sd = float(np.std(rets))
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(rets) / sd * math.sqrt(periods_per_year))


# --------------------------------------------------------------------------- #
# Aggregation across a basket
# --------------------------------------------------------------------------- #
def pooled_summary(results: list[BacktestResult]) -> dict:
    """Honest aggregate stats across many symbols."""
    all_trades = [tr for r in results for tr in r.trades]
    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)

    strat_returns = [r.total_return for r in results]
    bh_returns = [r.buy_hold_return for r in results]
    beat = sum(1 for r in results if r.total_return > r.buy_hold_return)

    def avg(x):
        return float(np.mean(x)) if x else 0.0

    def median(x):
        return float(np.median(x)) if x else 0.0

    return {
        "symbols": len(results),
        "n_trades": len(all_trades),
        "win_rate": len(wins) / len(all_trades) if all_trades else 0.0,
        "avg_win_pct": avg([t.return_pct for t in wins]),
        "avg_loss_pct": avg([t.return_pct for t in losses]),
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 1e-9 else float("inf"),
        "expectancy_pct": avg([t.return_pct for t in all_trades]),
        "avg_bars_held": avg([t.bars_held for t in all_trades]),
        "exit_breakdown": _exit_breakdown(all_trades),
        "strat_mean_return": avg(strat_returns),
        "strat_median_return": median(strat_returns),
        "buyhold_mean_return": avg(bh_returns),
        "buyhold_median_return": median(bh_returns),
        "pct_beating_buyhold": beat / len(results) if results else 0.0,
        "strat_mean_max_dd": avg([r.max_drawdown for r in results]),
        "buyhold_mean_max_dd": avg([r.buy_hold_max_dd for r in results]),
        "strat_mean_sharpe": avg([r.sharpe for r in results]),
        "buyhold_mean_sharpe": avg([r.buy_hold_sharpe for r in results]),
        "mean_exposure": avg([r.exposure for r in results]),
    }


def _exit_breakdown(trades: list[Trade]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in trades:
        out[t.exit_reason] = out.get(t.exit_reason, 0) + 1
    return out


# --------------------------------------------------------------------------- #
# Loading the "long" combined CSV (date,open,high,low,close,volume,Name)
# --------------------------------------------------------------------------- #
def load_long_csv(path: str, symbols: list[str] | None = None) -> dict[str, Bars]:
    import csv as _csv
    from collections import defaultdict

    wanted = set(s.upper() for s in symbols) if symbols else None
    cols = defaultdict(lambda: {"open": [], "high": [], "low": [], "close": [], "volume": []})
    with open(path, newline="") as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            sym = row["Name"].upper()
            if wanted is not None and sym not in wanted:
                continue
            try:
                vals = [float(row[k]) for k in ("open", "high", "low", "close", "volume")]
            except (ValueError, KeyError):
                continue  # skip rows with gaps (some symbols have missing days)
            d = cols[sym]
            d["open"].append(vals[0]); d["high"].append(vals[1]); d["low"].append(vals[2])
            d["close"].append(vals[3]); d["volume"].append(vals[4])

    out: dict[str, Bars] = {}
    for sym, d in cols.items():
        if len(d["close"]) < 120:
            continue
        out[sym] = Bars(
            open=np.array(d["open"]), high=np.array(d["high"]), low=np.array(d["low"]),
            close=np.array(d["close"]), volume=np.array(d["volume"]), symbol=sym, interval="1d",
        )
    return out
