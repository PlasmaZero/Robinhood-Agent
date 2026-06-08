"""The five trading factors, each scored independently.

Every factor takes a :class:`~robinhood_agent.market_data.Bars` series and emits
a :class:`SignalResult` with:

* ``score``      — direction in ``[-1, 1]`` (negative = bearish, positive = bullish)
* ``confidence`` — how strongly we believe this *one* factor, in ``[0, 1]``

The factors are deliberately decoupled so each can be reasoned about and tested
on its own. :mod:`robinhood_agent.strategy` blends them into a single decision.

Factors
-------
* ``momentum``           — RSI, MACD histogram, rate-of-change, MA stack
* ``support_resistance`` — position within the swing range + breakouts
* ``fibonacci``          — retracement of the dominant swing (buy-the-dip logic)
* ``orderflow``          — MFI, OBV slope, A/D line, up/down volume balance
* ``liquidity``          — tradeable depth (also used downstream as a gate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import indicators as ind
from .indicators import EPS
from .market_data import Bars


@dataclass(frozen=True)
class SignalResult:
    """One factor's read on a symbol."""

    name: str
    score: float  # [-1, 1]
    confidence: float  # [0, 1]
    rationale: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def evidence(self) -> float:
        """Signed, confidence-weighted contribution in [-1, 1]."""
        return self.score * self.confidence


@dataclass(frozen=True)
class SignalParams:
    """Tunable knobs shared by the factors (sensible defaults)."""

    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    roc_period: int = 10
    fast_ma: int = 20
    slow_ma: int = 50
    atr_period: int = 14
    mfi_period: int = 14
    swing_left: int = 3
    swing_right: int = 3
    sr_lookback: int = 120
    fib_lookback: int = 120
    flow_window: int = 20
    liquidity_window: int = 20
    # Liquidity gate calibration (average daily *dollar* volume).
    liq_min_dollar_vol: float = 1_000_000.0  # below this -> gate ~0 (too thin)
    liq_good_dollar_vol: float = 25_000_000.0  # at/above this -> gate ~1
    min_bars: int = 60


def _neutral(name: str, reason: str) -> SignalResult:
    return SignalResult(name=name, score=0.0, confidence=0.0, rationale=reason)


def _combine(components: list[float], weights: list[float] | None = None) -> tuple[float, float]:
    """Blend sub-component scores into a (score, confidence) pair.

    * score      = weighted mean of components, clipped to [-1, 1]
    * confidence = weighted magnitude scaled by how much the components agree
                   on direction (full agreement -> keep magnitude; split ->
                   discount it). Always in [0, 1].
    """
    a = np.nan_to_num(np.asarray(components, dtype=float), nan=0.0)
    if a.size == 0:
        return 0.0, 0.0
    w = np.ones(a.size) if weights is None else np.asarray(weights, dtype=float)[: a.size]
    w = w / (w.sum() + EPS)
    score = float(np.clip(float((a * w).sum()), -1.0, 1.0))
    dom = np.sign(score)
    agreement = float(w[np.sign(a) == dom].sum()) if dom != 0 else 0.0
    magnitude = float((np.abs(a) * w).sum())
    confidence = float(np.clip(magnitude * (0.45 + 0.55 * agreement), 0.0, 1.0))
    return score, confidence


# --------------------------------------------------------------------------- #
# 1. Momentum
# --------------------------------------------------------------------------- #
def momentum_signal(bars: Bars, p: SignalParams, price: float | None = None) -> SignalResult:
    name = "momentum"
    if len(bars) < max(p.slow_ma, p.macd_slow + p.macd_signal, p.rsi_period + 2):
        return _neutral(name, "insufficient history")

    close = bars.close
    px = float(price) if price is not None else bars.last_price

    rsi_last = ind.last_finite(ind.rsi(close, p.rsi_period))
    rsi_score = float(np.clip((rsi_last - 50.0) / 30.0, -1.0, 1.0))

    _, _, hist = ind.macd(close, p.macd_fast, p.macd_slow, p.macd_signal)
    hist_finite = hist[~np.isnan(hist)]
    if hist_finite.size:
        hist_std = float(np.std(hist_finite[-50:])) + EPS
        macd_score = float(np.tanh(hist_finite[-1] / hist_std))
    else:
        macd_score = 0.0

    roc_last = ind.last_finite(ind.roc(close, p.roc_period))
    roc_score = float(np.tanh(roc_last / 5.0)) if np.isfinite(roc_last) else 0.0

    fast = ind.last_finite(ind.sma(close, p.fast_ma))
    slow = ind.last_finite(ind.sma(close, p.slow_ma))
    fast_slope = ind.linreg_slope(close, p.fast_ma)
    trend = 0.0
    trend += 0.5 if px > fast else -0.5
    trend += 0.5 if fast > slow else -0.5
    if np.isfinite(fast_slope):
        trend = float(np.clip(trend + np.sign(fast_slope) * 0.0, -1.0, 1.0))
    trend_score = float(np.clip(trend, -1.0, 1.0))

    score, confidence = _combine(
        [rsi_score, macd_score, roc_score, trend_score],
        weights=[1.0, 1.0, 0.8, 1.2],
    )
    rationale = (
        f"RSI={rsi_last:.0f}, MACD-hist {'+' if macd_score >= 0 else '-'}, "
        f"ROC={roc_last:.1f}%, MA-stack {'bullish' if trend_score > 0 else 'bearish'}"
    )
    return SignalResult(
        name,
        score,
        confidence,
        rationale,
        detail={"rsi": rsi_last, "roc": roc_last, "sma_fast": fast, "sma_slow": slow},
    )


# --------------------------------------------------------------------------- #
# 2. Support & Resistance
# --------------------------------------------------------------------------- #
def support_resistance_signal(bars: Bars, p: SignalParams, price: float | None = None) -> SignalResult:
    name = "support_resistance"
    if len(bars) < p.min_bars:
        return _neutral(name, "insufficient history")

    lb = min(p.sr_lookback, len(bars))
    high = bars.high[-lb:]
    low = bars.low[-lb:]
    px = float(price) if price is not None else bars.last_price
    atr_last = ind.last_finite(ind.atr(bars.high, bars.low, bars.close, p.atr_period))
    if not np.isfinite(atr_last) or atr_last <= EPS:
        return _neutral(name, "no volatility estimate")

    # The established range is the prior extreme EXCLUDING the most recent push,
    # so a fresh move beyond it reads as a breakout/breakdown. This is robust to
    # sparse pivots: a smooth trend has few/no fractal pivots, but its range
    # extremes are always well defined.
    exclude = max(p.swing_right + 1, 3)
    if lb > exclude + 2:
        prior_high = float(high[:-exclude].max())
        prior_low = float(low[:-exclude].min())
    else:
        prior_high, prior_low = float(high.max()), float(low.min())

    avg_vol = float(np.mean(bars.volume[-p.flow_window :]))
    vol_ratio = float(bars.volume[-1] / (avg_vol + EPS))
    vol_confirmed = vol_ratio > 1.2
    buffer = 0.25 * atr_last

    nearest_res: float | None
    nearest_sup: float | None

    def _touches(level: float, series: np.ndarray) -> int:
        return int(np.sum(np.abs(series - level) <= 0.5 * atr_last))

    if px > prior_high + buffer:
        # Breakout to new highs: bullish continuation. Old ceiling -> new floor.
        ext = (px - prior_high) / atr_last
        score = float(np.clip(0.5 + 0.5 * min(ext, 1.0), 0.5, 1.0))
        nearest_sup, nearest_res = prior_high, None
        confidence = 0.55 + 0.25 * min(ext, 1.0)
        if vol_confirmed:
            confidence = max(confidence, 0.7 + 0.2 * min(vol_ratio - 1.0, 1.0))
        structure = f"breakout +{ext:.1f} ATR over {prior_high:.2f}"
    elif px < prior_low - buffer:
        # Breakdown to new lows: bearish continuation. Old floor -> new ceiling.
        ext = (prior_low - px) / atr_last
        score = float(np.clip(-(0.5 + 0.5 * min(ext, 1.0)), -1.0, -0.5))
        nearest_sup, nearest_res = None, prior_low
        confidence = 0.55 + 0.25 * min(ext, 1.0)
        if vol_confirmed:
            confidence = max(confidence, 0.7 + 0.2 * min(vol_ratio - 1.0, 1.0))
        structure = f"breakdown -{ext:.1f} ATR under {prior_low:.2f}"
    else:
        # Inside the established range: fade the extremes (mean reversion), but at
        # lower conviction than a breakout — fading is the lower-edge play.
        nearest_sup, nearest_res = prior_low, prior_high
        rng = prior_high - prior_low
        if rng <= EPS:
            return _neutral(name, "degenerate range")
        pos = float(np.clip((px - prior_low) / rng, 0.0, 1.0))
        score = float(np.clip((1.0 - 2.0 * pos) * 0.8, -1.0, 1.0))
        dist = min(px - prior_low, prior_high - px)
        proximity = float(np.exp(-max(dist, 0.0) / atr_last))
        near_level, near_series = (
            (prior_low, low) if (px - prior_low) <= (prior_high - px) else (prior_high, high)
        )
        touch_strength = float(np.clip(_touches(near_level, near_series) / 4.0, 0.0, 1.0))
        confidence = 0.30 * abs(score) + 0.45 * proximity * touch_strength
        structure = f"in range, pos={pos:.0%}"

    confidence = float(np.clip(confidence, 0.0, 1.0))
    breakout = structure.split()[0] if ("breakout" in structure or "breakdown" in structure) else ""
    sup_txt = f"{nearest_sup:.2f}" if nearest_sup is not None else "—"
    res_txt = f"{nearest_res:.2f}" if nearest_res is not None else "open"
    rationale = f"support={sup_txt}, resistance={res_txt}, {structure}"
    return SignalResult(
        name,
        score,
        confidence,
        rationale,
        detail={
            "support": nearest_sup,
            "resistance": nearest_res,
            "structure": structure,
            "breakout": breakout,
        },
    )


# --------------------------------------------------------------------------- #
# 3. Fibonacci retracement
# --------------------------------------------------------------------------- #
def fibonacci_signal(bars: Bars, p: SignalParams, price: float | None = None) -> SignalResult:
    name = "fibonacci"
    if len(bars) < p.min_bars:
        return _neutral(name, "insufficient history")

    lb = min(p.fib_lookback, len(bars))
    high = bars.high[-lb:]
    low = bars.low[-lb:]
    px = float(price) if price is not None else bars.last_price
    atr_last = ind.last_finite(ind.atr(bars.high, bars.low, bars.close, p.atr_period))
    if not np.isfinite(atr_last) or atr_last <= EPS:
        return _neutral(name, "no volatility estimate")

    hi_i = int(np.argmax(high))
    lo_i = int(np.argmin(low))
    swing_high = float(high[hi_i])
    swing_low = float(low[lo_i])
    diff = swing_high - swing_low
    if diff <= EPS:
        return _neutral(name, "degenerate swing")

    uptrend = hi_i > lo_i  # the high printed after the low -> impulse up
    ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    if uptrend:
        levels = {r: swing_high - diff * r for r in ratios}
        retr = (swing_high - px) / diff  # 0 at the high, 1 back at the low
    else:
        levels = {r: swing_low + diff * r for r in ratios}
        retr = (px - swing_low) / diff  # 0 at the low, 1 back at the high

    nearest_ratio = min(ratios, key=lambda r: abs(px - levels[r]))
    nearest_level = levels[nearest_ratio]
    proximity = float(np.exp(-abs(px - nearest_level) / atr_last))

    # Buy-the-dip zones (mirror for downtrends). Magnitude tags how "prime" the
    # retracement is; 50%/61.8% are the classic high-probability bounce zones.
    if retr < 0.236:
        mag = 0.3  # extended near the extreme, little edge
    elif retr <= 0.618:
        mag = 0.9 if nearest_ratio in (0.5, 0.618) else 0.7
    elif retr <= 0.786:
        mag = 0.45
    else:
        mag = -0.7  # retracement too deep: the swing is likely invalidated
    direction = 1.0 if uptrend else -1.0
    score = float(np.clip(direction * mag, -1.0, 1.0))

    trend_clarity = float(np.clip(diff / (atr_last * 8.0), 0.0, 1.0))
    confidence = float(
        np.clip((0.55 * proximity + 0.45 * trend_clarity) * (0.55 + 0.45 * abs(score)), 0.0, 1.0)
    )
    rationale = (
        f"{'up' if uptrend else 'down'}-swing {swing_low:.2f}->{swing_high:.2f}, "
        f"retraced {retr:.0%} (near {nearest_ratio:.3f} @ {nearest_level:.2f})"
    )
    return SignalResult(
        name,
        score,
        confidence,
        rationale,
        detail={
            "swing_low": swing_low,
            "swing_high": swing_high,
            "retracement": retr,
            "nearest_ratio": nearest_ratio,
            "nearest_level": nearest_level,
            "uptrend": uptrend,
        },
    )


# --------------------------------------------------------------------------- #
# 4. Order flow (proxied from OHLCV)
# --------------------------------------------------------------------------- #
def orderflow_signal(bars: Bars, p: SignalParams, price: float | None = None) -> SignalResult:
    name = "orderflow"
    if len(bars) < max(p.mfi_period + 2, p.flow_window + 1):
        return _neutral(name, "insufficient history")

    mfi_last = ind.last_finite(ind.mfi(bars.high, bars.low, bars.close, bars.volume, p.mfi_period))
    mfi_score = float(np.clip((mfi_last - 50.0) / 30.0, -1.0, 1.0)) if np.isfinite(mfi_last) else 0.0

    obv = ind.obv(bars.close, bars.volume)
    obv_score = float(np.tanh(ind.linreg_slope(obv, p.flow_window) * p.flow_window))

    ad = ind.chaikin_ad_line(bars.high, bars.low, bars.close, bars.volume)
    ad_score = float(np.tanh(ind.linreg_slope(ad, p.flow_window) * p.flow_window))

    # Up/down volume balance over the window.
    w = p.flow_window
    rets = np.diff(bars.close[-(w + 1) :])
    vols = bars.volume[-w:]
    up_vol = float(vols[rets > 0].sum())
    down_vol = float(vols[rets < 0].sum())
    vol_balance = float(np.clip((up_vol - down_vol) / (up_vol + down_vol + EPS), -1.0, 1.0))

    score, confidence = _combine(
        [mfi_score, obv_score, ad_score, vol_balance],
        weights=[1.0, 1.1, 1.0, 0.9],
    )
    rationale = (
        f"MFI={mfi_last:.0f}, OBV slope {'+' if obv_score >= 0 else '-'}, "
        f"A/D {'+' if ad_score >= 0 else '-'}, up/down-vol={vol_balance:+.2f}"
    )
    return SignalResult(
        name,
        score,
        confidence,
        rationale,
        detail={"mfi": mfi_last, "vol_balance": vol_balance},
    )


# --------------------------------------------------------------------------- #
# 5. Liquidity (depth gate + participation confirmation)
# --------------------------------------------------------------------------- #
def liquidity_signal(bars: Bars, p: SignalParams, price: float | None = None) -> SignalResult:
    name = "liquidity"
    if len(bars) < p.liquidity_window:
        return _neutral(name, "insufficient history")

    w = p.liquidity_window
    dollar_vol = bars.close[-w:] * bars.volume[-w:]
    adv_dollar = float(np.mean(dollar_vol))

    # Gate: log-scaled between the "too thin" floor and the "fully liquid" mark.
    lo = np.log10(p.liq_min_dollar_vol)
    hi = np.log10(p.liq_good_dollar_vol)
    gate = float(np.clip((np.log10(adv_dollar + 1.0) - lo) / (hi - lo + EPS), 0.0, 1.0))

    # Directional confirmation: is participation expanding in the trend direction?
    vol_slope = ind.linreg_slope(bars.volume, w)
    price_trend = np.sign(ind.linreg_slope(bars.close, w))
    score = float(np.clip(np.tanh(vol_slope * w) * price_trend, -1.0, 1.0))

    rationale = f"avg $vol≈${adv_dollar/1e6:.1f}M/bar, depth gate={gate:.0%}"
    return SignalResult(
        name,
        score,
        gate,  # confidence == the gate; strategy also uses this to scale the total
        rationale,
        detail={"avg_dollar_volume": adv_dollar, "gate": gate},
    )


def compute_all_signals(
    bars: Bars, params: SignalParams | None = None, price: float | None = None
) -> list[SignalResult]:
    """Run every factor and return the results in a stable order."""
    p = params or SignalParams()
    return [
        momentum_signal(bars, p, price),
        support_resistance_signal(bars, p, price),
        fibonacci_signal(bars, p, price),
        orderflow_signal(bars, p, price),
        liquidity_signal(bars, p, price),
    ]
