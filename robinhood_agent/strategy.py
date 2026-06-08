"""Blend the factor signals into a single decision with a confidence score.

The contract you asked for: each factor votes, the votes are combined into one
confidence in ``[0, 1]``, and **any symbol whose confidence clears the threshold
(default 65%) is actionable** — BUY when the blend is bullish, SELL when it is
bearish. Not every factor has to agree; they are weighted and netted.

How the confidence is built (all terms in ``[0, 1]``):

    evidence_i  = score_i * confidence_i                  # signed, per factor
    weighted    = Σ wᵢ·evidence_i / Σ wᵢ                  # net direction, [-1, 1]
    agreement   = share of |weighted evidence| on the winning side
    conviction  = tanh(gain · |weighted|)                 # magnitude → [0, 1)
    confidence  = conviction · agreement · liquidity_gate

The liquidity gate multiplies the whole thing, so thin names are deliberately
suppressed even when the other factors look exciting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import indicators as ind
from .indicators import EPS
from .market_data import Bars
from .signals import SignalParams, SignalResult, compute_all_signals

BUY = "BUY"
SELL = "SELL"
HOLD = "HOLD"


@dataclass(frozen=True)
class FactorWeights:
    """Relative importance of each factor in the blend (need not sum to 1)."""

    momentum: float = 0.25
    support_resistance: float = 0.20
    fibonacci: float = 0.15
    orderflow: float = 0.25
    liquidity: float = 0.15

    def as_map(self) -> dict[str, float]:
        return {
            "momentum": self.momentum,
            "support_resistance": self.support_resistance,
            "fibonacci": self.fibonacci,
            "orderflow": self.orderflow,
            "liquidity": self.liquidity,
        }


@dataclass(frozen=True)
class StrategyParams:
    """Decision-level configuration."""

    threshold: float = 0.65  # confidence needed to act
    gain: float = 3.0  # sensitivity of conviction to net evidence
    weights: FactorWeights = field(default_factory=FactorWeights)
    signal_params: SignalParams = field(default_factory=SignalParams)
    atr_stop_mult: float = 2.0  # protective stop distance, in ATRs
    min_risk_reward: float = 1.5  # minimum target:risk to prefer a structural target
    size_floor_frac: float = 0.5  # notional at exactly threshold = this × max order


@dataclass(frozen=True)
class Decision:
    """The engine's verdict for one symbol."""

    symbol: str
    action: str  # BUY / SELL / HOLD
    direction: int  # +1 bullish, -1 bearish, 0 neutral
    confidence: float  # [0, 1]
    threshold: float
    price: float
    weighted_score: float
    agreement: float
    liquidity_gate: float
    atr: float
    stop_price: float | None
    target_price: float | None
    risk_reward: float | None
    signals: list[SignalResult]
    contributions: dict[str, float]  # name -> weighted signed evidence
    rationale: str

    @property
    def confidence_pct(self) -> float:
        return 100.0 * self.confidence

    @property
    def actionable(self) -> bool:
        return self.action in (BUY, SELL)


def _aggregate(signals: list[SignalResult], weights: FactorWeights, gain: float):
    wmap = weights.as_map()
    by_name = {s.name: s for s in signals}
    liquidity_gate = by_name["liquidity"].confidence if "liquidity" in by_name else 1.0

    num = 0.0
    wsum = 0.0
    pos = 0.0
    neg = 0.0
    contributions: dict[str, float] = {}
    for s in signals:
        w = wmap.get(s.name, 0.0)
        ev = s.evidence  # score * confidence, in [-1, 1]
        contributions[s.name] = w * ev
        num += w * ev
        wsum += w
        if ev > 0:
            pos += w * abs(ev)
        elif ev < 0:
            neg += w * abs(ev)

    weighted = num / (wsum + EPS)
    tot = pos + neg
    agreement = (max(pos, neg) / tot) if tot > EPS else 0.0
    conviction = float(np.tanh(gain * abs(weighted)))
    confidence = float(np.clip(conviction * agreement * liquidity_gate, 0.0, 1.0))
    direction = 1 if weighted > 0 else (-1 if weighted < 0 else 0)
    return weighted, agreement, liquidity_gate, confidence, direction, contributions


def _risk_levels(
    bars: Bars,
    price: float,
    direction: int,
    atr_last: float,
    sp: StrategyParams,
    signals: list[SignalResult],
) -> tuple[float | None, float | None, float | None]:
    """Compute (stop, target, risk_reward) for a long entry; None for non-buys."""
    if direction <= 0 or not np.isfinite(atr_last) or atr_last <= EPS:
        return None, None, None

    sr = next((s for s in signals if s.name == "support_resistance"), None)
    support = sr.detail.get("support") if sr else None
    resistance = sr.detail.get("resistance") if sr else None

    atr_stop = price - sp.atr_stop_mult * atr_last
    if support is not None and support < price:
        structural_stop = support - 0.25 * atr_last
        stop = max(structural_stop, atr_stop)  # the tighter (higher) of the two
    else:
        stop = atr_stop
    stop = min(stop, price - 0.1 * atr_last)  # never above (or at) entry

    risk = price - stop
    if risk <= EPS:
        return None, None, None

    if resistance is not None and resistance > price + sp.min_risk_reward * risk:
        target = resistance
    else:
        target = price + max(sp.min_risk_reward, 2.0) * risk
    rr = (target - price) / risk
    return float(stop), float(target), float(rr)


def decide(
    symbol: str,
    bars: Bars,
    params: StrategyParams | None = None,
    price: float | None = None,
) -> Decision:
    """Score ``bars`` and return a :class:`Decision`.

    ``price`` optionally overrides the last close with a live quote.
    """
    sp = params or StrategyParams()
    px = float(price) if price is not None else bars.last_price

    if len(bars) < sp.signal_params.min_bars:
        return Decision(
            symbol=symbol,
            action=HOLD,
            direction=0,
            confidence=0.0,
            threshold=sp.threshold,
            price=px,
            weighted_score=0.0,
            agreement=0.0,
            liquidity_gate=0.0,
            atr=float("nan"),
            stop_price=None,
            target_price=None,
            risk_reward=None,
            signals=[],
            contributions={},
            rationale=f"only {len(bars)} bars; need ≥{sp.signal_params.min_bars}",
        )

    signals = compute_all_signals(bars, sp.signal_params, px)
    weighted, agreement, gate, confidence, direction, contributions = _aggregate(
        signals, sp.weights, sp.gain
    )
    atr_last = ind.last_finite(ind.atr(bars.high, bars.low, bars.close, sp.signal_params.atr_period))

    if confidence >= sp.threshold and direction > 0:
        action = BUY
    elif confidence >= sp.threshold and direction < 0:
        action = SELL
    else:
        action = HOLD

    stop = target = rr = None
    if action == BUY:
        stop, target, rr = _risk_levels(bars, px, direction, atr_last, sp, signals)

    verb = {BUY: "BUY", SELL: "SELL", HOLD: "HOLD"}[action]
    rationale = (
        f"{verb} @ {confidence:.0%} (net {weighted:+.2f}, agreement {agreement:.0%}, "
        f"liquidity {gate:.0%}); threshold {sp.threshold:.0%}"
    )
    return Decision(
        symbol=symbol,
        action=action,
        direction=direction,
        confidence=confidence,
        threshold=sp.threshold,
        price=px,
        weighted_score=weighted,
        agreement=agreement,
        liquidity_gate=gate,
        atr=atr_last,
        stop_price=stop,
        target_price=target,
        risk_reward=rr,
        signals=signals,
        contributions=contributions,
        rationale=rationale,
    )


def suggested_notional(decision: Decision, max_order_value: float, size_floor_frac: float) -> float:
    """Scale order size with confidence: ``size_floor_frac`` of the cap at the
    threshold, ramping to the full cap at 100% confidence."""
    if not decision.actionable:
        return 0.0
    span = max(1e-6, 1.0 - decision.threshold)
    frac = size_floor_frac + (1.0 - size_floor_frac) * (decision.confidence - decision.threshold) / span
    frac = float(np.clip(frac, 0.0, 1.0))
    return round(max_order_value * frac, 2)
