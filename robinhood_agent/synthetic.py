"""Deterministic synthetic OHLCV generators for tests and offline demos.

These let the engine be exercised with no network and no data provider, and give
the test-suite reproducible "textbook" regimes (clean uptrend, downtrend, chop)
to lock the decision calibration in place.
"""

from __future__ import annotations

import numpy as np

from .market_data import Bars


def _ohlc_from_close(
    close: np.ndarray, volume: np.ndarray, rng: np.random.Generator, wick: float
) -> Bars:
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    high = body_hi + np.abs(rng.normal(0, wick, len(close))) * close
    low = body_lo - np.abs(rng.normal(0, wick, len(close))) * close
    low = np.maximum(low, 0.01)
    return Bars(
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def trending_series(
    n: int = 200,
    start: float = 100.0,
    drift: float = 0.004,
    noise: float = 0.006,
    base_volume: float = 5_000_000.0,
    seed: int = 7,
    wick: float = 0.003,
) -> Bars:
    """Up (positive drift) or down (negative drift) trend with confirming volume.

    Volume leans toward the trend direction (heavier on with-trend days) so the
    order-flow and liquidity factors line up with price, as they tend to in a
    real, healthy trend.
    """
    rng = np.random.default_rng(seed)
    rets = drift + rng.normal(0, noise, n)
    close = start * np.cumprod(1.0 + rets)
    with_trend = np.sign(rets) == np.sign(drift)
    volume = base_volume * (1.0 + 0.5 * with_trend + rng.normal(0, 0.1, n))
    volume = np.clip(volume, base_volume * 0.2, None)
    return _ohlc_from_close(close, volume, rng, wick)


def choppy_series(
    n: int = 200,
    start: float = 100.0,
    noise: float = 0.008,
    base_volume: float = 5_000_000.0,
    seed: int = 11,
    wick: float = 0.003,
) -> Bars:
    """Sideways, mean-reverting price with no persistent direction."""
    rng = np.random.default_rng(seed)
    close = np.empty(n)
    close[0] = start
    for i in range(1, n):
        pull = (start - close[i - 1]) / start * 0.05  # mean reversion
        close[i] = close[i - 1] * (1.0 + pull + rng.normal(0, noise))
    volume = base_volume * (1.0 + rng.normal(0, 0.1, n))
    volume = np.clip(volume, base_volume * 0.2, None)
    return _ohlc_from_close(close, volume, rng, wick)


def pullback_in_uptrend(
    n: int = 200,
    start: float = 100.0,
    seed: int = 5,
    base_volume: float = 5_000_000.0,
) -> Bars:
    """A strong rally followed by a shallow pullback into the 0.5–0.618 zone.

    This is the canonical "buy the dip" Fibonacci/support setup.
    """
    rng = np.random.default_rng(seed)
    leg_up = int(n * 0.7)
    pull_n = int(n * 0.18)
    recover_n = n - leg_up - pull_n
    rets = 0.006 + rng.normal(0, 0.004, leg_up)
    up = start * np.cumprod(1.0 + rets)
    peak = up[-1]
    swing = peak - start
    # Retrace ~50% of the leg into the golden zone...
    trough = peak - 0.5 * swing
    pull = np.linspace(peak, trough, pull_n) * (1.0 + rng.normal(0, 0.002, pull_n))
    # ...then stabilize and tick back up: a genuine, holding dip.
    recover = np.linspace(pull[-1], pull[-1] * 1.04, recover_n) * (1.0 + rng.normal(0, 0.002, recover_n))
    close = np.concatenate([up, pull, recover])
    volume = base_volume * (1.0 + rng.normal(0, 0.1, n))
    volume[:leg_up] *= 1.4  # heavier volume on the impulse leg
    volume[leg_up : leg_up + pull_n] *= 0.7  # lighter volume on the pullback (healthy)
    volume = np.clip(volume, base_volume * 0.2, None)
    return _ohlc_from_close(close, volume, rng, wick=0.002)


def thin_uptrend(n: int = 200, seed: int = 9) -> Bars:
    """A clean uptrend but on tiny dollar volume (illiquid microcap)."""
    return trending_series(n=n, start=3.0, drift=0.005, base_volume=8_000.0, seed=seed)


class DemoProvider:
    """Offline :class:`DataProvider` that fabricates a regime per symbol.

    Lets ``python -m robinhood_agent trade --demo`` show the engine end-to-end
    with no network or credentials. The regime is chosen deterministically from
    the symbol so the same ticker always behaves the same way.
    """

    _REGIMES = ("uptrend", "downtrend", "choppy", "pullback")

    def get_bars(self, symbol: str, interval: str = "1d", lookback: int = 300) -> Bars:
        # Stable hash (builtin hash() is per-process randomized for strings, which
        # would make the demo non-deterministic across runs).
        import hashlib

        digest = int(hashlib.md5(symbol.upper().encode()).hexdigest(), 16)
        seed = digest % 1000
        regime = self._REGIMES[digest % len(self._REGIMES)]
        n = min(max(lookback, 120), 250)
        if regime == "uptrend":
            bars = trending_series(n=n, drift=0.005, seed=seed)
        elif regime == "downtrend":
            bars = trending_series(n=n, drift=-0.006, noise=0.003, seed=seed)
        elif regime == "pullback":
            bars = pullback_in_uptrend(n=n, seed=seed)
        else:
            bars = choppy_series(n=n, seed=seed)
        return Bars(
            open=bars.open,
            high=bars.high,
            low=bars.low,
            close=bars.close,
            volume=bars.volume,
            symbol=symbol.upper(),
            interval=interval,
        )
