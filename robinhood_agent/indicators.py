"""Low-level technical indicators.

Pure, dependency-light (numpy only) numeric functions. Every function is
deterministic and side-effect free so it can be unit-tested in isolation. All
series functions return a float array the same length as the input, padded with
``np.nan`` during the warm-up period, so callers can always index ``[-1]`` for
"the latest value" without bookkeeping.

These are the building blocks; the opinionated, trade-oriented interpretation
lives in :mod:`robinhood_agent.signals`.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def _as_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("expected a 1-D series")
    return arr


def sma(values, period: int) -> np.ndarray:
    """Simple moving average. NaN for the first ``period - 1`` entries."""
    v = _as_array(values)
    n = len(v)
    out = np.full(n, np.nan)
    if period <= 0 or n < period:
        return out
    cumsum = np.cumsum(np.insert(v, 0, 0.0))
    out[period - 1 :] = (cumsum[period:] - cumsum[:-period]) / period
    return out


def ema(values, period: int) -> np.ndarray:
    """Exponential moving average seeded with the first value."""
    v = _as_array(values)
    n = len(v)
    out = np.full(n, np.nan)
    if period <= 0 or n == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    out[0] = v[0]
    for i in range(1, n):
        out[i] = alpha * v[i] + (1.0 - alpha) * out[i - 1]
    return out


def _wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing (used by RSI/ATR/MFI). NaN until the first average."""
    n = len(values)
    out = np.full(n, np.nan)
    if n < period or period <= 0:
        return out
    out[period - 1] = values[:period].mean()
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + values[i]) / period
    return out


def rsi(close, period: int = 14) -> np.ndarray:
    """Wilder's Relative Strength Index in [0, 100]."""
    c = _as_array(close)
    n = len(c)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    delta = np.diff(c)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # diff drops one element; align smoothed averages back onto price indices.
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)
    rs = avg_gain / (avg_loss + EPS)
    rsi_vals = 100.0 - 100.0 / (1.0 + rs)
    # When there are no losses at all, RSI is exactly 100.
    rsi_vals = np.where(avg_loss <= EPS, 100.0, rsi_vals)
    out[1:] = rsi_vals
    return out


def macd(
    close, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line, and histogram (line - signal)."""
    c = _as_array(close)
    macd_line = ema(c, fast) - ema(c, slow)
    # Signal line is an EMA of the MACD line; compute on the finite tail.
    signal_line = np.full_like(macd_line, np.nan)
    finite = ~np.isnan(macd_line)
    if finite.any():
        first = int(np.argmax(finite))
        signal_line[first:] = ema(macd_line[first:], signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(high, low, close) -> np.ndarray:
    """True range. The first element uses high-low (no prior close)."""
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    prev_close = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum.reduce(
        [h - l, np.abs(h - prev_close), np.abs(l - prev_close)]
    )
    return tr


def atr(high, low, close, period: int = 14) -> np.ndarray:
    """Average True Range (Wilder)."""
    tr = true_range(high, low, close)
    return _wilder_smooth(tr, period)


def obv(close, volume) -> np.ndarray:
    """On-Balance Volume (cumulative signed volume)."""
    c, vol = _as_array(close), _as_array(volume)
    direction = np.sign(np.diff(c))
    out = np.zeros(len(c))
    out[1:] = np.cumsum(direction * vol[1:])
    return out


def chaikin_ad_line(high, low, close, volume) -> np.ndarray:
    """Accumulation/Distribution line (Chaikin)."""
    h, l, c, vol = (_as_array(high), _as_array(low), _as_array(close), _as_array(volume))
    rng = h - l
    clv = np.where(rng > EPS, ((c - l) - (h - c)) / (rng + EPS), 0.0)
    return np.cumsum(clv * vol)


def mfi(high, low, close, volume, period: int = 14) -> np.ndarray:
    """Money Flow Index in [0, 100] — volume-weighted RSI of typical price."""
    h, l, c, vol = (_as_array(high), _as_array(low), _as_array(close), _as_array(volume))
    n = len(c)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    tp = (h + l + c) / 3.0
    raw_flow = tp * vol
    delta_tp = np.diff(tp)
    pos_flow = np.where(delta_tp > 0, raw_flow[1:], 0.0)
    neg_flow = np.where(delta_tp < 0, raw_flow[1:], 0.0)
    # Rolling sums over `period` of the (n-1) flow values.
    pos_sum = _rolling_sum(pos_flow, period)
    neg_sum = _rolling_sum(neg_flow, period)
    mfr = pos_sum / (neg_sum + EPS)
    mfi_vals = 100.0 - 100.0 / (1.0 + mfr)
    mfi_vals = np.where(neg_sum <= EPS, 100.0, mfi_vals)
    out[1:] = mfi_vals
    return out


def _rolling_sum(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan)
    if n < period:
        return out
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    out[period - 1 :] = cumsum[period:] - cumsum[:-period]
    return out


def roc(close, period: int) -> np.ndarray:
    """Rate of change in percent over ``period`` bars."""
    c = _as_array(close)
    n = len(c)
    out = np.full(n, np.nan)
    if n <= period:
        return out
    out[period:] = (c[period:] / (c[:-period] + EPS) - 1.0) * 100.0
    return out


def linreg_slope(values, period: int) -> float:
    """Least-squares slope (per bar) of the last ``period`` points.

    Returns the slope normalized by the mean level, i.e. an approximate
    per-bar fractional change, so it is comparable across instruments of
    different price. ``nan`` if there is not enough data.
    """
    v = _as_array(values)
    if len(v) < period or period < 2:
        return float("nan")
    y = v[-period:]
    x = np.arange(period, dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    level = np.mean(np.abs(y)) + EPS
    return float(slope / level)


def swing_high_indices(high, left: int = 2, right: int = 2) -> list[int]:
    """Indices of fractal swing highs (local maxima with `left`/`right` bars)."""
    h = _as_array(high)
    n = len(h)
    out: list[int] = []
    for i in range(left, n - right):
        window = h[i - left : i + right + 1]
        if h[i] >= window.max() and h[i] > h[i - 1] and h[i] > h[i + 1]:
            out.append(i)
    return out


def swing_low_indices(low, left: int = 2, right: int = 2) -> list[int]:
    """Indices of fractal swing lows (local minima with `left`/`right` bars)."""
    l = _as_array(low)
    n = len(l)
    out: list[int] = []
    for i in range(left, n - right):
        window = l[i - left : i + right + 1]
        if l[i] <= window.min() and l[i] < l[i - 1] and l[i] < l[i + 1]:
            out.append(i)
    return out


def last_finite(series: np.ndarray) -> float:
    """Return the last non-NaN value of a series, or nan if none."""
    arr = np.asarray(series, dtype=float)
    finite = arr[~np.isnan(arr)]
    return float(finite[-1]) if finite.size else float("nan")
