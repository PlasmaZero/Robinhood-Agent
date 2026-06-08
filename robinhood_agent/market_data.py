"""Historical OHLCV data: the :class:`Bars` container and pluggable providers.

The Robinhood Agentic MCP server exposes quotes, positions, and order tools but
**not** historical candles, which every technical factor in this package needs.
So price history comes from a pluggable :class:`DataProvider`. The default is
:class:`YFinanceProvider` (free, no key). :class:`CSVProvider` reads local files
for backtesting / offline use, and :func:`bars_from_dict` builds a
:class:`Bars` from in-memory sequences (used by tests and the ``--demo`` mode).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Bars:
    """A validated OHLCV series, newest bar last.

    All five arrays share the same length. Timestamps are optional but, when
    present, must align with the price arrays.
    """

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    timestamps: np.ndarray | None = None
    symbol: str = ""
    interval: str = ""

    def __post_init__(self) -> None:
        arrays = {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }
        lengths = {name: len(a) for name, a in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"OHLCV arrays must share one length, got {lengths}")
        n = next(iter(lengths.values()))
        if n == 0:
            raise ValueError("Bars cannot be empty")
        for name, a in arrays.items():
            if not np.all(np.isfinite(a)):
                raise ValueError(f"{name} contains non-finite values")
        if np.any(self.high + 1e-9 < self.low):
            raise ValueError("found high < low")
        if np.any(self.volume < 0):
            raise ValueError("volume cannot be negative")
        if self.timestamps is not None and len(self.timestamps) != n:
            raise ValueError("timestamps length must match price arrays")

    def __len__(self) -> int:
        return len(self.close)

    @property
    def last_price(self) -> float:
        return float(self.close[-1])


def bars_from_dict(data: dict, symbol: str = "", interval: str = "") -> Bars:
    """Build :class:`Bars` from a dict of sequences (keys: open/high/low/close/volume)."""

    def col(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=float)

    ts = np.asarray(data["timestamps"]) if "timestamps" in data else None
    return Bars(
        open=col("open"),
        high=col("high"),
        low=col("low"),
        close=col("close"),
        volume=col("volume"),
        timestamps=ts,
        symbol=symbol,
        interval=interval,
    )


class DataProvider(Protocol):
    """Anything that can return recent OHLCV history for a symbol."""

    def get_bars(self, symbol: str, interval: str, lookback: int) -> Bars:  # noqa: D401
        ...


class YFinanceProvider:
    """Fetches OHLCV from Yahoo Finance via the optional ``yfinance`` package.

    Imported lazily so the rest of the package (and its tests) does not depend
    on yfinance or a network connection.
    """

    # Map our generic interval names to yfinance's interval + a period that
    # comfortably covers `lookback` bars.
    _INTERVAL_MAP = {
        "1d": ("1d", "2y"),
        "1h": ("60m", "60d"),
        "30m": ("30m", "30d"),
        "15m": ("15m", "30d"),
        "5m": ("5m", "30d"),
    }

    def __init__(self, *, auto_adjust: bool = True) -> None:
        self.auto_adjust = auto_adjust

    def get_bars(self, symbol: str, interval: str = "1d", lookback: int = 300) -> Bars:
        try:
            import yfinance as yf  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "yfinance is required for live market data. Install it with "
                "`pip install yfinance`, or supply a different DataProvider."
            ) from exc

        yf_interval, period = self._INTERVAL_MAP.get(interval, ("1d", "2y"))
        df = yf.download(
            symbol,
            period=period,
            interval=yf_interval,
            auto_adjust=self.auto_adjust,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            raise RuntimeError(f"no data returned for {symbol!r} ({interval})")
        # yfinance may return MultiIndex columns for a single ticker; flatten.
        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        df = df.dropna().tail(lookback)
        return Bars(
            open=df["Open"].to_numpy(dtype=float),
            high=df["High"].to_numpy(dtype=float),
            low=df["Low"].to_numpy(dtype=float),
            close=df["Close"].to_numpy(dtype=float),
            volume=df["Volume"].to_numpy(dtype=float),
            timestamps=df.index.to_numpy(),
            symbol=symbol,
            interval=interval,
        )


class CSVProvider:
    """Reads OHLCV from per-symbol CSV files: ``{dir}/{SYMBOL}.csv``.

    Expected header (case-insensitive): date/timestamp, open, high, low, close,
    volume. Rows must be chronological (oldest first).
    """

    def __init__(self, directory: str) -> None:
        self.directory = directory

    def get_bars(self, symbol: str, interval: str = "1d", lookback: int = 300) -> Bars:
        import csv
        import os

        path = os.path.join(self.directory, f"{symbol.upper()}.csv")
        cols: dict[str, list[float]] = {k: [] for k in ("open", "high", "low", "close", "volume")}
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            field_map = {name.lower().strip(): name for name in (reader.fieldnames or [])}
            for key in cols:
                if key not in field_map:
                    raise ValueError(f"{path} is missing a {key!r} column")
            for row in reader:
                for key in cols:
                    cols[key].append(float(row[field_map[key]]))
        bars = bars_from_dict(cols, symbol=symbol, interval=interval)
        if lookback and len(bars) > lookback:
            sl = slice(-lookback, None)
            bars = Bars(
                open=bars.open[sl],
                high=bars.high[sl],
                low=bars.low[sl],
                close=bars.close[sl],
                volume=bars.volume[sl],
                symbol=symbol,
                interval=interval,
            )
        return bars


def make_bars(
    open_: Sequence[float],
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    volume: Sequence[float],
    symbol: str = "",
    interval: str = "",
) -> Bars:
    """Convenience constructor from plain sequences."""
    return bars_from_dict(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        symbol=symbol,
        interval=interval,
    )
