"""Order execution against the Robinhood Agentic Trading MCP server.

The engine talks to an :class:`Executor` — an async interface for the account
reads and order calls it needs. :class:`RobinhoodMCPClient` is the real
implementation (a thin, direct MCP client over streamable HTTP); tests supply a
fake implementing the same protocol, so all the sizing/gating logic in
:mod:`robinhood_agent.engine` is exercised without touching a brokerage.

The ``mcp`` package is imported lazily inside :class:`RobinhoodMCPClient` so the
deterministic engine and its tests do not depend on it.

NOTE ON RESPONSE SHAPES: the exact JSON returned by the Robinhood MCP tools is
not pinned here. The parsing helpers look up several likely key spellings and
degrade gracefully (returning ``None``) rather than guessing wrong. If a field
comes back empty against the live server, adjust the key lists in
:func:`_first` — they are intentionally centralized.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Quote:
    symbol: str
    price: float  # last trade price
    bid: float | None = None
    ask: float | None = None


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    average_cost: float | None = None


@runtime_checkable
class Executor(Protocol):
    """Everything the engine needs from a brokerage backend."""

    async def get_buying_power(self, account: str) -> float | None: ...

    async def get_positions(self, account: str) -> dict[str, Position]: ...

    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]: ...

    async def review_order(self, **order: Any) -> dict[str, Any]: ...

    async def place_order(self, **order: Any) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- #
# Response parsing helpers (shared, tolerant)
# --------------------------------------------------------------------------- #
def _first(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_BUYING_POWER_KEYS = [
    "buying_power",
    "buyingPower",
    "equity_buying_power",
    "cash_available_for_trading",
    "withdrawable_amount",
]
_QTY_KEYS = ["quantity", "shares", "qty"]
_AVG_COST_KEYS = ["average_cost", "average_buy_price", "avg_cost", "averageCost"]
_PRICE_KEYS = ["last_trade_price", "last_price", "price", "mark_price", "last"]
_BID_KEYS = ["bid_price", "bid"]
_ASK_KEYS = ["ask_price", "ask"]


def parse_buying_power(payload: dict[str, Any]) -> float | None:
    # Buying power may live at the top level or nested under common containers.
    for container in (payload, payload.get("portfolio", {}), payload.get("account", {})):
        if isinstance(container, dict):
            val = _to_float(_first(container, _BUYING_POWER_KEYS))
            if val is not None:
                return val
    return None


def parse_positions(payload: dict[str, Any]) -> dict[str, Position]:
    rows = payload.get("positions") or payload.get("results") or payload.get("data") or []
    if isinstance(payload, list):
        rows = payload
    out: dict[str, Position] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sym = _first(row, ["symbol", "ticker", "instrument_symbol"])
        qty = _to_float(_first(row, _QTY_KEYS))
        if sym is None or qty is None:
            continue
        out[str(sym).upper()] = Position(
            symbol=str(sym).upper(),
            quantity=qty,
            average_cost=_to_float(_first(row, _AVG_COST_KEYS)),
        )
    return out


def parse_quotes(payload: dict[str, Any]) -> dict[str, Quote]:
    rows = payload.get("quotes") or payload.get("results") or payload.get("data") or []
    if isinstance(payload, list):
        rows = payload
    out: dict[str, Quote] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sym = _first(row, ["symbol", "ticker"])
        price = _to_float(_first(row, _PRICE_KEYS))
        if sym is None or price is None:
            continue
        out[str(sym).upper()] = Quote(
            symbol=str(sym).upper(),
            price=price,
            bid=_to_float(_first(row, _BID_KEYS)),
            ask=_to_float(_first(row, _ASK_KEYS)),
        )
    return out


class RobinhoodMCPClient:
    """Direct MCP client for the Robinhood Agentic Trading server.

    Use as an async context manager::

        async with RobinhoodMCPClient(url, token) as client:
            bp = await client.get_buying_power(account)
    """

    def __init__(self, url: str, token: str) -> None:
        self.url = url
        self.token = token
        self._session = None
        self._stack = None

    async def __aenter__(self) -> "RobinhoodMCPClient":
        try:
            from contextlib import AsyncExitStack

            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "The 'mcp' package is required for live execution. "
                "Install it with `pip install mcp`."
            ) from exc

        self._stack = AsyncExitStack()
        read, write, _ = await self._stack.enter_async_context(
            streamablehttp_client(self.url, headers={"Authorization": f"Bearer {self.token}"})
        )
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    async def _call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("RobinhoodMCPClient must be used as an async context manager")
        result = await self._session.call_tool(tool, args)
        if getattr(result, "isError", False):
            raise RuntimeError(f"{tool} failed: {_result_text(result)}")
        return _parse_result(result)

    async def get_buying_power(self, account: str) -> float | None:
        return parse_buying_power(await self._call("get_portfolio", {"account_number": account}))

    async def get_positions(self, account: str) -> dict[str, Position]:
        return parse_positions(await self._call("get_equity_positions", {"account_number": account}))

    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        return parse_quotes(await self._call("get_equity_quotes", {"symbols": symbols}))

    async def review_order(self, **order: Any) -> dict[str, Any]:
        return await self._call("review_equity_order", order)

    async def place_order(self, **order: Any) -> dict[str, Any]:
        return await self._call("place_equity_order", order)


def _result_text(result: Any) -> str:
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _parse_result(result: Any) -> dict[str, Any]:
    """Coerce an MCP tool result into a dict."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        # Some servers wrap the real payload in {"result": ...}.
        inner = structured.get("result", structured)
        if isinstance(inner, dict):
            return inner
        if isinstance(inner, list):
            return {"results": inner}
    text = _result_text(result)
    if text:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            if isinstance(data, list):
                return {"results": data}
        except json.JSONDecodeError:
            return {"raw": text}
    return {}
