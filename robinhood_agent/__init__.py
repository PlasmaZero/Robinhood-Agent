"""Robinhood trading agent.

Two cooperating layers live here:

* A **deterministic technical-analysis engine** — :mod:`indicators`,
  :mod:`signals`, :mod:`strategy`, :mod:`market_data`, :mod:`engine` — that
  scores symbols on momentum, support/resistance, Fibonacci, order flow and
  liquidity, and acts when confidence clears a threshold. It depends only on
  numpy (plus an optional data provider / MCP client at run time).

* An **LLM agent** (:class:`RobinhoodTrader`) built on the Claude Agent SDK.

The agent layer is imported lazily so the TA engine can be used (and tested)
without the Claude Agent SDK installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["Config", "RobinhoodTrader", "TradingEngine"]

if TYPE_CHECKING:  # pragma: no cover - type-checker only
    from .config import Config
    from .engine import TradingEngine
    from .trader import RobinhoodTrader


def __getattr__(name: str):
    # Lazy attribute access keeps `import robinhood_agent.strategy` free of the
    # claude-agent-sdk / dotenv dependencies that only the agent layer needs.
    if name == "Config":
        from .config import Config

        return Config
    if name == "RobinhoodTrader":
        from .trader import RobinhoodTrader

        return RobinhoodTrader
    if name == "TradingEngine":
        from .engine import TradingEngine

        return TradingEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
