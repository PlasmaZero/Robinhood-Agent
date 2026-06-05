"""Configuration and risk parameters, loaded from the environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# MCP tool names containing any of these substrings are treated as
# state-changing (order placement / cancellation / funding). They are kept out
# of the agent's tool allowlist unless ALLOW_TRADING is true. This is the
# code-enforced safety layer — the agent literally cannot call a tool that
# isn't in the allowlist. Adjust to match the Robinhood MCP server's actual
# tool names (inspect them with `/mcp` in Claude Code after connecting).
RISKY_TOOL_KEYWORDS: tuple[str, ...] = (
    "place_order",
    "submit_order",
    "cancel_order",
    "buy",
    "sell",
    "transfer",
    "withdraw",
    "deposit",
)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass
class Config:
    """Runtime configuration for the trading agent."""

    anthropic_api_key: str
    robinhood_mcp_token: str
    robinhood_mcp_url: str = "https://agent.robinhood.com/mcp/trading"
    model: str = "claude-opus-4-8"

    allow_trading: bool = False
    max_order_value_usd: float = 500.0
    max_daily_spend_usd: float = 2000.0
    allowed_symbols: list[str] = field(default_factory=list)
    strategy: str = ""

    # Logical name the MCP server is registered under inside the Agent SDK.
    # Tools are then exposed to the agent as `mcp__<server_name>__<tool>`.
    server_name: str = "robinhood"

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        token = os.environ.get("ROBINHOOD_MCP_TOKEN", "").strip()

        missing = [
            name
            for name, value in (
                ("ANTHROPIC_API_KEY", api_key),
                ("ROBINHOOD_MCP_TOKEN", token),
            )
            if not value
        ]
        if missing:
            raise SystemExit(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ".\nCopy .env.example to .env and fill them in."
            )

        symbols_raw = os.environ.get("ALLOWED_SYMBOLS", "")
        allowed_symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]

        return cls(
            anthropic_api_key=api_key,
            robinhood_mcp_token=token,
            robinhood_mcp_url=os.environ.get(
                "ROBINHOOD_MCP_URL", "https://agent.robinhood.com/mcp/trading"
            ),
            model=os.environ.get("CLAUDE_MODEL", "claude-opus-4-8"),
            allow_trading=_get_bool("ALLOW_TRADING", False),
            max_order_value_usd=_get_float("MAX_ORDER_VALUE_USD", 500.0),
            max_daily_spend_usd=_get_float("MAX_DAILY_SPEND_USD", 2000.0),
            allowed_symbols=allowed_symbols,
            strategy=os.environ.get("STRATEGY", "").strip(),
        )
