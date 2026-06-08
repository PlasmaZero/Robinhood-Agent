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


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _get_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


def _get_list(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


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

    # --- Deterministic technical-analysis engine ---
    # Brokerage account for the TA engine to trade. Required for live trading;
    # must be an agentic-enabled account number (not defaulted from the API).
    account_number: str = ""
    # Default universe scanned by `python -m robinhood_agent trade` (CLI args win).
    watch_symbols: list[str] = field(default_factory=list)

    data_interval: str = "1d"  # 1d / 1h / 30m / 15m / 5m
    lookback_bars: int = 300

    confidence_threshold: float = 0.65  # act when blended confidence ≥ this
    decision_gain: float = 3.0  # conviction sensitivity to net evidence

    # Factor weights (relative; need not sum to 1).
    weight_momentum: float = 0.25
    weight_support_resistance: float = 0.20
    weight_fibonacci: float = 0.15
    weight_orderflow: float = 0.25
    weight_liquidity: float = 0.15

    # Risk / sizing.
    atr_stop_mult: float = 2.0
    min_risk_reward: float = 1.5
    size_floor_frac: float = 0.5  # order size at exactly the threshold, × max order
    liq_min_dollar_vol: float = 1_000_000.0
    liq_good_dollar_vol: float = 25_000_000.0

    # Order routing.
    order_type: str = "limit"  # 'limit' (marketable, price-protected) or 'market'
    slippage_bps: float = 10.0  # marketable-limit cushion, in basis points
    time_in_force: str = "gfd"
    market_hours: str = "regular_hours"

    def build_strategy_params(self):
        """Construct a StrategyParams from this config (lazy import to keep the
        agent layer free of the numpy-dependent engine)."""
        from .signals import SignalParams
        from .strategy import FactorWeights, StrategyParams

        return StrategyParams(
            threshold=self.confidence_threshold,
            gain=self.decision_gain,
            weights=FactorWeights(
                momentum=self.weight_momentum,
                support_resistance=self.weight_support_resistance,
                fibonacci=self.weight_fibonacci,
                orderflow=self.weight_orderflow,
                liquidity=self.weight_liquidity,
            ),
            signal_params=SignalParams(
                liq_min_dollar_vol=self.liq_min_dollar_vol,
                liq_good_dollar_vol=self.liq_good_dollar_vol,
            ),
            atr_stop_mult=self.atr_stop_mult,
            min_risk_reward=self.min_risk_reward,
            size_floor_frac=self.size_floor_frac,
        )

    @classmethod
    def from_env(
        cls,
        require: tuple[str, ...] = ("ANTHROPIC_API_KEY", "ROBINHOOD_MCP_TOKEN"),
    ) -> "Config":
        """Load config from the environment.

        ``require`` lists env vars that must be present. The LLM agent needs both
        the Anthropic key and the MCP token; the deterministic engine's scan /
        dry-run modes need neither, so callers can pass a narrower set.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        token = os.environ.get("ROBINHOOD_MCP_TOKEN", "").strip()

        present = {"ANTHROPIC_API_KEY": api_key, "ROBINHOOD_MCP_TOKEN": token}
        missing = [name for name in require if not present.get(name, "")]
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
            account_number=_get_str("ROBINHOOD_ACCOUNT_NUMBER", ""),
            watch_symbols=_get_list("WATCH_SYMBOLS"),
            data_interval=_get_str("DATA_INTERVAL", "1d"),
            lookback_bars=_get_int("LOOKBACK_BARS", 300),
            confidence_threshold=_get_float("CONFIDENCE_THRESHOLD", 0.65),
            decision_gain=_get_float("DECISION_GAIN", 3.0),
            weight_momentum=_get_float("WEIGHT_MOMENTUM", 0.25),
            weight_support_resistance=_get_float("WEIGHT_SUPPORT_RESISTANCE", 0.20),
            weight_fibonacci=_get_float("WEIGHT_FIBONACCI", 0.15),
            weight_orderflow=_get_float("WEIGHT_ORDERFLOW", 0.25),
            weight_liquidity=_get_float("WEIGHT_LIQUIDITY", 0.15),
            atr_stop_mult=_get_float("ATR_STOP_MULT", 2.0),
            min_risk_reward=_get_float("MIN_RISK_REWARD", 1.5),
            size_floor_frac=_get_float("SIZE_FLOOR_FRAC", 0.5),
            liq_min_dollar_vol=_get_float("LIQ_MIN_DOLLAR_VOL", 1_000_000.0),
            liq_good_dollar_vol=_get_float("LIQ_GOOD_DOLLAR_VOL", 25_000_000.0),
            order_type=_get_str("ORDER_TYPE", "limit"),
            slippage_bps=_get_float("SLIPPAGE_BPS", 10.0),
            time_in_force=_get_str("TIME_IN_FORCE", "gfd"),
            market_hours=_get_str("MARKET_HOURS", "regular_hours"),
        )
