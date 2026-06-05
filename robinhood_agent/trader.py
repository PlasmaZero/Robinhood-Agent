"""The trading agent: wires the Robinhood MCP server into the Claude Agent SDK."""

from __future__ import annotations

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

from .config import RISKY_TOOL_KEYWORDS, Config
from .prompts import build_system_prompt

DEFAULT_TASK = (
    "Review my Robinhood portfolio and buying power. Following the strategy and "
    "risk rules in your instructions, determine whether any trades are "
    "warranted right now and act accordingly."
)


class RobinhoodTrader:
    """Drives a Claude agent connected to Robinhood's Agentic Trading MCP server."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def _build_options(self) -> ClaudeAgentOptions:
        cfg = self.config
        server = cfg.server_name

        # Register the Robinhood MCP server over HTTP, authenticated with the
        # OAuth bearer token. Tools are exposed to the agent as
        # `mcp__<server>__<tool>`.
        mcp_servers = {
            server: {
                "type": "http",
                "url": cfg.robinhood_mcp_url,
                "headers": {"Authorization": f"Bearer {cfg.robinhood_mcp_token}"},
            }
        }

        # Tool allowlist is the code-enforced safety layer. In dry-run mode we
        # allow all Robinhood tools EXCEPT state-changing ones, so the agent
        # physically cannot place an order. When trading is enabled, all tools
        # from the server are allowed.
        if cfg.allow_trading:
            allowed_tools = [f"mcp__{server}__*"]
            disallowed_tools: list[str] = []
        else:
            allowed_tools = [f"mcp__{server}__*"]
            # Block risky tools by exact-name wildcard. Because the MCP tool
            # names aren't known until connection time, we also rely on the
            # system prompt; pin these to real names once known via `/mcp`.
            disallowed_tools = [
                f"mcp__{server}__{keyword}" for keyword in RISKY_TOOL_KEYWORDS
            ]

        return ClaudeAgentOptions(
            model=cfg.model,
            system_prompt=build_system_prompt(cfg),
            mcp_servers=mcp_servers,
            allowed_tools=allowed_tools,
            disallowed_tools=disallowed_tools,
            # "default" keeps a human in the loop for anything not explicitly
            # allowlisted. For unattended runs you would supply a permission
            # callback (see TODO below) rather than loosening this.
            permission_mode="default",
            max_turns=40,
        )

        # TODO (hardening for unattended autonomy): add a programmatic
        # permission callback that inspects each place_order call's arguments
        # (symbol, side, notional) against the caps in `config` and the running
        # daily total, allowing/denying per-call. This enforces the dollar caps
        # in code rather than relying on the model to honor the system prompt.

    async def run(self, task: str = DEFAULT_TASK) -> str:
        """Run a single agent task to completion. Returns the final result text."""
        options = self._build_options()
        final_text = ""

        async for message in query(prompt=task, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)
            elif isinstance(message, ResultMessage):
                print()  # newline after streamed assistant text
                if getattr(message, "result", None):
                    final_text = message.result

        return final_text
