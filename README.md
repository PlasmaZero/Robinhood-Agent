# Robinhood Agent Trader

An autonomous trading agent built on the **[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/python)** that
connects to **[Robinhood's official Agentic Trading MCP server](https://robinhood.com/us/en/support/articles/agentic-trading-overview/)**.

The agent reads your portfolio, applies a strategy and risk rules you define, and (when explicitly enabled) places
orders — into your dedicated Robinhood **Agentic** account, with order previews. Claude provides the judgment; the
Robinhood MCP server provides the hands.

> ⚠️ **This trades real money.** Read the [Safety](#safety) section before enabling live trading. Start in dry-run mode.

---

## Architecture

```
┌─────────────────────────┐    MCP over HTTP (OAuth)    ┌────────────────────────────┐
│  robinhood_agent.trader │ ──────────────────────────▶ │  Robinhood Agentic Trading │
│  (Claude Agent SDK loop)│                             │  MCP server (official)     │
│                         │ ◀────────────────────────── └────────────────────────────┘
│  • system prompt:       │   tools: get_portfolio, get_quote,
│    strategy + risk caps │          preview_order, place_order, ...
│  • tool allowlist gate  │
│  • one-shot or scheduled│
└─────────────────────────┘
```

The agent's *judgment* lives in `robinhood_agent/prompts.py` (strategy + hard risk rules). The *capabilities* come
from the Robinhood MCP server. A **tool allowlist** (`config.py`) is the load-bearing safety control: order-placement
tools are excluded unless you opt in.

---

## Prerequisites

1. **Python 3.11+**
2. **An Anthropic API key** — `ANTHROPIC_API_KEY` (the Agent SDK uses it to drive Claude).
3. **Node.js + the Claude Code CLI** — the Python Agent SDK drives the agent loop through the Claude Code runtime:
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```
4. **A Robinhood Agentic Trading credential** — an OAuth bearer token for `https://agent.robinhood.com/mcp/trading`.
   Obtain it by following Robinhood's Agentic Trading setup
   ([docs](https://robinhood.com/us/en/support/articles/agentic-trading-overview/)). Store it as
   `ROBINHOOD_MCP_TOKEN`.

> Verify the MCP endpoint against Robinhood's own published docs before connecting — you are granting an external
> server the ability to act on a brokerage account.

---

## Setup

```bash
pip install -e .                 # or: pip install -r requirements.txt
cp .env.example .env             # then fill in ANTHROPIC_API_KEY and ROBINHOOD_MCP_TOKEN
```

---

## Usage

**Dry run (default — analyzes and proposes, never places orders):**
```bash
python -m robinhood_agent "Review my portfolio and propose any rebalancing trades."
```

**Live trading (must be explicitly enabled):**
```bash
ALLOW_TRADING=true python -m robinhood_agent "Rebalance toward my target allocation."
```

**Scheduled loop (runs the default task on an interval):**
```bash
python -m robinhood_agent --loop --interval 3600
```

Run `python -m robinhood_agent --help` for all flags.

---

## Configuration

All knobs live in `.env` (loaded by `robinhood_agent/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Anthropic API key (required) |
| `ROBINHOOD_MCP_TOKEN` | — | OAuth bearer token for the Robinhood MCP server (required) |
| `ROBINHOOD_MCP_URL` | `https://agent.robinhood.com/mcp/trading` | MCP endpoint |
| `CLAUDE_MODEL` | `claude-opus-4-8` | Model id |
| `ALLOW_TRADING` | `false` | When `false`, order-placement tools are removed from the allowlist (dry run) |
| `MAX_ORDER_VALUE_USD` | `500` | Hard per-order cap, enforced in the system prompt |
| `MAX_DAILY_SPEND_USD` | `2000` | Daily cumulative cap, enforced in the system prompt |
| `ALLOWED_SYMBOLS` | (empty = any) | Comma-separated ticker allowlist, e.g. `VOO,VTI,AAPL` |
| `STRATEGY` | a conservative default | Free-text strategy injected into the system prompt |

---

## Safety

This agent can move real money. The design has three layers of defense:

1. **Tool allowlist (code-enforced).** In dry-run mode, the order-placement tool is not in `allowed_tools` at all, so
   the agent *cannot* place an order even if it tries. See `RISKY_TOOL_KEYWORDS` in `config.py`.
2. **Risk caps (prompt-enforced).** Per-order and daily spend caps, an optional symbol allowlist, and a
   "preview-then-confirm" rule are baked into the system prompt.
3. **Robinhood's own guardrails.** The official MCP scopes the agent to your Agentic account and previews orders.

**Recommended rollout:** dry run → tiny caps with one symbol → widen gradually. Never run with
`permission_mode="bypassPermissions"` and live trading at the same time without a human in the loop.

For true unattended autonomy, add a programmatic permission callback that inspects each `place_order` call's arguments
against the caps before allowing it — see the TODO in `trader.py`.
