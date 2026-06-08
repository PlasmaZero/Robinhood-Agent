# Robinhood Agent Trader

Two cooperating ways to trade a Robinhood **Agentic** account, both wired to
**[Robinhood's official Agentic Trading MCP server](https://robinhood.com/us/en/support/articles/agentic-trading-overview/)**:

1. **`trade` — a deterministic technical-analysis engine.** It scores each symbol on **momentum, support/resistance,
   Fibonacci, order flow, and liquidity**, blends those into a single **confidence**, and acts when confidence clears a
   threshold (default **65%**). Pure, rule-based, and fully unit-tested — no LLM in the decision loop. **This is what
   most of this README is about.**
2. **`agent` — an LLM trading agent** built on the **[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/python)**,
   where Claude reads your portfolio and applies a free-text strategy. (The original mode; see [The LLM agent](#the-llm-agent).)

> ⚠️ **This trades real money.** Read the [Safety](#safety) section before enabling live trading. Both modes default to
> dry-run / analysis-only — you have to explicitly opt into placing orders.

---

## Quick start

```bash
pip install -e ".[live,dev]"      # or: pip install -r requirements.txt
cp .env.example .env              # fill in tokens for live use (not needed for --demo/scan)

# 1) See it work offline on synthetic data — no credentials, no network:
python -m robinhood_agent trade --demo AAPL MSFT TSLA NVDA GME

# 2) Analyze real symbols (needs internet for price data; no brokerage creds):
python -m robinhood_agent scan AAPL NVDA SPY

# 3) Size & review real orders against your account, WITHOUT placing them (dry run):
python -m robinhood_agent trade --execute AAPL NVDA          # ALLOW_TRADING=false

# 4) Actually place orders (real money — opt in explicitly):
ALLOW_TRADING=true python -m robinhood_agent trade --execute AAPL NVDA
```

---

## The technical-analysis engine

### The five factors

Each factor independently scores a symbol with a **direction** in `[-1, +1]` (bearish ↔ bullish) and its own
**confidence** in `[0, 1]`. They are deliberately decoupled — each is a small, testable function over OHLCV history.

| Factor | What it measures | Bullish read |
|---|---|---|
| **Momentum** | RSI, MACD histogram, rate-of-change, and the 20/50 MA stack | Strong, accelerating, trend-aligned price |
| **Support / Resistance** | Position within the prior range; **breakouts to new highs** vs. fades at the edges | Breakout on volume, or a bounce off well-tested support |
| **Fibonacci** | Retracement of the dominant swing | A shallow "buy-the-dip" pullback into the 0.5–0.618 zone in an uptrend |
| **Order flow** | MFI, OBV slope, Chaikin A/D line, up/down-volume balance | Net buying pressure / accumulation |
| **Liquidity** | Average **dollar** volume + participation trend | Deep, tradeable, with expanding volume in the trend |

> The S/R factor is **trend-aware**: a stock making new highs with no overhead supply reads as bullish *continuation*,
> not "at resistance" — a subtlety that trips up naive mean-reversion logic.

### From factors to a single confidence

Not every factor has to agree. They are weighted, netted, and squashed into one confidence:

```
evidence_i = score_i · confidence_i                    # signed, per factor
weighted   = Σ wᵢ·evidence_i / Σ wᵢ                    # net direction, [-1, +1]
agreement  = share of the weighted evidence on the winning side
conviction = tanh(gain · |weighted|)                   # magnitude → [0, 1)
confidence = conviction · agreement · liquidity_gate   # final, [0, 1]
```

The **liquidity gate multiplies the whole score**, so thin names are deliberately suppressed even when everything else
looks great. The decision is symmetric:

- `confidence ≥ threshold` and bullish → **BUY**
- `confidence ≥ threshold` and bearish → **SELL** (closes a long; shorting is off by default)
- otherwise → **HOLD**

Order size scales with confidence (from `SIZE_FLOOR_FRAC × MAX_ORDER_VALUE_USD` at the threshold up to the full cap at
100%), and every BUY gets an **ATR/structure-based stop and a risk-reward target**.

### Example output

```
SYMBOL  SIGNAL    CONF      PRICE    NET  AGREE   LIQ
-----------------------------------------------------
MSFT    ▲ BUY      85%     353.57  +0.42   100%  100%
SPY     · HOLD     28%      96.35  -0.11    88%  100%

  MSFT — BUY @ 85% (net +0.42, agreement 100%, liquidity 100%); threshold 65%
    momentum           +0.83 × 0.83  [     |###  ]  RSI=94, MACD-hist +, ROC=4.5%, MA-stack bullish
    support_resistance +1.00 × 0.80  [     |#### ]  breakout +1.9 ATR over 346.70
    orderflow          +0.47 × 0.47  [     |#    ]  MFI=91, OBV +, A/D +, up/down-vol=+0.80
    liquidity          +0.12 × 1.00  [     |#    ]  avg $vol≈$2403.7M/bar, depth gate=100%
    entry≈353.57  stop=346.27  target=368.19  R:R=2.00
```

### Where price data comes from

The Robinhood Agentic MCP server exposes quotes, positions, and order tools but **not historical candles**, which every
factor needs. So history comes from a pluggable `DataProvider` (`market_data.py`):

- **`YFinanceProvider`** (default) — free Yahoo Finance data, no key. `pip install ".[live]"`.
- **`CSVProvider`** — per-symbol CSV files, for backtests / offline.
- **`DemoProvider`** — deterministic synthetic regimes for `--demo`.

Live quotes from Robinhood are used as the current price at decision time; the candle history sets up the factors.

### Architecture (engine)

```
 DataProvider ─OHLCV─▶ signals.py ─5×(score,conf)─▶ strategy.py ─Decision─▶ engine.py ──▶ Robinhood MCP
 (yfinance/CSV)        momentum, S/R, fib,          weighted blend,         sizing,        review_equity_order
                       orderflow, liquidity         65% threshold,          caps, gate,    place_equity_order
                                                    stop/target             dry-run        (direct MCP client)
```

`indicators.py` (numpy-only math) → `signals.py` (the factors) → `strategy.py` (the blend + threshold) are **pure and
unit-tested**. `engine.py` adds sizing/caps; `execution.py` is the only piece that talks to the network.

---

## The LLM agent

The original mode: Claude reads your portfolio and applies a free-text strategy via the Claude Agent SDK.

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

The agent's *judgment* lives in `robinhood_agent/prompts.py` (strategy + hard risk rules). A **tool allowlist**
(`config.py`) is the load-bearing safety control: order-placement tools are excluded unless you opt in.

```bash
python -m robinhood_agent agent "Review my portfolio and propose any rebalancing trades."
ALLOW_TRADING=true python -m robinhood_agent agent "Rebalance toward my target allocation."
```

---

## Prerequisites

**TA engine (`trade` / `scan`):**
1. **Python 3.11+**
2. For live data & execution: `pip install ".[live]"` (numpy + yfinance + mcp), a **Robinhood Agentic Trading
   credential** (`ROBINHOOD_MCP_TOKEN`, an OAuth bearer token for `https://agent.robinhood.com/mcp/trading`), and your
   agentic **account number** (`ROBINHOOD_ACCOUNT_NUMBER`). `--demo` and `scan` need much less (none, and just data
   respectively).

**LLM agent (`agent`), additionally:**
3. **An Anthropic API key** — `ANTHROPIC_API_KEY` (the Agent SDK uses it to drive Claude).
4. **Node.js + the Claude Code CLI** — the Python Agent SDK drives the agent loop through the Claude Code runtime:
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

Obtain the Robinhood credential by following Robinhood's Agentic Trading setup
([docs](https://robinhood.com/us/en/support/articles/agentic-trading-overview/)).

> Verify the MCP endpoint against Robinhood's own published docs before connecting — you are granting an external
> server the ability to act on a brokerage account.

---

## Setup

```bash
pip install -e ".[live,dev]"     # engine + live data/exec + tests; or: pip install -r requirements.txt
cp .env.example .env             # fill in tokens (only needed for live execution)
```

The TA engine's `--demo` and `scan` modes need **no credentials and no network for `--demo`**. Live execution
(`trade --execute`) needs `ROBINHOOD_MCP_TOKEN` + `ROBINHOOD_ACCOUNT_NUMBER`; the LLM `agent` additionally needs
`ANTHROPIC_API_KEY`.

---

## Usage

```bash
# Offline demo on synthetic data (no creds, no network):
python -m robinhood_agent trade --demo AAPL MSFT NVDA

# Analyze real symbols (price data only; no brokerage connection):
python -m robinhood_agent scan AAPL NVDA SPY
python -m robinhood_agent scan NVDA --threshold 0.75 --no-factors

# Review real orders against your account but DON'T place them (dry run):
python -m robinhood_agent trade --execute AAPL NVDA

# Place orders (real money — must opt in):
ALLOW_TRADING=true python -m robinhood_agent trade --execute AAPL NVDA

# Run continuously (re-scan every 15 min):
python -m robinhood_agent trade --execute --loop --every 900
```

With no symbols, the engine uses `WATCH_SYMBOLS` from `.env`. Run `python -m robinhood_agent trade --help` for all flags.

---

## Configuration

All knobs live in `.env` (loaded by `robinhood_agent/config.py`).

**Shared / agent:**

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Anthropic API key (required for the `agent` command only) |
| `ROBINHOOD_MCP_TOKEN` | — | OAuth bearer token for the Robinhood MCP server (required for live trading) |
| `ROBINHOOD_MCP_URL` | `https://agent.robinhood.com/mcp/trading` | MCP endpoint |
| `CLAUDE_MODEL` | `claude-opus-4-8` | Model id (agent) |
| `ALLOW_TRADING` | `false` | When `false`, orders are reviewed but never placed (dry run) |
| `MAX_ORDER_VALUE_USD` | `500` | Hard per-order notional cap |
| `MAX_DAILY_SPEND_USD` | `2000` | Cumulative buy cap per run |
| `ALLOWED_SYMBOLS` | (empty = any) | Comma-separated ticker allowlist |
| `STRATEGY` | conservative default | Free-text strategy for the LLM agent |

**TA engine:**

| Variable | Default | Meaning |
|---|---|---|
| `ROBINHOOD_ACCOUNT_NUMBER` | — | Agentic account to trade (required for `trade --execute`) |
| `WATCH_SYMBOLS` | (empty) | Default universe when no symbols are passed |
| `DATA_INTERVAL` / `LOOKBACK_BARS` | `1d` / `300` | Candle interval and history depth |
| `CONFIDENCE_THRESHOLD` | `0.65` | Act when blended confidence ≥ this |
| `DECISION_GAIN` | `3.0` | Conviction sensitivity to net evidence |
| `WEIGHT_*` | `.25/.20/.15/.25/.15` | Per-factor weights (momentum / S-R / fib / orderflow / liquidity) |
| `LIQ_MIN_DOLLAR_VOL` / `LIQ_GOOD_DOLLAR_VOL` | `1e6` / `25e6` | Liquidity gate calibration |
| `ATR_STOP_MULT` / `MIN_RISK_REWARD` | `2.0` / `1.5` | Stop distance (ATRs) and minimum reward:risk |
| `SIZE_FLOOR_FRAC` | `0.5` | Order size at exactly the threshold, × per-order cap |
| `ORDER_TYPE` / `SLIPPAGE_BPS` | `limit` / `10` | `limit` (marketable, price-protected) or `market`; limit cushion |

---

## Safety

This software can move real money. Layers of defense:

1. **Dry-run by default.** Without `ALLOW_TRADING=true`, the engine calls `review_equity_order` (a simulation) and
   prints what it *would* do, but never calls `place_equity_order`. `scan`/`--demo` don't touch your account at all.
2. **Hard caps (code-enforced).** Per-order and per-run dollar caps and the optional symbol allowlist are enforced in
   `engine.py` *before* an order is built — not left to a model's discretion. Size is also bounded by buying power, and
   the engine refuses to place a buy if buying power can't be read.
3. **Liquidity gate.** Thin names are suppressed toward zero confidence, so the engine won't chase illiquid tickers.
4. **One position per symbol.** No pyramiding; SELL only closes existing longs (shorting is off).
5. **Robinhood's own guardrails.** The official MCP scopes everything to your Agentic account and previews orders.

**Recommended rollout:** `--demo` → `scan` on real data → `--execute` dry run → `ALLOW_TRADING=true` with tiny caps and
one symbol → widen gradually.

> **Not financial advice.** These factors are heuristics; a 65% "confidence" is a model score, **not** a probability of
> profit. Markets can gap through stops. Backtest and paper-trade before risking capital, and never trade money you
> can't afford to lose.

### Known limitations

- The daily cap is enforced **per run** (in-memory); it does not persist across separate process invocations.
- Order-flow factors are **proxied from OHLCV** (no level-2 / tick data via this MCP).
- Response-field parsing in `execution.py` is tolerant but not pinned to a verified live schema — confirm the first live
  fills against the Robinhood app, and adjust the key lists in `execution.py` if a field comes back empty.

For unattended autonomy on the **LLM agent**, also add a programmatic permission callback — see the TODO in `trader.py`.
