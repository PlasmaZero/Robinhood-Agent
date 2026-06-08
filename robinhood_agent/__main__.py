"""CLI entrypoint.

Two subcommands:

* ``agent`` — the LLM trading agent (Claude Agent SDK + Robinhood MCP).
* ``trade`` — the deterministic technical-analysis engine (this is the one that
  scores symbols on momentum / S-R / Fibonacci / order-flow / liquidity and
  acts above the confidence threshold).

For backward compatibility, if the first argument is not a known subcommand the
input is treated as an ``agent`` task, so ``python -m robinhood_agent "..."``
still works.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from .config import Config

_SUBCOMMANDS = {"agent", "trade", "scan"}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="robinhood-agent",
        description="Robinhood trading: deterministic TA engine (`trade`) or LLM agent (`agent`).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- agent ---
    p_agent = sub.add_parser("agent", help="Run the LLM trading agent.")
    p_agent.add_argument("task", nargs="?", default=None, help="Instruction for the agent.")
    p_agent.add_argument("--loop", action="store_true", help="Repeat on an interval.")
    p_agent.add_argument("--interval", type=int, default=3600, help="Seconds between agent runs.")

    # --- trade / scan ---
    for cmd in ("trade", "scan"):
        p = sub.add_parser(
            cmd,
            help="Score symbols and (optionally) trade them with the TA engine."
            if cmd == "trade"
            else "Analyze symbols only (alias for `trade` without execution).",
        )
        p.add_argument("symbols", nargs="*", help="Tickers to evaluate (default: WATCH_SYMBOLS).")
        p.add_argument("--demo", action="store_true", help="Use offline synthetic data (no creds).")
        p.add_argument(
            "--execute",
            action="store_true",
            help="Connect to Robinhood, size & review orders (places only if ALLOW_TRADING=true).",
        )
        p.add_argument("--threshold", type=float, default=None, help="Override confidence threshold (0-1).")
        p.add_argument("--bar-interval", default=None, help="Candle interval: 1d/1h/30m/15m/5m.")
        p.add_argument("--no-factors", action="store_true", help="Hide the per-factor breakdown.")
        p.add_argument("--loop", action="store_true", help="Repeat on an interval.")
        p.add_argument("--every", type=int, default=900, help="Seconds between loops (with --loop).")

    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# agent command
# --------------------------------------------------------------------------- #
def _run_agent(args: argparse.Namespace) -> None:
    from .trader import DEFAULT_TASK, RobinhoodTrader  # lazy: needs claude-agent-sdk

    config = Config.from_env()
    task = args.task or DEFAULT_TASK
    mode = "LIVE TRADING" if config.allow_trading else "DRY RUN (analysis only)"
    print("=" * 60)
    print(f"  Robinhood Agent (LLM) — {mode}")
    print(f"  Model: {config.model}")
    print("=" * 60)
    trader = RobinhoodTrader(config)

    async def once() -> None:
        await trader.run(task)

    async def loop() -> None:
        while True:
            start = time.monotonic()
            try:
                await trader.run(task)
            except Exception as exc:  # keep the loop alive across transient errors
                print(f"\n[run error] {exc!r}")
            sleep_for = max(0, args.interval - (time.monotonic() - start))
            print(f"\nNext run in {sleep_for:.0f}s (Ctrl-C to stop)...")
            await asyncio.sleep(sleep_for)

    asyncio.run(loop() if args.loop else once())


# --------------------------------------------------------------------------- #
# trade / scan command
# --------------------------------------------------------------------------- #
def _build_engine(args: argparse.Namespace, config: Config, executor=None):
    import dataclasses

    from .engine import TradingEngine

    params = config.build_strategy_params()
    if args.threshold is not None:
        params = dataclasses.replace(params, threshold=args.threshold)
    if args.bar_interval:
        config.data_interval = args.bar_interval

    if args.demo:
        from .synthetic import DemoProvider

        provider = DemoProvider()
    else:
        from .market_data import YFinanceProvider

        provider = YFinanceProvider()
    return TradingEngine(config, provider, executor=executor, strategy_params=params)


def _print_trade_banner(config: Config, args: argparse.Namespace, symbols: list[str]) -> None:
    if args.demo:
        mode = "DEMO (synthetic data)"
    elif not args.execute:
        mode = "SCAN (analysis only)"
    elif config.allow_trading:
        mode = "LIVE TRADING"
    else:
        mode = "DRY RUN (review only, no orders placed)"
    print("=" * 60)
    print(f"  Robinhood TA Engine — {mode}")
    print(f"  Threshold: {(args.threshold or config.confidence_threshold):.0%}   "
          f"Interval: {config.data_interval}   Symbols: {', '.join(symbols)}")
    print(f"  Per-order cap: ${config.max_order_value_usd:,.0f}   "
          f"Daily cap: ${config.max_daily_spend_usd:,.0f}")
    print("=" * 60)


def _run_trade(args: argparse.Namespace) -> None:
    from .report import render_decisions, render_records

    # Scan/demo need no credentials; execution needs the MCP token (+ account).
    require: tuple[str, ...] = ("ROBINHOOD_MCP_TOKEN",) if (args.execute and not args.demo) else ()
    config = Config.from_env(require=require)

    symbols = [s.upper() for s in args.symbols] or config.watch_symbols
    if not symbols:
        raise SystemExit("No symbols given. Pass them as arguments or set WATCH_SYMBOLS in .env.")

    do_execute = args.execute and not args.demo
    _print_trade_banner(config, args, symbols)

    async def execute_once() -> None:
        from .execution import RobinhoodMCPClient

        if not config.account_number:
            raise SystemExit("ROBINHOOD_ACCOUNT_NUMBER is required with --execute.")
        async with RobinhoodMCPClient(config.robinhood_mcp_url, config.robinhood_mcp_token) as client:
            engine = _build_engine(args, config, executor=client)
            records = await engine.run(symbols)
        print(render_decisions([r.decision for r in records], show_factors=not args.no_factors))
        print("\n" + "=" * 60)
        print(render_records(records))

    def scan_once() -> None:
        engine = _build_engine(args, config)
        decisions = engine.scan(symbols)
        print(render_decisions(decisions, show_factors=not args.no_factors))

    def do_once() -> None:
        if do_execute:
            asyncio.run(execute_once())
        else:
            scan_once()

    if not args.loop:
        do_once()
        return
    while True:
        start = time.monotonic()
        try:
            do_once()
        except SystemExit:
            raise
        except Exception as exc:
            print(f"\n[run error] {exc!r}")
        sleep_for = max(0, args.every - (time.monotonic() - start))
        print(f"\nNext scan in {sleep_for:.0f}s (Ctrl-C to stop)...")
        time.sleep(sleep_for)


def main() -> None:
    argv = sys.argv[1:]
    # Backward compat: bare task -> agent task.
    if argv and argv[0] not in _SUBCOMMANDS and argv[0] not in ("-h", "--help"):
        argv = ["agent", *argv]
    args = _parse_args(argv)

    try:
        if args.command == "agent":
            _run_agent(args)
        else:  # trade / scan
            if args.command == "scan":
                args.execute = False
            _run_trade(args)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
