"""CLI entrypoint for the Robinhood trading agent."""

from __future__ import annotations

import argparse
import asyncio
import time

from .config import Config
from .trader import DEFAULT_TASK, RobinhoodTrader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="robinhood-agent",
        description="Autonomous Robinhood trading agent (Claude Agent SDK + Robinhood MCP).",
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=DEFAULT_TASK,
        help="The instruction to run. Defaults to a portfolio review.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run the task repeatedly on an interval instead of once.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Seconds between runs when --loop is set (default: 3600).",
    )
    return parser.parse_args()


def _print_banner(config: Config) -> None:
    mode = "LIVE TRADING" if config.allow_trading else "DRY RUN (analysis only)"
    print("=" * 60)
    print(f"  Robinhood Agent Trader — {mode}")
    print(f"  Model: {config.model}")
    print(f"  Per-order cap: ${config.max_order_value_usd:,.2f}")
    print(f"  Daily cap:     ${config.max_daily_spend_usd:,.2f}")
    if config.allowed_symbols:
        print(f"  Symbols:       {', '.join(config.allowed_symbols)}")
    print("=" * 60)


async def _run_once(trader: RobinhoodTrader, task: str) -> None:
    await trader.run(task)


async def _run_loop(trader: RobinhoodTrader, task: str, interval: int) -> None:
    while True:
        start = time.monotonic()
        try:
            await trader.run(task)
        except Exception as exc:  # keep the loop alive across transient errors
            print(f"\n[run error] {exc!r}")
        elapsed = time.monotonic() - start
        sleep_for = max(0, interval - elapsed)
        print(f"\nNext run in {sleep_for:.0f}s (Ctrl-C to stop)...")
        await asyncio.sleep(sleep_for)


def main() -> None:
    args = _parse_args()
    config = Config.from_env()
    _print_banner(config)
    trader = RobinhoodTrader(config)

    try:
        if args.loop:
            asyncio.run(_run_loop(trader, args.task, args.interval))
        else:
            asyncio.run(_run_once(trader, args.task))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
