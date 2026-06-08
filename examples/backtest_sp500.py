#!/usr/bin/env python3
"""Backtest the TA engine on a combined OHLCV CSV (date,open,high,low,close,volume,Name).

Usage:
    python examples/backtest_sp500.py path/to/all_stocks_5yr.csv            # bias-free basket
    python examples/backtest_sp500.py path/to/all_stocks_5yr.csv AAPL MSFT  # specific symbols

Dataset used in BACKTEST.md (S&P 500, 2013-2018):
    https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv

This reports results honestly against a buy-and-hold benchmark. It does not tune
anything to the data — see BACKTEST.md for why that matters.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# Allow running directly from a checkout without `pip install -e .`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robinhood_agent.backtest import load_long_csv, pooled_summary, run_backtest


def main(argv: list[str]) -> None:
    if not argv:
        print(__doc__)
        raise SystemExit(2)

    path = argv[0]
    symbols = [s.upper() for s in argv[1:]] or None
    if symbols is None:
        # Performance-blind selection: every 11th symbol alphabetically.
        import csv

        names = set()
        with open(path) as fh:
            for row in csv.DictReader(fh):
                names.add(row["Name"])
        symbols = sorted(names)[::11]
        print(f"No symbols given; using a bias-free basket of {len(symbols)} (every 11th).")

    t = time.time()
    data = load_long_csv(path, symbols=symbols)
    results = [run_backtest(sym, bars) for sym, bars in data.items()]
    s = pooled_summary(results)
    print(f"Backtested {len(results)} symbols in {time.time() - t:.0f}s\n")

    print("=== PER-TRADE EDGE (pooled) ===")
    print(f"  trades={s['n_trades']}  win_rate={s['win_rate']:.1%}  profit_factor={s['profit_factor']:.2f}")
    print(f"  avg_win={s['avg_win_pct']:+.2%}  avg_loss={s['avg_loss_pct']:+.2%}  "
          f"expectancy/trade={s['expectancy_pct']:+.2%}")
    print(f"  avg_bars_held={s['avg_bars_held']:.0f}  exits={s['exit_breakdown']}\n")

    print("=== RETURN vs BUY-AND-HOLD ===")
    print(f"  strategy  mean={s['strat_mean_return']:+.1%}  median={s['strat_median_return']:+.1%}")
    print(f"  buy&hold  mean={s['buyhold_mean_return']:+.1%}  median={s['buyhold_median_return']:+.1%}")
    print(f"  % of symbols strategy beat buy&hold: {s['pct_beating_buyhold']:.0%}\n")

    print("=== RISK-ADJUSTED ===")
    print(f"  strategy  maxDD={s['strat_mean_max_dd']:.1%}  Sharpe={s['strat_mean_sharpe']:.2f}  "
          f"exposure={s['mean_exposure']:.0%}")
    print(f"  buy&hold  maxDD={s['buyhold_mean_max_dd']:.1%}  Sharpe={s['buyhold_mean_sharpe']:.2f}  "
          f"exposure=100%")


if __name__ == "__main__":
    main(sys.argv[1:])
