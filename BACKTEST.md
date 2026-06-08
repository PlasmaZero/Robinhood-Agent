# Backtest results — an honest assessment

**Bottom line:** As built, this strategy is **not profitable in a way you should
trade on**. On real, out-of-sample data it has a razor-thin per-trade edge that
disappears at the portfolio level, and it badly underperforms simply buying and
holding the same stocks. Its only redeeming trait is lower drawdown — and that
comes from sitting in cash most of the time, not from skill (Sharpe ≈ 0).

No one can *guarantee* a trading strategy is profitable. What this document does
is measure it correctly and report the truth.

---

## Method (why these numbers are trustworthy)

- **Out-of-sample.** The decision thresholds/weights were calibrated on synthetic
  data, never on the test data below. So this is a genuine out-of-sample test.
- **No lookahead.** Every decision uses *only* data through the close of bar `t`;
  fills happen at bar `t+1`'s open. This invariant is enforced by a unit test
  (`test_no_lookahead_bias`) that checks the equity path up to bar K is identical
  whether or not future bars exist.
- **Realistic costs & exits.** 5 bps slippage on every fill; stops/targets checked
  intrabar with gap handling (a gap through your stop fills at the open).
- **Honest benchmark.** Every symbol is compared to buy-and-hold of *that symbol*
  over the *same* window.

## Data

[Kaggle / plotly **S&P 500 five-year** daily OHLCV](https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv),
Feb 2013 – Feb 2018 (505 symbols). To avoid cherry-picking winners, the basket is
**every 11th symbol alphabetically** (46 symbols, 976 trades) — selection made
before looking at any returns.

## Results (default parameters, 65% threshold)

**Per-trade edge (pooled, 976 trades)**

| metric | value |
|---|---|
| win rate | 36.0% |
| profit factor | **1.07** |
| avg win / avg loss | +5.02% / −2.58% |
| expectancy / trade | **+0.16%** |
| avg holding | 10 bars |
| exits | 620 stops, 351 targets, 4 signal, 1 eod |

**Return vs. buy-and-hold (2013–2018 bull market)**

| | strategy | buy & hold |
|---|---|---|
| mean return | +1.9% | **+78.0%** |
| median return | **−0.7%** | +62.9% |
| % of symbols strategy beat B&H | **15%** | — |

**Risk-adjusted**

| | strategy | buy & hold |
|---|---|---|
| mean max drawdown | **14.4%** | 32.2% |
| mean Sharpe | 0.02 | **0.62** |
| time in market | 19% | 100% |

Single-name spot checks: AAPL +12% vs +144%, MSFT **−14%** vs +169%,
NVDA +59% vs +1576%.

## What this means

- **Profit factor 1.07 / expectancy +0.16%/trade** is a real-but-tiny edge. It is
  *inside the noise* — well within what data-mining 46 symbols can produce by
  chance, and thin enough that real-world frictions (borrow, partial fills, taxes)
  could erase it.
- The strategy **misses the bull market** by being ~81% in cash. In a market that
  rose ~78%, any market-timing system that sits out most of the time will lose to
  buy-and-hold. The ~2× lower drawdown is mostly the mechanical result of low
  exposure, not alpha — the near-zero Sharpe confirms there's no risk-adjusted edge
  here either.
- **This 5-year window contains no major bear market.** A defensive timing
  strategy's whole value proposition — sidestepping crashes — can't show up in a
  sample with no crash. Judging it only here is structurally unfair *and* it still
  shouldn't be called profitable. A fair verdict needs data spanning 2008 and/or
  2020–2022.

## Honest verdict

The code is correct and bug-free (45 passing tests, including the no-lookahead
guard). The **strategy**, as configured, does not have a tradeable edge on this
data. Do **not** run it with real money on the strength of these results.

## Reproduce

```bash
pip install -e ".[live,dev]"
python examples/backtest_sp500.py /path/to/all_stocks_5yr.csv          # full basket
python examples/backtest_sp500.py /path/to/all_stocks_5yr.csv AAPL MSFT # specific names
```

## Legitimate next steps (no guarantees)

1. **Test across regimes**, including a bear market (2008, 2020, 2022). This is the
   environment the defensive logic is designed for.
2. **Let winners run.** The fixed 2R target caps the fat right tail that trend
   systems live on; a trailing stop instead of a fixed target is the textbook fix.
   (Hypothesis — must be validated walk-forward, not in-sample.)
3. **Walk-forward optimization**: tune on a rolling train window, measure on the
   next unseen window. Anything tuned and reported on the *same* data is overfit.
4. Treat it as a **risk overlay** (drawdown reduction), not a return-maximizer, and
   judge it on Sharpe/Calmar across a full cycle — not raw return in a bull market.
