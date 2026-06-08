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
**every 3rd symbol alphabetically** (168 symbols, **3,423 trades**) — selection
made before looking at any returns.

## Results (default parameters, 65% threshold, fixed 2:1 target)

**Per-trade edge (pooled, 3,423 trades)**

| metric | value |
|---|---|
| win rate | 36.1% |
| profit factor | **1.11** |
| avg win / avg loss | +5.30% / −2.72% |
| expectancy / trade | **+0.18%** |
| avg holding | 11 bars |
| exits | 2173 stops, 1233 targets, 10 signal, 7 eod |

**Return vs. buy-and-hold (2013–2018 bull market)**

| | strategy | buy & hold |
|---|---|---|
| mean return | +2.8% | **+68.7%** |
| median return | **−0.4%** | +42.3% |
| % of symbols strategy beat B&H | **16%** | — |

**Risk-adjusted**

| | strategy | buy & hold |
|---|---|---|
| mean max drawdown | **13.6%** | 37.0% |
| mean Sharpe | 0.04 | **0.51** |
| time in market | 19% | 100% |

Single-name spot checks: AAPL +12% vs +144%, MSFT **−14%** vs +169%,
NVDA +59% vs +1576%.

## Tested improvement: trailing stop — **rejected**

Trend systems are supposed to make money by letting a few winners run, so the
fixed 2:1 target (which clips the right tail) was the obvious suspect. I added a
trailing-stop exit (`exit_mode="trailing"`) and re-ran the same 168 symbols. It
made things **worse**, not better:

| | fixed 2:1 target | trailing stop |
|---|---|---|
| profit factor | **1.11** | 1.04 |
| expectancy / trade | **+0.18%** | +0.03% |
| win rate | 36.1% | 28.9% |
| mean / median return | +2.8% / −0.4% | +0.6% / −1.0% |
| mean Sharpe | 0.04 | **−0.02** |

Holding longer just gave back more open profit on pullbacks and dropped the win
rate. An honest negative result: the "obvious fix" didn't fix it.

## What this means

- **Profit factor 1.11 / expectancy +0.18%/trade** is a real-but-tiny edge. It is
  *inside the noise* — thin enough that real-world frictions (partial fills,
  spreads on less-liquid names, taxes) could erase it.
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
   environment the defensive logic is designed for, and this 2013–2018 sample has
   no crash for it to avoid.
2. ~~Let winners run (trailing stop).~~ **Tested above — it made things worse.**
3. **Walk-forward optimization**: tune on a rolling train window, measure on the
   next unseen window. Anything tuned and reported on the *same* data is overfit.
4. Treat it as a **risk overlay** (drawdown reduction), not a return-maximizer, and
   judge it on Sharpe/Calmar across a full cycle — not raw return in a bull market.

> Reproduce the comparison: `run_backtest(sym, bars, exit_mode="trailing")` vs the
> default `exit_mode="target"`.
