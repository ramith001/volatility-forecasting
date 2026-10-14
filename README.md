# Volatility Forecasting: GARCH vs Machine Learning
### Built on Real Market Data
**Ramith Vaid | NTU Singapore | Data Science & Artificial Intelligence**

---

## Overview

Volatility forecasting sits at the core of risk management, options pricing, and portfolio construction at any bank or trading desk. This project builds and rigorously compares two approaches to forecasting volatility on real market data:

1. **GARCH(1,1)**, the industry standard econometric model for volatility clustering, implemented from scratch via Maximum Likelihood Estimation, validated against simulated data with known parameters before being applied to real prices
2. **A machine learning alternative** (Gradient Boosting), using lagged realized volatility and return features

Both forecasts are evaluated out-of-sample against actual realized volatility, and the GARCH model is used to compute a Value at Risk (VaR) estimate, properly backtested for breach frequency, the same basic validation real risk teams run on their own models.

**Data:** JPMorgan Chase (JPM) and the S&P 500 (^GSPC), 2014-2024, via Yahoo Finance.

---

## A Methodological Issue Worth Documenting

An early version of this comparison produced a result that looked too good to be true: the ML model beat GARCH by close to 10x on RMSE. Rather than report that number, it was investigated properly.

The cause was a subtle but real data leakage problem. The original evaluation target, tomorrow's 21-day trailing realized volatility, overlapped with the ML model's own lagged input feature (yesterday's 21-day trailing realized volatility) on 20 of their 21 underlying days. The two values were correlated at roughly 0.97 purely from that overlap, regardless of any genuine predictive relationship, meaning the ML model's apparent accuracy was largely an artifact of being handed a near-copy of the answer.

The fix: both models were re-evaluated against a genuinely forward-looking, non-overlapping target, realized volatility over the next 21 days starting tomorrow, computed only from data not yet seen by either model. GARCH was extended to forecast its own 21-day-ahead average variance (the standard GARCH variance term structure) rather than a single one-day-ahead number, so both models are judged on the same, fairly defined task. After the fix, GARCH and ML RMSE came out close (9.86 vs 8.46), a believable, modest gap rather than an implausible one.

A second, smaller issue surfaced once the first was fixed: the Value at Risk section was then using the smoothed 21-day average forecast for what should be a sharp, single-day risk estimate. This was fixed by having the rolling GARCH function output both a true 1-day-ahead forecast and the 21-day average forecast from the same fitted model, since both come from the same parameters at effectively no extra computational cost, and using the correct one for each purpose.

This is included here deliberately. Catching a result that looks suspiciously good, diagnosing why, and fixing the underlying methodology rather than reporting a misleading number is a real and recurring part of applied data science work.

---

## GARCH(1,1) Implemented from Scratch

Rather than rely on a packaged library, GARCH(1,1) is implemented directly via Maximum Likelihood Estimation using `scipy.optimize`. This was a practical necessity as much as a design choice: the standard `arch` package has no prebuilt wheel for Windows on ARM, and compiling it from source requires a full Visual Studio build toolchain. Implementing the model directly avoided that dependency entirely.

The implementation is validated before being trusted on real data: a return series is simulated from known GARCH(1,1) parameters (omega=0.05, alpha=0.10, beta=0.85), and the fitting procedure is confirmed to recover those parameters closely (fitted: omega=0.0871, alpha=0.1006, beta=0.8024) before being applied to JPMorgan Chase and S&P 500 price data.

---

## Methodology

**Returns and realized volatility.** Daily log returns are computed from closing prices. Realized volatility is the rolling 21-day standard deviation of returns, annualized.

**GARCH(1,1) fitting.** Fit via MLE, refitting on a rolling basis across the out-of-sample test window. The fitted parameters feed a variance term structure: a one-step-ahead forecast using the last known return and variance, then each subsequent day forecast using the property that expected future squared returns equal the forecast variance itself, the standard technique for multi-day-ahead GARCH forecasts.

**Machine learning model.** Gradient Boosting trained on lagged realized volatility, lagged returns, and short-window return statistics, predicting a non-overlapping, forward-looking 21-day realized volatility target.

**Value at Risk backtesting.** A 95% one-day VaR threshold is computed from the GARCH model's true one-day-ahead forecast, then backtested: if well calibrated, actual daily losses should exceed the threshold roughly 5% of the time.

---

## Key Findings

**GARCH and ML perform comparably once evaluated fairly.** GARCH RMSE of 9.86 percentage points versus ML RMSE of 8.46, a modest, believable gap consistent with general findings in volatility forecasting research, rather than the artifact-driven 10x gap seen before the leakage was caught and fixed.

**The VaR model is conservative.** Backtested breach rate of 2.87% against an expected 5% for a 95% VaR threshold, meaning the model under-predicts risk slightly, fewer large losses occurred than the threshold implied should happen. This is a common, real characteristic of GARCH-based VaR models in practice, and is reported honestly rather than adjusted to look more precisely calibrated than it is.

---

## Known Limitations

**Single ticker depth.** The full GARCH-vs-ML comparison and VaR backtest were run in depth on JPMorgan Chase specifically; the S&P 500 data was collected and processed but not carried through the full rolling forecast pipeline. Extending the full comparison to the index, and to other individual stocks, is a natural next step.

**Normal innovations assumption.** The GARCH likelihood assumes normally distributed return shocks. Real financial returns are well known to have fatter tails than the normal distribution; a Student's t-distribution innovation assumption is a standard, fairly straightforward extension that was not implemented here.

**VaR backtesting on a single test window.** The Kupiec-style breach rate check was run on one 209-day out-of-sample period. A more rigorous backtest would repeat this across multiple, non-overlapping historical windows.

---

## Part 1: Extended Date Range and Completed S&P 500 Comparison

The original run covered 2014 to 2024 and left the S&P 500 side of the comparison for future work. This extends both tickers through October 2026 and completes that comparison.

With two more years of data, GARCH now outperforms the ML model on both tickers, a reversal from the original finding (GARCH RMSE 9.86 vs ML RMSE 8.46 at the time):

| Ticker | GARCH RMSE | ML RMSE |
|---|---|---|
| JPM | 4.17 | 4.82 |
| S&P 500 | 3.47 | 4.70 |

Error on JPM roughly halved, from the original ~9-10 RMSE range down to ~4-5. The 250-day test window now falls in a comparatively calm stretch (October 2025 to October 2026), which plausibly explains part of the drop rather than a change in either model's underlying accuracy -- a reminder that a single out-of-sample window reflects the period it happens to cover.

VaR calibration also moved closer to target: 10 breaches in 209 test days (4.78%), close to the 5% nominal rate, versus 2.87% (noticeably over-conservative) in the original run.

The leakage check still holds at the extended range: lag-1 realized volatility correlates with the forward realized-vol target at 0.26 (JPM) and 0.22 (S&P 500), both well below the ~0.97 threshold that would indicate a leak.

---

## Files

- `volatility_forecasting.ipynb` -- the original JPM-only, 2014-2024 notebook, left untouched.
- `garch_utils.py` -- the GARCH(1,1), realized-volatility and ML-feature functions factored out of the original notebook (bodies unchanged), plus `run_full_pipeline()` to run the whole per-ticker comparison on any ticker or date range.
- `part1_extended_and_sp500.ipynb` -- Part 1: both tickers through October 2026, using `garch_utils.py`.
- `chart_extended_jpm_vs_sp500.png` -- GARCH vs. ML RMSE, JPM vs. S&P 500, extended period.

---

## Stack

- **Language:** Python 3.13
- **Libraries:** pandas, numpy, scipy, scikit-learn, yfinance, matplotlib
- **Data source:** Yahoo Finance (free, no API key required)
- **GARCH implementation:** custom, via scipy.optimize, no external GARCH package dependency

---

*Built by Ramith Vaid | NTU Singapore | ramithvaid@gmail.com*
