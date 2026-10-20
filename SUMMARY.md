# Summary: GARCH and Machine Learning for Volatility Forecasting

**Ramith Vaid | NTU Singapore**

This project started from one question: does a GARCH(1,1) model, written from scratch, forecast volatility better than a machine learning model? The first version answered with one ticker and one test window. The later parts asked the same question on ten assets, with three non-overlapping windows each, several ML models, other horizons and a Value at Risk backtest, and the answer got more careful at each step. The detail is in `README.md`; `chart_summary.png` shows one result from each part.

## What holds up

- **GARCH is a sound baseline.** It beats a naive trailing-volatility forecast on all ten assets, and at 5, 10 and 21 days it is level with HAR, a standard regression benchmark.
- **Student-t innovations matter for tail risk.** They fit far better than normal innovations (AIC lower by about 320 on JPM and 190 on the S&P 500) and fix much of the 99% VaR under-coverage in the volatile 2017 to 2023 blocks. Pooled over 2,250 days, JPM's 99% VaR has 29 breaches against 22.5 expected with the t model, and 39 with the normal.
- **Leverage is real in US equities.** GJR-GARCH and EGARCH improve the fit a lot on the S&P 500, JPM and Apple, and find nothing in long Treasuries or EUR/USD.

## What did not hold up

- **"GARCH clearly beats ML."** Part 1's result came from one window and a forward target that was misaligned by 20 days. With the target corrected, GARCH's lead over Gradient Boosting shrinks (JPM 0.87 to 0.93 as a ratio of RMSEs, S&P 500 0.74 to 0.89). Over ten assets GARCH is ahead on 8, but only Bitcoin is separated statistically, and in the most recent block the two are level.
- **The ranking depends on the ML model.** A Ridge regression on richer features and a small GRU are about 6% below GARCH's error on six assets, HAR about 4% below, and Gradient Boosting gets worse as features are added.
- **Leverage models barely help forecasts.** Out of sample the improvement is small and not significant, and they do not fix the JPM 99% VaR shortfall.
- **Ensembles add little.** GARCH's errors and the ML models' are correlated 0.8 to 0.9. Averaging with the better ML models lowers the error, but about as much as those models do alone, and weights learned from the previous block lose to a plain 50/50 average.

## A practical reading

GARCH's multi-day forecasts are too spread out. Realized volatility moves about half as much as the forecasts say (a calibration slope of 0.50 to 0.61), and GARCH over-forecasts by 0.7 to 2.0 percentage points on average. HAR and the ML models avoid this by shrinking toward the mean. For a baseline, HAR is as good as GARCH and simpler; GARCH's advantage is that it also gives a one-day distribution for VaR.

## Limits

Ten assets, one data source and twelve years, three scored windows per asset and overlapping targets, so most of the comparisons between models are suggestive, not tested to a conventional standard. The ML models were not tuned. Two things were found and not explained: JPM's fitted GARCH persistence is much lower than its volatility data suggest, and GARCH's upward bias in multi-day forecasts has no identified cause.
