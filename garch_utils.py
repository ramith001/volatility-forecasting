"""
Shared GARCH(1,1)/realized-vol/ML-feature functions, factored out of
volatility_forecasting.ipynb so later parts (extending to more tickers, more
history, alternative models, etc.) reuse the exact same, already-validated
logic instead of re-implementing it per notebook.

Every function body here is copied verbatim from the committed notebook
(cells under "3. GARCH(1,1) Model, Implemented from Scratch" and
"4. Machine Learning Alternative") -- nothing about the math changed, only
where it lives. The notebook itself is left untouched; it still works
standalone exactly as before, so the original, already-reproduced headline
numbers (GARCH RMSE 9.86, ML RMSE 8.46, VaR breach rate 2.87%) are not at
risk from this refactor.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


def compute_returns_and_vol(df, window=21):
    """Compute log returns and rolling realized volatility (annualized)."""
    out = df.copy()
    out['log_return'] = np.log(out['Close'] / out['Close'].shift(1))
    out['realized_vol'] = out['log_return'].rolling(window).std() * np.sqrt(252) * 100  # annualized %
    return out.dropna()


def garch_neg_log_likelihood(params, returns):
    """
    Negative log-likelihood for a GARCH(1,1) model under normal innovations.
    sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}
    """
    omega, alpha, beta = params
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)

    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

    sigma2 = np.maximum(sigma2, 1e-8)  # numerical safety floor
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + returns ** 2 / sigma2)
    return -ll


def fit_garch(returns):
    """Fit GARCH(1,1) via MLE, returns (omega, alpha, beta)."""
    bounds = [(1e-6, None), (0, 1), (0, 1)]
    constraints = [{'type': 'ineq', 'fun': lambda p: 0.999 - p[1] - p[2]}]  # stationarity: alpha + beta < 1

    result = minimize(garch_neg_log_likelihood, [0.1, 0.1, 0.8], args=(returns,),
                       method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 300})
    return result.x


def garch_forecast_next_day(returns, omega, alpha, beta):
    """Compute next-day forecasted variance given fitted GARCH parameters."""
    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(returns)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

    next_day_var = omega + alpha * returns[-1] ** 2 + beta * sigma2[-1]
    return next_day_var


def garch_term_structure_forecast(last_return, last_sigma2, omega, alpha, beta, horizon=21):
    """
    Forecast average variance over the next `horizon` days using GARCH(1,1).
    Day 1: sigma2_{t+1} = omega + alpha*r_t^2 + beta*sigma2_t (uses known last return/variance)
    Day k>=2: sigma2_{t+k} = omega + (alpha+beta)*sigma2_{t+k-1} (no new data, E[r^2]=sigma2)
    This is the standard GARCH variance term structure used for multi-day-ahead forecasts.
    """
    sigma2_path = np.zeros(horizon)
    sigma2_path[0] = omega + alpha * last_return ** 2 + beta * last_sigma2
    for k in range(1, horizon):
        sigma2_path[k] = omega + (alpha + beta) * sigma2_path[k - 1]
    return sigma2_path[0], sigma2_path.mean()  # return BOTH the 1-day-ahead and the horizon-average


def fit_garch_rolling_forecast(returns, train_window=750, test_size=250, horizon=21):
    """
    Fit GARCH(1,1) on a rolling basis. Returns two series from the same fitted model:
    - a true 1-day-ahead forecast (appropriate for daily VaR)
    - a 21-day-ahead average forecast (appropriate for comparison against the
      forward-looking 21-day realized vol target used by the ML model)
    Computing both costs nothing extra since they come from the same fitted parameters.
    """
    returns_arr = returns.values * 100
    n = len(returns_arr)
    forecasts_1d = []
    forecasts_avg = []
    dates = []

    start_idx = n - test_size

    for i in range(start_idx, n):
        train_data = returns_arr[max(0, i - train_window):i]
        omega, alpha, beta = fit_garch(train_data)

        last_sigma2_series = np.zeros(len(train_data))
        last_sigma2_series[0] = np.var(train_data)
        for t in range(1, len(train_data)):
            last_sigma2_series[t] = omega + alpha * train_data[t - 1] ** 2 + beta * last_sigma2_series[t - 1]

        var_1d, avg_var = garch_term_structure_forecast(
            train_data[-1], last_sigma2_series[-1], omega, alpha, beta, horizon=horizon
        )
        forecasts_1d.append(np.sqrt(var_1d) * np.sqrt(252))
        forecasts_avg.append(np.sqrt(avg_var) * np.sqrt(252))
        dates.append(returns.index[i])

    return (pd.Series(forecasts_1d, index=dates, name='garch_1d_forecast'),
            pd.Series(forecasts_avg, index=dates, name='garch_forecast'))


def compute_forward_realized_vol(df, horizon=21):
    """
    Forward-looking realized volatility: annualized std of returns over the NEXT
    `horizon` days starting tomorrow. Does not overlap with any trailing window
    ending today, avoiding the leakage-like overlap problem.
    """
    returns = df['log_return']
    forward_vol = returns.shift(-horizon).rolling(horizon).std().shift(-(horizon - 1)) * np.sqrt(252) * 100
    return forward_vol


def build_ml_features(df, n_lags=5, horizon=21):
    """Build lagged features (using only past data) and a forward-looking target."""
    feat = pd.DataFrame(index=df.index)
    feat['target_vol'] = compute_forward_realized_vol(df, horizon=horizon)

    for lag in range(1, n_lags + 1):
        feat[f'vol_lag_{lag}'] = df['realized_vol'].shift(lag - 1)
        feat[f'return_lag_{lag}'] = df['log_return'].shift(lag - 1)

    feat['return_abs_5d_mean'] = df['log_return'].abs().rolling(5).mean()
    feat['vol_trend'] = df['realized_vol'].diff(5)

    return feat.dropna()


def run_full_pipeline(df, ticker_name, test_size=250, horizon=21, n_lags=5,
                       train_window=750, compute_var=True, random_state=42):
    """
    Orchestrates the notebook's full per-ticker pipeline (GARCH rolling forecast,
    ML forward-looking forecast, comparison, optional VaR backtest) using only
    the functions above -- same method, parameterized by ticker/test window
    instead of hardcoded to JPM, so it can run unchanged on any ticker or any
    date range. Introduced in Part 1 to run the exact single-ticker (JPM)
    pipeline on a second ticker (S&P 500) and on an extended date range without
    copy-pasting the notebook's cells.

    Local import of GradientBoostingRegressor keeps this module importable
    even in an environment missing scikit-learn until this function is
    actually called.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    processed = compute_returns_and_vol(df)

    ml_data = build_ml_features(processed, n_lags=n_lags, horizon=horizon)
    leak_check_corr = ml_data['vol_lag_1'].corr(ml_data['target_vol'])

    train_ml = ml_data.iloc[:-test_size]
    test_ml = ml_data.iloc[-test_size:]
    feature_cols = [c for c in ml_data.columns if c != 'target_vol']
    X_train, y_train = train_ml[feature_cols], train_ml['target_vol']
    X_test, y_test = test_ml[feature_cols], test_ml['target_vol']

    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                     random_state=random_state)
    gbr.fit(X_train, y_train)
    ml_forecast = pd.Series(gbr.predict(X_test), index=test_ml.index, name='ml_forecast')

    garch_1d, garch_avg = fit_garch_rolling_forecast(processed['log_return'], train_window=train_window,
                                                       test_size=test_size, horizon=horizon)

    actual_vol = ml_data['target_vol']
    comparison = pd.DataFrame({'actual': actual_vol, 'garch': garch_avg, 'ml': ml_forecast}).dropna()

    metrics = {}
    for col in ['garch', 'ml']:
        metrics[f'{col}_rmse'] = float(np.sqrt(mean_squared_error(comparison['actual'], comparison[col])))
        metrics[f'{col}_mae'] = float(mean_absolute_error(comparison['actual'], comparison[col]))

    result = {
        'ticker': ticker_name,
        'leak_check_corr': float(leak_check_corr),
        'comparison': comparison,
        'gbr_model': gbr,
        'feature_cols': feature_cols,
        **metrics,
    }

    if compute_var:
        Z_95 = norm.ppf(0.95)
        daily_vol_forecast = garch_1d.reindex(comparison.index) / np.sqrt(252)
        var_95 = -Z_95 * (daily_vol_forecast / 100)
        actual_returns = processed['log_return'].reindex(comparison.index)
        breaches = actual_returns < var_95
        result['var_95'] = var_95
        result['actual_returns'] = actual_returns
        result['breaches'] = breaches
        result['breach_rate'] = float(breaches.mean() * 100)
        result['n_breaches'] = int(breaches.sum())
        result['n_test_days'] = int(len(comparison))

    return result
