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
from scipy.signal import lfilter
from scipy.special import gammaln
from scipy.stats import chi2, norm
from scipy.stats import t as student_t


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


# ---------------------------------------------------------------------------
# Part 2: Student-t innovations and VaR coverage tests
#
# Everything above is the Part 1 module and is unchanged. Everything below is
# new in Part 2. The conditional-variance recursion is the same one
# garch_neg_log_likelihood uses, computed with scipy.signal.lfilter instead of
# a Python loop (it is a first-order linear filter), because the t model has a
# 4th parameter and is refit hundreds of times in the rolling backtest. The
# notebook checks that the vectorised path matches the loop.
# ---------------------------------------------------------------------------
def garch_variance_path(returns, omega, alpha, beta):
    """sigma2[t] for t = 0..n-1 with sigma2[0] = var(returns): the same recursion as
    garch_neg_log_likelihood, vectorised."""
    r = np.asarray(returns, dtype=float)
    s0 = np.var(r)
    x = omega + alpha * r[:-1] ** 2
    y, _ = lfilter([1.0], [1.0, -beta], x, zi=[beta * s0])
    return np.concatenate([[s0], y])


def garch_t_neg_log_likelihood(params, returns):
    """
    Negative log-likelihood for GARCH(1,1) with Student-t innovations, scaled to unit
    variance so that sigma2_t stays the conditional variance (same meaning as under
    normal innovations; only the tail shape changes):
        r_t = sigma_t * z_t,  z_t ~ t_nu / sqrt(nu / (nu - 2)),  nu > 2
    As nu -> infinity this converges to garch_neg_log_likelihood.
    """
    omega, alpha, beta, nu = params
    r = np.asarray(returns, dtype=float)
    sigma2 = np.maximum(garch_variance_path(r, omega, alpha, beta), 1e-8)
    const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    ll = len(r) * const - 0.5 * np.sum(np.log(sigma2)) \
        - (nu + 1) / 2 * np.sum(np.log1p(r ** 2 / ((nu - 2) * sigma2)))
    return -ll


NU_MIN, NU_MAX = 2.05, 200.0  # nu bounded away from 2 (infinite variance) and from the normal limit


def fit_garch_t(returns, start=None):
    """Fit GARCH(1,1)-t via MLE, returns (omega, alpha, beta, nu). `start` = (omega, alpha, beta)
    from the normal fit is used as the starting point when given (nu starts at 8)."""
    bounds = [(1e-6, None), (0, 1), (0, 1), (NU_MIN, NU_MAX)]
    constraints = [{'type': 'ineq', 'fun': lambda p: 0.999 - p[1] - p[2]}]  # stationarity
    x0 = [*(start if start is not None else (0.1, 0.1, 0.8)), 8.0]
    result = minimize(garch_t_neg_log_likelihood, x0, args=(returns,), method='SLSQP',
                       bounds=bounds, constraints=constraints, options={'maxiter': 300})
    return result.x


def standardized_residuals(returns, omega, alpha, beta):
    r = np.asarray(returns, dtype=float)
    return r / np.sqrt(np.maximum(garch_variance_path(r, omega, alpha, beta), 1e-8))


def log_density_normal(r, sigma2):
    return -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + r ** 2 / sigma2)


def log_density_t(r, sigma2, nu):
    const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    return const - 0.5 * np.log(sigma2) - (nu + 1) / 2 * np.log1p(r ** 2 / ((nu - 2) * sigma2))


def var_threshold(sigma, level, dist, nu=None):
    """One-day return threshold (negative number, same units as sigma) such that a return below it is a
    VaR breach at `level` (0.95 -> 5% tail). For the t, the quantile is rescaled to unit variance."""
    if dist == 'normal':
        return -norm.ppf(level) * sigma
    return -student_t.ppf(level, nu) * np.sqrt((nu - 2) / nu) * sigma


def kupiec_pof(n_breaches, n, p):
    """Kupiec proportion-of-failures LR test of the breach rate against the nominal rate p. -> (LR, p-value)."""
    x, pi = n_breaches, n_breaches / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - pi) + (x * np.log(pi) if x > 0 else 0.0)
    lr = -2 * (ll0 - ll1)
    return lr, float(chi2.sf(lr, 1))


def christoffersen(hits, p):
    """Christoffersen (1998) tests on a 0/1 breach sequence: independence (are breaches clustered?) and
    conditional coverage (POF + independence). -> dict with LR and p-values. NaN when a transition state
    never occurs (too few breaches to test)."""
    h = np.asarray(hits, dtype=int)
    n00 = int(np.sum((h[:-1] == 0) & (h[1:] == 0)))
    n01 = int(np.sum((h[:-1] == 0) & (h[1:] == 1)))
    n10 = int(np.sum((h[:-1] == 1) & (h[1:] == 0)))
    n11 = int(np.sum((h[:-1] == 1) & (h[1:] == 1)))
    x, n = int(h.sum()), len(h)
    if n01 + n11 == 0 or n00 + n01 == 0 or n10 + n11 == 0:
        return {'lr_ind': np.nan, 'p_ind': np.nan, 'lr_cc': np.nan, 'p_cc': np.nan}

    def term(k, m):
        return k * np.log(m) if k > 0 else 0.0
    pi01, pi11, pi = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_ind1 = term(n00, 1 - pi01) + term(n01, pi01) + term(n10, 1 - pi11) + term(n11, pi11)
    ll_ind0 = term(n00 + n10, 1 - pi) + term(n01 + n11, pi)
    lr_ind = -2 * (ll_ind0 - ll_ind1)
    lr_pof, _ = kupiec_pof(x, n, p)
    lr_cc = lr_pof + lr_ind
    return {'lr_ind': lr_ind, 'p_ind': float(chi2.sf(lr_ind, 1)),
            'lr_cc': lr_cc, 'p_cc': float(chi2.sf(lr_cc, 2))}


def rolling_normal_vs_t(returns, train_window=750, test_size=750):
    """
    For each of the last `test_size` days, refit BOTH GARCH(1,1)-normal (fit_garch, as in Part 1) and
    GARCH(1,1)-t on the previous `train_window` days and record each model's one-day-ahead conditional
    variance (daily, in percent^2), the fitted nu, and the realized return that day. The t fit starts
    from the normal fit's parameters. Returns a DataFrame indexed by date.
    """
    r_all = returns.values * 100
    n = len(r_all)
    rows = []
    for i in range(n - test_size, n):
        train = r_all[max(0, i - train_window):i]
        w, a, b = fit_garch(train)
        s2_n = w + a * train[-1] ** 2 + b * garch_variance_path(train, w, a, b)[-1]
        wt, at, bt, nu = fit_garch_t(train, start=(w, a, b))
        s2_t = wt + at * train[-1] ** 2 + bt * garch_variance_path(train, wt, at, bt)[-1]
        rows.append({'date': returns.index[i], 'ret': r_all[i], 'sigma2_normal': s2_n,
                     'sigma2_t': s2_t, 'nu': nu})
    return pd.DataFrame(rows).set_index('date')
