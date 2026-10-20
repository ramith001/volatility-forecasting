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


# ── Part 3: multi-window VaR backtest ────────────────────────────────────────────────────────

def block_slices(n_days, block_len=750, n_blocks=3):
    """Non-overlapping blocks of `block_len` days, aligned to the END of a series of `n_days`, oldest first.
    Returns {label: slice}. Aligning to the end makes the last block identical to Part 2's 750-day window."""
    out = {}
    for k in range(n_blocks):
        stop = n_days - (n_blocks - 1 - k) * block_len
        out[f'block {k + 1}'] = slice(stop - block_len, stop)
    return out


def var_hits(d, level, dist):
    """0/1 VaR breach series from a rolling_normal_vs_t frame, for dist 'normal' or 't'."""
    if dist == 'normal':
        thr = var_threshold(np.sqrt(d['sigma2_normal']), level, 'normal')
    else:
        thr = var_threshold(np.sqrt(d['sigma2_t']), level, 't', nu=d['nu'])
    return (d['ret'] < thr).values.astype(int)


def coverage_row(hits, p):
    """Breach count, rate and Kupiec / Christoffersen p-values for one 0/1 breach sequence."""
    x, n = int(hits.sum()), len(hits)
    _, p_pof = kupiec_pof(x, n, p)
    c = christoffersen(hits, p)
    return {'n_days': n, 'breaches': x, 'expected': round(p * n, 1), 'breach_rate_%': 100 * x / n,
            'nominal_%': 100 * p, 'kupiec_p': p_pof, 'indep_p': c['p_ind'], 'cond_cov_p': c['p_cc']}


# ---------------------------------------------------------------------------
# Part 4: aligned forward target, multi-block GARCH-vs-ML evaluation, Diebold-Mariano test
#
# Everything above is unchanged. compute_forward_realized_vol (Part 1) returns the realized volatility
# over days t+21 .. t+41, not t+1 .. t+21 as its docstring says: it applies
# rolling(horizon).std() (which already gives the std of the NEXT `horizon` days after the shift(-horizon))
# and then shifts by another horizon-1 days. Part 1's GARCH-vs-ML RMSEs were scored against that
# target. The functions below use the target that matches the docstring and the GARCH forecast, and are
# what Parts 4-8 use. The original function is left as it was so Part 1 still reproduces.
# ---------------------------------------------------------------------------
def compute_forward_realized_vol_aligned(df, horizon=21):
    """Annualized std (in %) of the daily log returns over days t+1 .. t+horizon, known at the close of day t."""
    return df['log_return'].rolling(horizon).std().shift(-horizon) * np.sqrt(252) * 100


def build_ml_features_aligned(df, n_lags=5, horizon=21):
    """Same features as build_ml_features (all use data up to and including day t), with the aligned target."""
    feat = pd.DataFrame(index=df.index)
    feat['target_vol'] = compute_forward_realized_vol_aligned(df, horizon=horizon)
    for lag in range(1, n_lags + 1):
        feat[f'vol_lag_{lag}'] = df['realized_vol'].shift(lag - 1)
        feat[f'return_lag_{lag}'] = df['log_return'].shift(lag - 1)
    feat['return_abs_5d_mean'] = df['log_return'].abs().rolling(5).mean()
    feat['vol_trend'] = df['realized_vol'].diff(5)
    return feat.dropna()


def eval_positions(n, n_eval, max_horizon=21):
    """Positions (row numbers) of the last `n_eval` days that have a complete forward window of max_horizon days."""
    return np.arange(n - max_horizon - n_eval, n - max_horizon)


def rolling_garch_term_structure(returns, positions, train_window=750, horizons=(21,), fitter=None,
                                  term_structure=None, nll=None):
    """
    For each row number in `positions`, fit the model on the `train_window` returns ending at and INCLUDING that
    day (information known at that day's close) and forecast the average variance over the next h days, for each
    h in `horizons`. Returns a DataFrame indexed by date with annualized volatility forecasts (%) in columns
    f'h{h}' and the fitted parameters in 'p0', 'p1', ...

    fitter(train_pct) -> parameter vector; term_structure(params, train_pct, horizon) -> mean variance over the next
    `horizon` days (percent^2). The defaults are the repo's GARCH(1,1): fit_garch and garch_term_structure_forecast.

    Failed fits. If `nll` (nll(params, train_pct)) is given, each fit is checked against the constant-variance
    model, which every model here nests: a maximum-likelihood fit cannot have a higher negative log-likelihood than
    variance = sample variance. A fit that does (or has non-finite parameters) is a failed optimization, and the
    previous day's parameters are used instead (the constant-variance parameters if there is no previous day). The
    number of such days is returned in the frame's `attrs['n_fit_failures']`.
    """
    fitter = fitter or fit_garch
    r_pct = returns.values * 100
    rows, last_good, failures = [], None, 0
    for i in positions:
        train = r_pct[max(0, i - train_window + 1):i + 1]
        params = fitter(train)
        if nll is not None:
            v = np.var(train)
            nll_const = 0.5 * len(train) * (np.log(2 * np.pi) + np.log(v) + 1.0)
            if not np.all(np.isfinite(params)) or nll(params, train) > nll_const + 1e-6:
                failures += 1
                params = last_good if last_good is not None else params
        last_good = params
        row = {'date': returns.index[i]}
        if term_structure is None:
            omega, alpha, beta = params[:3]
            last_s2 = _garch_filter(omega, alpha, beta, train)[-1]
        for h in horizons:
            if term_structure is None:
                _, avg_var = garch_term_structure_forecast(train[-1], last_s2, omega, alpha, beta, horizon=h)
            else:
                avg_var = term_structure(params, train, h)
            row[f'h{h}'] = np.sqrt(avg_var) * np.sqrt(252)
        for k, p in enumerate(params):
            row[f'p{k}'] = p
        rows.append(row)
    out = pd.DataFrame(rows).set_index('date')
    out.attrs['n_fit_failures'] = failures
    return out


def fit_garch_fast(returns):
    """Same objective as garch_neg_log_likelihood, with the variance recursion done by lfilter instead of a Python
    loop (it is a first-order linear filter). Same optimizer, bounds and start values as fit_garch."""
    bounds = [(1e-6, None), (0, 1), (0, 1)]
    constraints = [{'type': 'ineq', 'fun': lambda p: 0.999 - p[1] - p[2]}]

    def nll(params):
        return garch_neg_log_likelihood_fast(params, returns)
    return minimize(nll, [0.1, 0.1, 0.8], method='SLSQP', bounds=bounds, constraints=constraints,
                    options={'maxiter': 300}).x


def garch_neg_log_likelihood_fast(params, returns):
    omega, alpha, beta = params
    s2 = np.maximum(_garch_filter(omega, alpha, beta, returns), 1e-8)
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + returns ** 2 / s2)


def _garch_filter(omega, alpha, beta, returns):
    """sigma2[0] = var(returns); sigma2[t] = omega + alpha r[t-1]^2 + beta sigma2[t-1], via lfilter."""
    n = len(returns)
    drive = np.empty(n)
    drive[0] = np.var(returns)
    drive[1:] = omega + alpha * returns[:-1] ** 2
    # y[0] = drive[0]; y[t] = drive[t] + beta*y[t-1]
    return lfilter([1.0], [1.0, -beta], drive)


def dm_test(loss_a, loss_b, lag):
    """
    Diebold-Mariano test of equal expected loss, with a Newey-West (Bartlett) long-run variance at `lag` and the
    Harvey-Leybourne-Newbold small-sample correction. d = loss_a - loss_b: a NEGATIVE statistic means model A has
    the lower loss. `lag` should be at least horizon - 1 because the forward targets overlap. -> (statistic, p-value)
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lag + 1):
        lrv += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(lrv / n)
    h = lag + 1
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= corr
    from scipy.stats import t as _t
    return float(stat), float(2 * _t.sf(abs(stat), n - 1))


def prepare_vol_data(close, horizon=21, n_lags=5):
    """Daily log returns, 21-day realized vol and the aligned ML feature/target table for one price series."""
    frame = pd.DataFrame({'Close': pd.Series(close).astype(float)})
    processed = compute_returns_and_vol(frame)
    ml = build_ml_features_aligned(processed, n_lags=n_lags, horizon=horizon)
    return processed, ml


def make_blocks(n, n_blocks=3, block_len=250, horizon=21):
    """Non-overlapping blocks of `block_len` evaluation days aligned to the end of the usable data (the last
    `horizon` days have no complete forward window and are never scored). Returns {label: row numbers}."""
    pos = eval_positions(n, n_blocks * block_len, horizon)
    return {f'block {k + 1}': pos[k * block_len:(k + 1) * block_len] for k in range(n_blocks)}


def default_gbr(random_state=42):
    """The Gradient Boosting settings Part 1 used."""
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=random_state)


def ml_block_forecast(processed, ml, positions, embargo=22, model_factory=None, feature_cols=None):
    """
    Train one model on every ml row whose forward window is complete before the block starts (row number
    <= positions[0] - embargo; with a 21-day target and embargo 22 the last training target ends one day before the
    first scored day) and predict the block. embargo=0 reproduces Part 1's split, which trained on targets that reach
    into the test period. Returns (predictions Series indexed by date, fitted model).
    """
    model = (model_factory or default_gbr)()
    cols = feature_cols or [c for c in ml.columns if c != 'target_vol']
    row_pos = processed.index.get_indexer(ml.index)
    train = ml[row_pos <= positions[0] - embargo]
    model.fit(train[cols], train['target_vol'])
    dates = processed.index[positions]
    return pd.Series(model.predict(ml.loc[dates, cols]), index=dates, name='ml'), model


def score_forecasts(actual, forecast):
    """RMSE, MAE and mean error (forecast minus actual) of a volatility forecast, in annualized percentage points."""
    e = np.asarray(forecast, dtype=float) - np.asarray(actual, dtype=float)
    return {'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))), 'bias': float(np.mean(e))}


# ---------------------------------------------------------------------------
# Part 5: leverage effects. GJR-GARCH(1,1) and EGARCH(1,1), fit by maximum likelihood like the symmetric model.
#
#   GJR:    sigma2_t = omega + (alpha + gamma * 1[r_{t-1} < 0]) * r_{t-1}^2 + beta * sigma2_{t-1}
#           gamma > 0 means negative returns raise volatility more than positive ones of the same size.
#           Stationary when alpha + gamma/2 + beta < 1 (a negative return has probability 1/2 under symmetry).
#   EGARCH: ln sigma2_t = omega + beta * ln sigma2_{t-1} + alpha * (|z_{t-1}| - E|z|) + gamma * z_{t-1},
#           z = r / sigma. gamma < 0 means negative returns raise volatility more. No positivity constraints.
# All returns are in percent, as everywhere else in this module. The GJR recursion is a linear filter in sigma2
# (the drive depends only on past returns), so it uses lfilter like the symmetric model; the EGARCH recursion
# is not (z depends on sigma), so it is a Python loop.
# ---------------------------------------------------------------------------
def gjr_variance_path(returns, omega, alpha, gamma, beta):
    n = len(returns)
    drive = np.empty(n)
    drive[0] = np.var(returns)
    drive[1:] = omega + (alpha + gamma * (returns[:-1] < 0)) * returns[:-1] ** 2
    return lfilter([1.0], [1.0, -beta], drive)


def gjr_neg_log_likelihood(params, returns):
    omega, alpha, gamma, beta = params
    s2 = np.maximum(gjr_variance_path(returns, omega, alpha, gamma, beta), 1e-8)
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + returns ** 2 / s2)


def fit_gjr(returns):
    """GJR-GARCH(1,1), normal innovations. Returns (omega, alpha, gamma, beta)."""
    bounds = [(1e-6, None), (0, 1), (-1, 1), (0, 1)]
    cons = [{'type': 'ineq', 'fun': lambda p: 0.999 - p[1] - p[2] / 2 - p[3]},
            {'type': 'ineq', 'fun': lambda p: p[1] + p[2]}]
    best = None
    for start in ([0.05, 0.03, 0.08, 0.88], [0.1, 0.1, 0.0, 0.8]):
        res = minimize(gjr_neg_log_likelihood, start, args=(returns,), method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 300})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def gjr_t_neg_log_likelihood(params, returns):
    omega, alpha, gamma, beta, nu = params
    s2 = np.maximum(gjr_variance_path(returns, omega, alpha, gamma, beta), 1e-8)
    return -np.sum(log_density_t(returns, s2, nu))


def fit_gjr_t(returns, start=None):
    """GJR-GARCH(1,1) with unit-variance Student-t innovations. Returns (omega, alpha, gamma, beta, nu)."""
    s = list(start) if start is not None else list(fit_gjr(returns))
    bounds = [(1e-6, None), (0, 1), (-1, 1), (0, 1), (2.05, 200)]
    cons = [{'type': 'ineq', 'fun': lambda p: 0.999 - p[1] - p[2] / 2 - p[3]},
            {'type': 'ineq', 'fun': lambda p: p[1] + p[2]}]
    res = minimize(gjr_t_neg_log_likelihood, s + [8.0], args=(returns,), method='SLSQP', bounds=bounds,
                   constraints=cons, options={'maxiter': 400})
    return res.x


_EG_C = np.sqrt(2 / np.pi)  # E|z| for a standard normal


def egarch_logvar_path(returns, omega, alpha, gamma, beta):
    n = len(returns)
    lv = np.empty(n)
    lv[0] = np.log(np.var(returns))
    for t in range(1, n):
        z = returns[t - 1] * np.exp(-0.5 * lv[t - 1])
        lv[t] = omega + beta * lv[t - 1] + alpha * (abs(z) - _EG_C) + gamma * z
        if lv[t] > 20.0:
            lv[t] = 20.0
        elif lv[t] < -20.0:
            lv[t] = -20.0
    return lv


def egarch_neg_log_likelihood(params, returns):
    omega, alpha, gamma, beta = params
    lv = egarch_logvar_path(returns, omega, alpha, gamma, beta)
    return 0.5 * np.sum(np.log(2 * np.pi) + lv + returns ** 2 * np.exp(-lv))


def fit_egarch(returns):
    """EGARCH(1,1), normal innovations. Returns (omega, alpha, gamma, beta)."""
    bounds = [(-5, 5), (-1, 1.5), (-1, 1), (-0.999, 0.999)]
    best = None
    for start in ([0.02, 0.12, -0.06, 0.96], [0.0, 0.2, -0.1, 0.9]):
        res = minimize(egarch_neg_log_likelihood, start, args=(returns,), method='L-BFGS-B', bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def gjr_term_structure(params, train, horizon):
    """Mean variance (percent^2) over the next `horizon` days, from the state at the end of `train`."""
    omega, alpha, gamma, beta = params[:4]
    s2 = gjr_variance_path(train, omega, alpha, gamma, beta)[-1]
    s2_next = omega + (alpha + gamma * (train[-1] < 0)) * train[-1] ** 2 + beta * s2
    persistence = alpha + gamma / 2 + beta
    path = np.empty(horizon)
    path[0] = s2_next
    for k in range(1, horizon):
        path[k] = omega + persistence * path[k - 1]
    return path.mean()


def _egarch_mgf(b, alpha, gamma):
    """E[exp(b * (alpha * (|z| - E|z|) + gamma * z))] for z ~ N(0, 1)."""
    from scipy.stats import norm as _norm
    p, m = b * (alpha + gamma), b * (alpha - gamma)
    return np.exp(-b * alpha * _EG_C) * (np.exp(p * p / 2) * _norm.cdf(p) + np.exp(m * m / 2) * _norm.cdf(m))


def egarch_term_structure(params, train, horizon):
    """
    Mean EXPECTED variance (percent^2) over the next `horizon` days. ln sigma2 on day t+1 is known at t. For later days,
    E[sigma2_{t+k}] = exp(omega * sum_{i<k-1} beta^i + beta^(k-1) * ln sigma2_{t+1}) * prod_{m=0}^{k-2} E[exp(beta^m u)],
    u = alpha(|z| - E|z|) + gamma z with z standard normal, so the exponentiated shocks are averaged exactly,
    not just the log variance.
    """
    omega, alpha, gamma, beta = params[:4]
    lv = egarch_logvar_path(train, omega, alpha, gamma, beta)
    z = train[-1] * np.exp(-0.5 * lv[-1])
    l1 = np.clip(omega + beta * lv[-1] + alpha * (abs(z) - _EG_C) + gamma * z, -20.0, 20.0)
    out = np.empty(horizon)
    out[0] = np.exp(l1)
    prod = 1.0
    for k in range(2, horizon + 1):
        prod *= _egarch_mgf(beta ** (k - 2), alpha, gamma)
        const = omega * sum(beta ** i for i in range(k - 1))
        out[k - 1] = np.exp(const + beta ** (k - 1) * l1) * prod
    return out.mean()


def sigma2_next_gjr(params, train):
    omega, alpha, gamma, beta = params[:4]
    s2 = gjr_variance_path(train, omega, alpha, gamma, beta)[-1]
    return omega + (alpha + gamma * (train[-1] < 0)) * train[-1] ** 2 + beta * s2


def news_impact(shocks, model, params, sigma2_prev):
    """Next-day variance as a function of today's return shock (percent), holding yesterday's variance at sigma2_prev."""
    shocks = np.asarray(shocks, dtype=float)
    if model == 'garch':
        omega, alpha, beta = params[:3]
        return omega + alpha * shocks ** 2 + beta * sigma2_prev
    if model == 'gjr':
        omega, alpha, gamma, beta = params[:4]
        return omega + (alpha + gamma * (shocks < 0)) * shocks ** 2 + beta * sigma2_prev
    omega, alpha, gamma, beta = params[:4]
    z = shocks / np.sqrt(sigma2_prev)
    return np.exp(omega + beta * np.log(sigma2_prev) + alpha * (np.abs(z) - _EG_C) + gamma * z)


# ---------------------------------------------------------------------------
# Part 6: a richer feature set and a sequence model for the ML side
#
# Same leakage rules as before: every feature at day t uses data through day t only (checked by a truncation test
# in the notebook), and the target is the aligned forward volatility over days t+1 .. t+horizon.
# Feature sets, in the order they are added:
#   'base'  the 12 features Part 1 used (5 lagged realized vols, 5 lagged returns, 5-day mean |return|, vol trend)
#   'har'   realized volatility over 5, 21 and 63 days (the heterogeneous autoregressive model's three horizons)
#   'rich'  base + har + downside and upside semivolatility, high-low range volatility (needs OHLC), the largest
#           absolute return in 21 days and, when supplied, the VIX level and its 5-day change
# ---------------------------------------------------------------------------
def build_rich_features(ohlc, horizon=21, vix=None, n_lags=5):
    """
    ohlc: DataFrame with 'Open', 'High', 'Low', 'Close' indexed by date. vix: optional Series of the VIX close on the
    same dates (known at the close of day t). Returns a DataFrame of features plus 'target_vol', rows with any NaN dropped,
    and the underlying processed frame (log_return, realized_vol, ...).
    """
    frame = pd.DataFrame({'Close': ohlc['Close'].astype(float)})
    processed = compute_returns_and_vol(frame)
    r = processed['log_return']
    feat = pd.DataFrame(index=processed.index)
    feat['target_vol'] = compute_forward_realized_vol_aligned(processed, horizon=horizon)
    for lag in range(1, n_lags + 1):
        feat[f'vol_lag_{lag}'] = processed['realized_vol'].shift(lag - 1)
        feat[f'return_lag_{lag}'] = r.shift(lag - 1)
    feat['return_abs_5d_mean'] = r.abs().rolling(5).mean()
    feat['vol_trend'] = processed['realized_vol'].diff(5)
    ann = np.sqrt(252) * 100
    feat['rv5'] = r.rolling(5).std() * ann
    feat['rv21'] = r.rolling(21).std() * ann
    feat['rv63'] = r.rolling(63).std() * ann
    feat['semi_down21'] = np.sqrt((r.clip(upper=0) ** 2).rolling(21).mean()) * ann
    feat['semi_up21'] = np.sqrt((r.clip(lower=0) ** 2).rolling(21).mean()) * ann
    hl = np.log(ohlc['High'].astype(float) / ohlc['Low'].astype(float)).reindex(processed.index)
    feat['range_vol5'] = np.sqrt((hl ** 2).rolling(5).mean() / (4 * np.log(2))) * np.sqrt(252) * 100
    feat['range_vol21'] = np.sqrt((hl ** 2).rolling(21).mean() / (4 * np.log(2))) * np.sqrt(252) * 100
    feat['max_abs_ret21'] = r.abs().rolling(21).max() * 100
    if vix is not None:
        v = vix.reindex(processed.index).ffill()
        feat['vix'] = v
        feat['vix_chg5'] = v.diff(5)
    return feat.dropna(), processed


FEATURE_SETS = {
    'base': ['vol_lag_1', 'vol_lag_2', 'vol_lag_3', 'vol_lag_4', 'vol_lag_5', 'return_lag_1', 'return_lag_2',
             'return_lag_3', 'return_lag_4', 'return_lag_5', 'return_abs_5d_mean', 'vol_trend'],
    'har': ['rv5', 'rv21', 'rv63'],
}
FEATURE_SETS['rich'] = (FEATURE_SETS['base'] + ['rv5', 'rv21', 'rv63', 'semi_down21', 'semi_up21', 'range_vol5',
                                                 'range_vol21', 'max_abs_ret21'])
FEATURE_SETS['rich_vix'] = FEATURE_SETS['rich'] + ['vix', 'vix_chg5']


class NumpyGRU:
    """
    A single-layer GRU with a linear read-out of the last hidden state, trained by minibatch Adam with manual
    backpropagation through time. Written in numpy because PyTorch is not available for this Python build.
        z = sigmoid(Wz x + Uz h + bz)      r = sigmoid(Wr x + Ur h + br)
        n = tanh(Wn x + Un (r * h) + bn)   h' = (1 - z) * n + z * h
        y = Wo h_T + bo
    Inputs X: (B, T, D). Output: (B,). Loss: mean squared error.
    """
    def __init__(self, n_in, n_hidden=16, seed=0):
        rng = np.random.default_rng(seed)
        H, D = n_hidden, n_in
        s_in, s_h = 1 / np.sqrt(D), 1 / np.sqrt(H)
        self.H, self.D = H, D
        self.p = {
            'Wz': rng.normal(0, s_in, (D, H)), 'Uz': rng.normal(0, s_h, (H, H)), 'bz': np.zeros(H),
            'Wr': rng.normal(0, s_in, (D, H)), 'Ur': rng.normal(0, s_h, (H, H)), 'br': np.zeros(H),
            'Wn': rng.normal(0, s_in, (D, H)), 'Un': rng.normal(0, s_h, (H, H)), 'bn': np.zeros(H),
            'Wo': rng.normal(0, s_h, (H,)), 'bo': np.zeros(1),
        }

    @staticmethod
    def _sig(a):
        return 1 / (1 + np.exp(-np.clip(a, -30, 30)))

    def forward(self, X, cache=False):
        p, B, T = self.p, X.shape[0], X.shape[1]
        h = np.zeros((B, self.H))
        steps = []
        for t in range(T):
            x = X[:, t, :]
            z = self._sig(x @ p['Wz'] + h @ p['Uz'] + p['bz'])
            r = self._sig(x @ p['Wr'] + h @ p['Ur'] + p['br'])
            n = np.tanh(x @ p['Wn'] + (r * h) @ p['Un'] + p['bn'])
            h_new = (1 - z) * n + z * h
            if cache:
                steps.append((x, h, z, r, n))
            h = h_new
        y = h @ p['Wo'] + p['bo'][0]
        return (y, steps, h) if cache else y

    def loss_and_grads(self, X, y_true):
        p = self.p
        y, steps, h_T = self.forward(X, cache=True)
        B = X.shape[0]
        err = y - y_true
        loss = float(np.mean(err ** 2))
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dy = 2 * err / B
        g['Wo'] = h_T.T @ dy
        g['bo'] = np.array([dy.sum()])
        dh = np.outer(dy, p['Wo'])
        for t in range(len(steps) - 1, -1, -1):
            x, h_prev, z, r, n = steps[t]
            dn = dh * (1 - z)
            dz = dh * (h_prev - n)
            dh_prev = dh * z
            dan = dn * (1 - n ** 2)
            g['Wn'] += x.T @ dan
            g['bn'] += dan.sum(0)
            g['Un'] += (r * h_prev).T @ dan
            drh = dan @ p['Un'].T
            dr = drh * h_prev
            dh_prev += drh * r
            daz = dz * z * (1 - z)
            dar = dr * r * (1 - r)
            g['Wz'] += x.T @ daz
            g['bz'] += daz.sum(0)
            g['Uz'] += h_prev.T @ daz
            g['Wr'] += x.T @ dar
            g['br'] += dar.sum(0)
            g['Ur'] += h_prev.T @ dar
            dh_prev += daz @ p['Uz'].T + dar @ p['Ur'].T
            dh = dh_prev
        return loss, g

    def fit(self, X, y, X_val=None, y_val=None, epochs=40, batch=64, lr=3e-3, weight_decay=1e-4, patience=8, seed=0):
        rng = np.random.default_rng(seed)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v = {k: np.zeros_like(val) for k, val in self.p.items()}
        best, best_p, bad, step = np.inf, None, 0, 0
        n = len(X)
        for ep in range(epochs):
            order = rng.permutation(n)
            for i in range(0, n, batch):
                idx = order[i:i + batch]
                _, g = self.loss_and_grads(X[idx], y[idx])
                step += 1
                for k in self.p:
                    grad = np.clip(g[k], -5, 5) + weight_decay * self.p[k]
                    m[k] = 0.9 * m[k] + 0.1 * grad
                    v[k] = 0.999 * v[k] + 0.001 * grad ** 2
                    self.p[k] -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
            if X_val is not None:
                val = float(np.mean((self.forward(X_val) - y_val) ** 2))
                if val < best - 1e-6:
                    best, best_p, bad = val, {k: a.copy() for k, a in self.p.items()}, 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_p is not None:
            self.p = best_p
        return self

    def predict(self, X):
        return self.forward(X)


def make_sequences(channels, ml, processed, window=42):
    """
    channels: DataFrame of per-day inputs (aligned to `processed.index`). For each row of `ml` (a day t) returns the
    (n, window, D) array of the channels on days t-window+1 .. t. Only past and present days enter. Rows that do not
    have a full window are dropped; returns (X, kept_index).
    """
    row_pos = processed.index.get_indexer(ml.index)
    arr = channels.values
    keep = row_pos >= window - 1
    X = np.stack([arr[p - window + 1:p + 1] for p in row_pos[keep]])
    return X, ml.index[keep]
