"""Quality-adjusted (hedonic time-dummy) price index and forecasting of the index.

The index is the exponential of the month coefficients of a regression of log1p(target) on month
dummies plus structural controls, rebased to 100 at the first month (2022-01). Forecasts are made on
the log of the index. Everything that is evaluated before 2025 uses only months up to the forecast
origin (index vintages), so no future month enters a fit.
"""
import itertools
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.linalg import qr
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from . import config
from .features import build_features, build_target, select_track

MONTH_COLUMN = 'contract_month'
CONTROL_CATEGORICAL = ['district_name', 'building_type', 'contract_type', 'renewal_right_used']
CONTROL_NUMERIC = ['log_area', 'floor_filled', 'floor_missing', 'building_age']
SEASONAL_PERIOD = 12

FIRST_ORIGIN = 18   # months observed at the first forecast origin
ORIGIN_STEP = 3
MAX_HORIZON = 12
EVAL_HORIZONS = (1, 3, 6, 12)

CANDIDATES = ['naive', 'seasonal_naive', 'drift', 'linear_trend', 'ets_damped', 'arima', 'sarima', 'theta']
ALPHAS = {'80': 0.20, '95': 0.05}

ARIMA_GRID = [(p, 1, q) for p, q in itertools.product((0, 1, 2), (0, 1, 2))]
SARIMA_GRID = [(order, seasonal) for order in [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)]
               for seasonal in [(0, 1, 0, SEASONAL_PERIOD), (1, 1, 0, SEASONAL_PERIOD)]]


# ----------------------------------------------------------------------------------------------
# Step 1 - hedonic time-dummy index
# ----------------------------------------------------------------------------------------------
def independent_columns(frame, tol=1e-8):
    """Names of a maximal set of linearly independent columns (QR with pivoting on the Gram matrix)."""
    gram = frame.to_numpy().T @ frame.to_numpy()
    _, r, pivot = qr(gram, pivoting=True)
    diag = np.abs(np.diag(r))
    keep = sorted(pivot[diag > tol * diag.max()])
    return [frame.columns[i] for i in keep]


def hedonic_index(df, track, last_month=None):
    """Hedonic index of one track using the months up to `last_month` (inclusive).

    Returns a DataFrame indexed by month with `index` (100 at the first month), the 95% interval of
    the month coefficient (`index_low95`, `index_high95`), `log_coef` and `n_rows`.
    """
    d = select_track(df, track)
    if last_month is not None:
        d = d[d[MONTH_COLUMN] <= pd.Timestamp(last_month)]
    X = build_features(d, track)
    y = build_target(d, track).to_numpy()

    controls = pd.get_dummies(X[CONTROL_CATEGORICAL], drop_first=True, dtype=float)
    controls['log_area'] = X['log_area']
    controls['floor_filled'] = X['floor'].fillna(X['floor'].median())
    controls['floor_missing'] = X['floor_missing'].astype(float)
    controls['building_age'] = X['building_age']

    kept = independent_columns(controls.assign(const=1.0))
    dropped = [c for c in controls.columns if c not in kept]
    controls = controls[[c for c in controls.columns if c in kept]]   # redundant controls (e.g. floor_missing)

    months = pd.get_dummies(d[MONTH_COLUMN], dtype=float)
    months = months.reindex(sorted(months.columns), axis=1)
    month_labels = list(months.columns)
    month_dummies = months.iloc[:, 1:]          # first month is the reference (index = 100)

    A = pd.concat([controls, month_dummies], axis=1)
    A.insert(0, 'const', 1.0)
    fit = sm.OLS(y, A.to_numpy()).fit()

    coef = pd.Series(fit.params, index=A.columns)[month_dummies.columns]
    se = pd.Series(fit.bse, index=A.columns)[month_dummies.columns]
    out = pd.DataFrame(index=pd.DatetimeIndex(month_labels, name='month', freq='MS'))
    out['log_coef'] = 0.0
    out.loc[coef.index, 'log_coef'] = coef.to_numpy()
    lo = pd.Series(0.0, index=out.index)
    hi = pd.Series(0.0, index=out.index)
    lo.loc[coef.index] = (coef - 1.96 * se).to_numpy()
    hi.loc[coef.index] = (coef + 1.96 * se).to_numpy()
    out['index'] = 100 * np.exp(out['log_coef'])
    out['index_low95'] = 100 * np.exp(lo)
    out['index_high95'] = 100 * np.exp(hi)
    out['n_rows'] = d.groupby(MONTH_COLUMN).size().reindex(out.index).to_numpy()
    out.attrs['dropped_controls'] = dropped
    out.attrs['n_controls'] = int(controls.shape[1])
    return out


def log_index(index_table):
    """Log of the index (the series that is forecast), with a monthly DatetimeIndex."""
    y = np.log(index_table['index'].astype(float))
    y.index = pd.DatetimeIndex(y.index, freq='MS')
    return y


# ----------------------------------------------------------------------------------------------
# Step 2 - candidate forecasters (all work on the log index)
# ----------------------------------------------------------------------------------------------
def _normal_interval(mean, se, alpha):
    z = stats.norm.ppf(1 - alpha / 2)
    return mean - z * se, mean + z * se


def _future_index(y, h):
    return pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq='MS')


def _best_sarimax(y, grid, trend):
    best, best_aic = None, np.inf
    for order, seasonal in grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = SARIMAX(y, order=order, seasonal_order=seasonal, trend=trend,
                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        except Exception:
            continue
        if np.isfinite(res.aic) and res.aic < best_aic:
            best, best_aic = res, res.aic
    if best is None:
        raise RuntimeError('no SARIMAX candidate could be fit')
    return best


def forecast(name, y, h, alphas=None):
    """Forecast `h` months of the log index `y` with candidate `name`.

    Returns a DataFrame indexed by the future months with column `mean` and, if `alphas` is a dict
    such as {'80': 0.2, '95': 0.05}, the columns `lo80`, `hi80`, `lo95`, `hi95` (log scale).
    """
    y = y.astype(float)
    T = len(y)
    steps = np.arange(1, h + 1)
    idx = _future_index(y, h)
    values = y.to_numpy()
    frame = pd.DataFrame(index=idx)
    se = None   # analytic standard errors for the simple methods

    if name == 'naive':
        frame['mean'] = values[-1]
        se = np.std(np.diff(values), ddof=1) * np.sqrt(steps)
    elif name == 'seasonal_naive':
        frame['mean'] = values[T - SEASONAL_PERIOD + (steps - 1) % SEASONAL_PERIOD]
        resid = values[SEASONAL_PERIOD:] - values[:-SEASONAL_PERIOD]
        se = np.std(resid, ddof=1) * np.sqrt(np.ceil(steps / SEASONAL_PERIOD))
    elif name == 'drift':
        slope = (values[-1] - values[0]) / (T - 1)
        frame['mean'] = values[-1] + slope * steps
        se = np.std(np.diff(values) - slope, ddof=1) * np.sqrt(steps * (1 + steps / T))
    elif name == 'linear_trend':
        res = sm.OLS(values, sm.add_constant(np.arange(T, dtype=float))).fit()
        new = sm.add_constant(np.arange(T, T + h, dtype=float), has_constant='add')
        pred = res.get_prediction(new)
        frame['mean'] = pred.predicted_mean
        for key, a in (alphas or {}).items():
            ci = pred.conf_int(obs=True, alpha=a)
            frame[f'lo{key}'], frame[f'hi{key}'] = ci[:, 0], ci[:, 1]
    elif name == 'ets_damped':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = ETSModel(y, error='add', trend='add', damped_trend=True).fit(disp=False)
            pred = res.get_prediction(start=T, end=T + h - 1)
            frame['mean'] = pred.summary_frame(alpha=0.2)['mean'].to_numpy()
            for key, a in (alphas or {}).items():
                sf = pred.summary_frame(alpha=a)
                frame[f'lo{key}'], frame[f'hi{key}'] = sf['pi_lower'].to_numpy(), sf['pi_upper'].to_numpy()
    elif name in ('arima', 'sarima'):
        res = _best_sarimax(y, [(o, (0, 0, 0, 0)) for o in ARIMA_GRID] if name == 'arima' else SARIMA_GRID,
                            trend='c' if name == 'arima' else 'n')
        pred = res.get_forecast(h)
        frame['mean'] = pred.predicted_mean.to_numpy()
        for key, a in (alphas or {}).items():
            ci = pred.conf_int(alpha=a)
            frame[f'lo{key}'], frame[f'hi{key}'] = ci.iloc[:, 0].to_numpy(), ci.iloc[:, 1].to_numpy()
    elif name == 'theta':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = ThetaModel(y, period=SEASONAL_PERIOD, deseasonalize=True, use_test=True, method='additive').fit()
            frame['mean'] = np.asarray(res.forecast(h))
            for key, a in (alphas or {}).items():
                ci = res.prediction_intervals(h, alpha=a)
                frame[f'lo{key}'], frame[f'hi{key}'] = ci.iloc[:, 0].to_numpy(), ci.iloc[:, 1].to_numpy()
    else:
        raise ValueError(f'unknown candidate: {name}')

    if se is not None:
        for key, a in (alphas or {}).items():
            frame[f'lo{key}'], frame[f'hi{key}'] = _normal_interval(frame['mean'].to_numpy(), se, a)
    return frame


# ----------------------------------------------------------------------------------------------
# Rolling-origin evaluation
# ----------------------------------------------------------------------------------------------
def rolling_origins(n_months, first_origin=FIRST_ORIGIN, step=ORIGIN_STEP, max_horizon=MAX_HORIZON):
    """List of (n_observed, horizon): origin after `n_observed` months, forecasting `horizon` months.

    The horizon is cut so that every target month exists in the evaluation window."""
    return [(t, min(max_horizon, n_months - t)) for t in range(first_origin, n_months, step)]


def rolling_origin_forecasts(df, track, final_index, candidates=CANDIDATES, first_origin=FIRST_ORIGIN,
                             step=ORIGIN_STEP, max_horizon=MAX_HORIZON):
    """Forecasts of every candidate from index vintages fit on months up to each origin only.

    `final_index` is the TRAIN-only index (the truth for the target months). Returns a long
    DataFrame: origin, horizon, candidate, month, forecast (index points) and actual."""
    months = list(final_index.index)
    rows = []
    for t, h_max in rolling_origins(len(months), first_origin, step, max_horizon):
        vintage = hedonic_index(df, track, last_month=months[t - 1])
        assert vintage.index.max() == months[t - 1], 'vintage includes a month after the origin'
        y = log_index(vintage)
        for name in candidates:
            try:
                fc = forecast(name, y, h_max)['mean'].to_numpy()
            except Exception:
                fc = np.full(h_max, np.nan)
            for h in range(1, h_max + 1):
                month = months[t - 1 + h]
                rows.append({'origin': months[t - 1], 'n_observed': t, 'horizon': h, 'candidate': name,
                             'month': month, 'forecast': float(np.exp(fc[h - 1])),
                             'actual': float(final_index.loc[month, 'index'])})
    return pd.DataFrame(rows)


def error_table(forecasts, horizons=EVAL_HORIZONS):
    """MAE (index points) and MAPE (%) of each candidate at the given horizons, and their mean."""
    f = forecasts.dropna(subset=['forecast']).copy()
    f['abs_error'] = (f['forecast'] - f['actual']).abs()
    f['ape'] = f['abs_error'] / f['actual'] * 100
    sel = f[f['horizon'].isin(horizons)]
    mae = sel.pivot_table(index='candidate', columns='horizon', values='abs_error', aggfunc='mean')
    mape = sel.pivot_table(index='candidate', columns='horizon', values='ape', aggfunc='mean')
    n = sel.pivot_table(index='candidate', columns='horizon', values='ape', aggfunc='count')
    table = pd.concat({'MAE': mae, 'MAPE': mape, 'n_evals': n}, axis=1)
    table[('MAPE', 'mean_h')] = mape.mean(axis=1)
    return table


def select_candidate(table):
    """Lowest mean of the MAPE at the evaluated horizons (equal weights)."""
    return str(table[('MAPE', 'mean_h')].idxmin())


# ----------------------------------------------------------------------------------------------
# Diagnostics and persistence
# ----------------------------------------------------------------------------------------------
def adf_report(y, regression='c'):
    """Augmented Dickey-Fuller test (AIC lag selection) of a series."""
    res = adfuller(np.asarray(y, dtype=float), regression=regression, autolag='AIC', result_object=True)
    return {'adf_stat': float(res.statistic), 'p_value': float(res.pvalue), 'lags_used': int(res.lags),
            'n_obs': int(res.nobs), 'critical_5pct': float(res.critical_values['5%'])}


def _jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        return {str(k): _jsonable(v) for k, v in obj.to_dict(orient='index').items()}
    if isinstance(obj, pd.Series):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime('%Y-%m')
    return obj


def save_ts_report(track, report, reports_dir=None):
    """Write reports/metrics/ts_index_{track}.json (this module's own file; overwritten on rerun)."""
    base = config.METRICS_DIR if reports_dir is None else reports_dir
    base.mkdir(parents=True, exist_ok=True)
    path = base / f'ts_index_{track}.json'
    path.write_text(json.dumps(_jsonable({'track': track, 'random_seed': config.RANDOM_SEED, **report}), indent=1),
                    encoding='utf-8')
    return path
