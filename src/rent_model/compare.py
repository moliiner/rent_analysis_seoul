"""Model comparison: leaderboard, one-standard-error selection and extrapolation check.

Everything here reads the runs saved under reports/ (metrics JSON and prediction parquet files of
every model). Runs flagged `valid=False` (oracle diagnostics) never enter a ranking.
"""
import json
import re

import numpy as np
import pandas as pd

from . import config
from .metrics import paired_bootstrap_wape_diff, wape

METRIC_COLUMNS = ['wape_pct', 'mae', 'mdape_pct', 'median_bias_pct', 'aggregate_bias_pct', 'mape_dq_median_pct',
                  'mape_dq_max_pct', 'rmse_log', 'r2_log']
FAMILIES = ['baseline', 'linear', 'forest', 'boosting', 'unsupervised-assisted', 'pure unsupervised', 'time-series/hybrid']
SUPERVISED_FAMILIES = ['linear', 'forest', 'boosting']

# Ordinal complexity used only to break ties (higher = more complex). A judgment-based scale:
# constants and group medians < unsupervised medians < linear < trees < boosting < pipelines that
# combine several fitted components (cluster / index + supervised model).
_COMPLEXITY = {
    'baseline_train_median': 0, 'baseline_global_median': 0, 'baseline_district': 1, 'baseline_district_type': 2,
    'baseline_district_type_area': 3, 'baseline_drift': 3,
    'kmeans_median': 4, 'gmm_median': 5,
    'ols_static': 6, 'ridge_static': 6, 'lasso_static': 6, 'elasticnet_static': 6,
    'ridge_time': 7, 'lasso_time': 7, 'elasticnet_time': 7, 'ridge_time_rich': 8,
    'ridge_time_kmeans': 9, 'ridge_time_tier': 9, 'ridge_time_gmm': 9,
    'rf_static': 10, 'rf_time': 10, 'rf_detrended': 11,
    'hgb_static': 12, 'hgb_time': 12, 'hgb_detrended': 13, 'hgb_legal_dong': 13,
    'hybrid_ridge': 10, 'hybrid_hgb': 14,
    'hgb_time_kmeans': 15, 'hgb_time_tier': 15, 'hgb_time_gmm': 15,
}


def family_of(model_id):
    """Model family used in the family-level comparison."""
    if model_id.startswith('baseline'):
        return 'baseline'
    if model_id.startswith('hybrid'):
        return 'time-series/hybrid'
    if model_id in ('kmeans_median', 'gmm_median'):
        return 'pure unsupervised'
    if re.search(r'_(kmeans|tier|gmm)$', model_id):
        return 'unsupervised-assisted'
    if model_id.startswith('rf_'):
        return 'forest'
    if model_id.startswith('hgb_'):
        return 'boosting'
    return 'linear'


def complexity_of(model_id):
    base = model_id.removesuffix('_oracle')
    if base not in _COMPLEXITY:
        raise KeyError(f'no complexity level for {model_id}')
    return _COMPLEXITY[base]


def extrapolation_kind(model_id):
    """(how the model handles time, can it extrapolate a trend beyond TRAIN?).

    - forecasted index: structure model + forecast of the quality-adjusted index (hybrid)
    - log-linear trend: detrended models add back a linear month trend fit on TRAIN
    - linear time feature: a linear model with `t` extrapolates its fitted slope
    - tree time feature: trees with `t` are flat beyond the last TRAIN month (cannot extrapolate)
    - none: no time information (static models, baselines, unsupervised medians)
    """
    base = model_id.removesuffix('_oracle')
    if base.startswith('hybrid'):
        return 'forecasted index', True
    if base.endswith('_detrended'):
        return 'log-linear trend (detrended)', True
    if base.startswith(('ridge_time', 'lasso_time', 'elasticnet_time')):
        return 'linear time feature', True
    if base.startswith(('rf_time', 'hgb_time')):
        return 'tree time feature (cannot extrapolate)', False
    if base == 'baseline_drift':
        return 'fixed level factor (no trend)', False
    return 'none', False


def load_runs(metrics_dir=None):
    """Last saved run of every model and track (JSON files named {track}__{model_id}.json)."""
    metrics_dir = config.METRICS_DIR if metrics_dir is None else metrics_dir
    rows = []
    for path in sorted(metrics_dir.glob('*__*.json')):
        track, model_id = path.stem.split('__', 1)
        if track not in config.TRACKS:
            continue
        run = json.loads(path.read_text(encoding='utf-8'))[-1]
        kind, can = extrapolation_kind(model_id)
        rows.append({'track': track, 'model_id': model_id, 'family': family_of(model_id),
                     'valid': bool(run.get('meta', {}).get('valid', True)), 'extrapolation': kind,
                     'can_extrapolate': can, 'complexity': complexity_of(model_id), 'n': run['n'],
                     **{c: run.get(c, np.nan) for c in METRIC_COLUMNS}})
    return pd.DataFrame(rows)


def load_predictions(track, model_id, predictions_dir=None):
    predictions_dir = config.PREDICTIONS_DIR if predictions_dir is None else predictions_dir
    return pd.read_parquet(predictions_dir / f'{track}__{model_id}.parquet')


def ranking(runs):
    """Valid runs ranked by WAPE within each track; invalid runs are returned separately."""
    valid = runs[runs['valid']].copy()
    valid['rank'] = valid.groupby('track')['wape_pct'].rank(method='min').astype(int)
    return valid.sort_values(['track', 'rank']), runs[~runs['valid']].copy()


def family_table(valid_runs):
    """Best and median WAPE by family and track."""
    table = valid_runs.groupby(['track', 'family']).agg(
        n_models=('model_id', 'size'), best_wape=('wape_pct', 'min'), median_wape=('wape_pct', 'median'),
        worst_wape=('wape_pct', 'max'))
    best = valid_runs.loc[valid_runs.groupby(['track', 'family'])['wape_pct'].idxmin()].set_index(['track', 'family'])['model_id']
    table['best_model'] = best
    return table.reset_index()


def one_se_selection(y, predictions, runs_track, n_boot=1000, seed=config.RANDOM_SEED):
    """Lowest-WAPE leader, paired bootstrap of every model against it and the one-SE tie set.

    `predictions` maps model_id -> predicted values (same rows as `y`); `runs_track` holds the
    columns complexity and aggregate_bias_pct of the candidate models. A model is a statistical tie
    with the leader when its WAPE difference to the leader is at most one bootstrap standard error of
    that difference. Among ties the winner has the lowest complexity, then the smallest absolute
    aggregate bias, then the lowest WAPE. Returns (table, leader, selected).
    """
    y = np.asarray(y, dtype=float)
    wapes = {m: wape(y, p) for m, p in predictions.items()}
    leader = min(wapes, key=wapes.get)
    rows = []
    for model_id, pred in predictions.items():
        if model_id == leader:
            boot = {'diff': 0.0, 'se': 0.0, 'ci_low': 0.0, 'ci_high': 0.0}
        else:
            boot = paired_bootstrap_wape_diff(y, pred, predictions[leader], n=n_boot, seed=seed)
        rows.append({'model_id': model_id, 'wape_pct': wapes[model_id], 'diff_vs_leader': boot['diff'],
                     'se_diff': boot['se'], 'ci_low': boot['ci_low'], 'ci_high': boot['ci_high'],
                     'tie_with_leader': model_id == leader or boot['diff'] <= boot['se']})
    table = pd.DataFrame(rows).merge(runs_track[['model_id', 'complexity', 'aggregate_bias_pct']], on='model_id')
    table['abs_aggregate_bias'] = table['aggregate_bias_pct'].abs()
    table = table.sort_values('wape_pct').reset_index(drop=True)
    ties = table[table['tie_with_leader']].sort_values(['complexity', 'abs_aggregate_bias', 'wape_pct'])
    return table, leader, str(ties.iloc[0]['model_id'])


def bootstrap_best_share(y, predictions, n_boot=1000, seed=config.RANDOM_SEED):
    """Share of bootstrap resamples (the same rows for every model) in which each model has the
    lowest WAPE. The shares sum to one (ties are split equally)."""
    y = np.asarray(y, dtype=float)
    names = list(predictions)
    abs_err = np.vstack([np.abs(y - np.asarray(predictions[m], dtype=float)) for m in names])
    abs_y = np.abs(y)
    rng = np.random.default_rng(seed)
    wins = np.zeros(len(names))
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        scores = abs_err[:, idx].sum(axis=1) / abs_y[idx].sum()
        best = scores == scores.min()
        wins[best] += 1.0 / best.sum()
    return pd.Series(wins / n_boot, index=names, name='share_best')


# ----------------------------------------------------------------------------------------------
# Robustness tables
# ----------------------------------------------------------------------------------------------
def error_by_group(groups, y, pred):
    """n, WAPE, median bias and aggregate bias of one model within each group."""
    frame = pd.DataFrame({'g': np.asarray(groups), 'y': np.asarray(y, dtype=float), 'p': np.asarray(pred, dtype=float)})

    def stats(g):
        return pd.Series({'n': len(g), 'wape_pct': wape(g['y'], g['p']),
                          'median_bias_pct': float(np.median((g['p'] - g['y']) / g['y']) * 100),
                          'aggregate_bias_pct': float((g['p'].sum() - g['y'].sum()) / g['y'].sum() * 100)})
    out = frame.groupby('g').apply(stats, include_groups=False)
    out['n'] = out['n'].astype(int)
    return out


def area_buckets(train_area, area, n_buckets=5):
    """Quintile bucket labels of `area` with the edges computed on TRAIN only."""
    edges = np.quantile(np.asarray(train_area, dtype=float), np.linspace(0, 1, n_buckets + 1)[1:-1])
    index = np.searchsorted(edges, np.asarray(area, dtype=float), side='right')
    bounds = [0.0, *edges, np.inf]
    labels = [f'A{i + 1} ({bounds[i]:.0f}-{bounds[i + 1]:.0f} m2)' if np.isfinite(bounds[i + 1])
              else f'A{i + 1} (>{bounds[i]:.0f} m2)' for i in range(n_buckets)]
    return pd.Series(np.asarray(labels, dtype=object)[index])


# ----------------------------------------------------------------------------------------------
# Scenario helpers (projection 2026-2028)
# ----------------------------------------------------------------------------------------------
def representative_profile(track_df, district, building_type, area_sqm, min_rows=30):
    """Structural attributes of a representative contract: median floor and building age of the
    (district, building type) group (type-wide medians if the group has fewer than `min_rows` rows),
    the most frequent contract type and the usual `renewal_right_used`. Returns (profile, n_rows)."""
    group = track_df[(track_df['district_name'] == district) & (track_df['building_type'] == building_type)]
    source = group if len(group) >= min_rows else track_df[track_df['building_type'] == building_type]
    age = (source['contract_date'].dt.year - source['year_built']).clip(lower=0)
    floor = source['floor'].median()
    return {'district_name': district, 'building_type': building_type, 'log_area': float(np.log(area_sqm)),
            'floor': float(floor) if pd.notna(floor) else np.nan, 'floor_missing': int(pd.isna(floor)),
            'building_age': float(age.median()), 'contract_type': str(source['contract_type'].mode().iloc[0]),
            'renewal_right_used': 'No/Not reported'}, int(len(group))


def profile_frame(profile, months):
    """One row per month for a profile, with the time columns the models need."""
    months = pd.DatetimeIndex(months)
    frame = pd.DataFrame([profile] * len(months))
    frame['month_of_year'] = months.month
    frame['month_sin'] = np.sin(2 * np.pi * months.month / 12)
    frame['month_cos'] = np.cos(2 * np.pi * months.month / 12)
    frame[config.TIME_FEATURE] = (months.year - config.TIME_ORIGIN.year) * 12 + (months.month - config.TIME_ORIGIN.month)
    frame.index = months
    return frame


def annual_cash_cost(monthly_values, track):
    """Average monthly prediction of each calendar year; Jeonse = deposit, Wolse = 12 x monthly rent.
    `monthly_values` is a Series indexed by month."""
    yearly = monthly_values.groupby(monthly_values.index.year).mean()
    return yearly * (12 if track == 'wolse' else 1)


def interval_multipliers(index_forecast, lower='lo80', upper='hi80'):
    """Low / high multipliers of the index forecast around its central path (index points)."""
    return (index_forecast[lower] / index_forecast['mean']), (index_forecast[upper] / index_forecast['mean'])
