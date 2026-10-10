"""Rolling-origin pseudo-tests used to select among candidates (CLAUDE.md rule 8, amended).

Two windows, both strictly before the 2025 TEST year:
  fit on rows before 2023-01-01 -> test on 2023
  fit on rows before 2024-01-01 -> test on 2024
A recipe (see `recipes.py`) is fit on the training rows only and its predictions on the test rows
are returned together with the metrics. 2025 never appears here.
"""
import numpy as np
import pandas as pd

from . import config
from .compare import classify_projection, complexity_of, one_se_selection
from .metrics import all_metrics

WINDOWS = (
    {'name': '2023', 'fit_end': pd.Timestamp('2023-01-01'), 'test_year': 2023},
    {'name': '2024', 'fit_end': pd.Timestamp('2024-01-01'), 'test_year': 2024},
)


def check_windows(windows=WINDOWS):
    """Windows are ordered, non-overlapping and end before the 2025 TEST start."""
    ends = [w['fit_end'] for w in windows]
    assert ends == sorted(ends) and len(set(ends)) == len(ends), 'windows must be ordered'
    for w in windows:
        assert w['fit_end'] == pd.Timestamp(f"{w['test_year']}-01-01"), 'the test year starts at the fit end'
        assert pd.Timestamp(f"{w['test_year']}-12-31") < config.TEST_START, 'a pseudo-test window reaches the 2025 TEST year'


def window_split(df_track, window):
    """(train, test) of one window: train is strictly before the fit end, test is the test year."""
    train = df_track[df_track['contract_date'] < window['fit_end']]
    test = df_track[(df_track['contract_date'] >= window['fit_end']) & (df_track['contract_date'].dt.year == window['test_year'])]
    assert train['contract_date'].max() < test['contract_date'].min()
    return train, test


def run_windows(recipe, df_track, features, windows=WINDOWS):
    """Run `recipe` on every window. `features(frame)` builds the feature frame of the test rows.

    Returns {window name: {'metrics', 'y_true', 'y_pred', 'months', 'districts', 'n_train', 'n_test'}}.
    The recipe is fit on the training rows only (it asserts that none is at or after the fit end).
    """
    check_windows(windows)
    target = config.TRACKS[recipe.track]['target']
    out = {}
    for window in windows:
        train, test = window_split(df_track, window)
        recipe.fit(train, window['fit_end'])
        pred = recipe.predict(features(test), test['contract_month'])
        y = test[target].to_numpy()
        out[window['name']] = {
            'metrics': all_metrics(y, pred), 'y_true': y, 'y_pred': np.asarray(pred, dtype=float),
            'months': test['contract_month'].to_numpy(), 'districts': test['district_name'].to_numpy(),
            'n_train': int(len(train)), 'n_test': int(len(test))}
    return out


def pool(results_by_model):
    """Concatenate the windows of every model: (y, {model: pred}, months, districts).

    All models must have predicted the same rows. Months are unique across windows, so month
    blocks of different windows stay distinct."""
    first = next(iter(results_by_model.values()))
    names = list(first)
    y = np.concatenate([first[n]['y_true'] for n in names])
    months = np.concatenate([first[n]['months'] for n in names])
    districts = np.concatenate([first[n]['districts'] for n in names])
    preds = {}
    for model_id, res in results_by_model.items():
        assert all(np.allclose(res[n]['y_true'], first[n]['y_true']) for n in names), f'{model_id}: different test rows'
        preds[model_id] = np.concatenate([res[n]['y_pred'] for n in names])
    assert pd.DatetimeIndex(months).max() < config.TEST_START, 'a selection row belongs to the 2025 TEST year'
    return y, preds, months, districts


def select(results_by_model, n_boot=1000, seed=config.RANDOM_SEED, only=None):
    """One-standard-error selection on the pooled pseudo-test windows (most conservative SE).

    `only` restricts the candidates (for example the models that extrapolate). Returns
    (table, leader, selected); the table is indexed by rank of pooled WAPE."""
    ids = list(results_by_model) if only is None else [m for m in results_by_model if m in set(only)]
    y, preds, months, districts = pool({m: results_by_model[m] for m in ids})
    bias = pd.DataFrame({'model_id': ids,
                         'aggregate_bias_pct': [(preds[m].sum() - y.sum()) / y.sum() * 100 for m in ids],
                         'complexity': [complexity_of(m) for m in ids]})
    return one_se_selection(y, preds, bias, n_boot=n_boot, seed=seed, months=months, districts=districts)


def duan_factor(y_log, pred_log):
    """Duan smearing factor: mean of exp(residual) of a log1p-scale model (residuals of the given rows)."""
    return float(np.mean(np.exp(np.asarray(y_log, dtype=float) - np.asarray(pred_log, dtype=float))))


def smeared_prediction(pred_log, factor):
    """Back-transform with smearing: E[1 + y] = exp(z_hat) * factor, so y_hat = (1 + expm1(z_hat)) * factor - 1."""
    return np.maximum((1.0 + np.expm1(np.asarray(pred_log, dtype=float))) * factor - 1.0, 0.0)


def classify(path):
    """Behavioural label of a projected path (see `compare.classify_projection`)."""
    return classify_projection(path)
