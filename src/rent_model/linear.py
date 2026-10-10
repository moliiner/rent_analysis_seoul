"""Regularized linear models on log1p(target): pipeline, time-aware tuning and fit/predict.

Scaling and one-hot encoding live inside the sklearn Pipeline, so they are fit on whatever rows
are passed to `fit` (TRAIN, or the training part of a CV fold) and never see validation/TEST rows.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .features import inverse_target
from .metrics import wape

STATIC_NUMERIC = ['log_area', 'floor', 'floor_missing', 'building_age']
TIME_NUMERIC = ['month_sin', 'month_cos', config.TIME_FEATURE]
INTERACTION_COLUMN = 'district_x_type'

GRIDS = {
    'ridge': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    'lasso': {'alpha': [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]},
    'elasticnet': {'alpha': [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2], 'l1_ratio': [0.2, 0.5, 0.8]},
}


class RichInputs(BaseEstimator, TransformerMixin):
    """Stateless helper: adds the district x building_type key used by the "rich" variant."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[INTERACTION_COLUMN] = X['district_name'].astype(str) + ' | ' + X['building_type'].astype(str)
        return X


class AreaByType(BaseEstimator, TransformerMixin):
    """log_area x building_type interaction: one slope column per building type seen in fit."""

    def fit(self, X, y=None):
        self.types_ = sorted(X['building_type'].dropna().unique())
        return self

    def transform(self, X):
        area = X['log_area'].to_numpy(dtype=float)
        kind = X['building_type'].to_numpy()
        return np.column_stack([area * (kind == t) for t in self.types_])

    def get_feature_names_out(self, names=None):
        return np.array([f'log_area x {t}' for t in self.types_], dtype=object)


def numeric_columns(time=False):
    return STATIC_NUMERIC + (TIME_NUMERIC if time else [])


def build_preprocessor(time=False, rich=False):
    """ColumnTransformer: one-hot for categoricals, impute + scale for numerics."""
    categorical = list(config.CATEGORICAL_FEATURES) + ([INTERACTION_COLUMN] if rich else [])
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    parts = [
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
        ('num', numeric, numeric_columns(time)),
    ]
    if rich:
        parts.append(('area_x_type', Pipeline([('inter', AreaByType()), ('scale', StandardScaler())]),
                      ['log_area', 'building_type']))
    prep = ColumnTransformer(parts, remainder='drop', verbose_feature_names_out=False)
    if rich:
        return Pipeline([('inputs', RichInputs()), ('columns', prep)])
    return prep


def make_estimator(family, **params):
    if family == 'ols':
        return LinearRegression()
    if family == 'ridge':
        return Ridge(random_state=config.RANDOM_SEED, **params)
    if family == 'lasso':
        return Lasso(max_iter=100_000, precompute=True, random_state=config.RANDOM_SEED, **params)
    if family == 'elasticnet':
        return ElasticNet(max_iter=100_000, precompute=True, random_state=config.RANDOM_SEED, **params)
    raise ValueError(f'unknown family: {family}')


def build_pipeline(family, time=False, rich=False, **params):
    """Full pipeline (preprocessing + estimator) for one linear family."""
    return Pipeline([('prep', build_preprocessor(time, rich)),
                     ('model', make_estimator(family, **params))])


def feature_names(pipe):
    """Readable design-matrix column names of a fitted pipeline."""
    prep = pipe.named_steps['prep']
    if isinstance(prep, Pipeline):
        prep = prep.named_steps['columns']
    names = []
    for name, transformer, columns in prep.transformers_:
        if name == 'cat':
            names += list(transformer.get_feature_names_out(columns))
        elif name == 'num':
            names += list(columns)
        elif name == 'area_x_type':
            names += list(transformer.named_steps['inter'].get_feature_names_out())
    return names


def _to_original(pred_log, floor_log):
    """Back-transform to wones; predictions are floored at the smallest TRAIN target."""
    return inverse_target(np.maximum(pred_log, floor_log))


def tune(family, X, y_log, folds, time=False, rich=False, grid=None):
    """Grid search with the expanding-window folds; score = WAPE on the original scale.

    `folds` are positional (train_pos, valid_pos) pairs from `splits.expanding_window_folds`
    computed on TRAIN. Preprocessing is fit on the training part of each fold only.
    Returns (best_params, cv_table) with cv_table sorted by mean WAPE.
    """
    grid = GRIDS[family] if grid is None else grid
    keys = list(grid)
    combos = [dict(zip(keys, values)) for values in itertools.product(*grid.values())]
    scores = {i: [] for i in range(len(combos))}

    for train_pos, valid_pos in folds:
        prep = build_preprocessor(time, rich)
        A_train = prep.fit_transform(X.iloc[train_pos])
        A_valid = prep.transform(X.iloc[valid_pos])
        y_train = y_log.iloc[train_pos].to_numpy()
        y_valid = inverse_target(y_log.iloc[valid_pos].to_numpy())
        floor_log = y_train.min()
        # Lasso / Elastic Net: walk each alpha path from strong to weak penalty, warm-starting
        # from the previous solution (same optimum, much faster convergence).
        order = sorted(range(len(combos)), key=lambda i: -combos[i].get('alpha', 0))
        warm = {}
        for i in order:
            params = combos[i]
            if family in ('lasso', 'elasticnet'):
                key = params.get('l1_ratio')
                model = warm.get(key) or make_estimator(family, **params).set_params(warm_start=True)
                warm[key] = model.set_params(**params)
            else:
                model = make_estimator(family, **params)
            model.fit(A_train, y_train)
            scores[i].append(wape(y_valid, _to_original(model.predict(A_valid), floor_log)))

    table = pd.DataFrame(combos)
    table['cv_wape_mean'] = [np.mean(scores[i]) for i in range(len(combos))]
    table['cv_wape_std'] = [np.std(scores[i], ddof=1) for i in range(len(combos))]
    for k in range(len(folds)):
        table[f'fold{k + 1}'] = [scores[i][k] for i in range(len(combos))]
    table = table.sort_values('cv_wape_mean').reset_index(drop=True)
    best = {k: table.loc[0, k] for k in keys}
    return best, table


def fit_predict(pipe, X_train, y_train_log, X_test):
    """Fit on TRAIN only and predict TEST in original wones. Returns (fitted pipeline, predictions)."""
    pipe = clone(pipe).fit(X_train, y_train_log)
    floor_log = float(np.min(y_train_log))
    return pipe, _to_original(pipe.predict(X_test), floor_log)
