"""Random Forest models on log1p(target): pipeline, detrended wrapper and time-aware tuning.

Categorical encoding lives inside the Pipeline (fit on TRAIN only). `floor` keeps its NaN, which
scikit-learn forests handle natively, next to the `floor_missing` flag.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from . import config
from .features import inverse_target
from .metrics import wape

STATIC_NUMERIC = ['log_area', 'floor', 'floor_missing', 'building_age']
TIME = config.TIME_FEATURE

PARAM_DISTRIBUTIONS = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_leaf': [1, 2, 5, 10, 20],
    'max_features': [0.3, 0.5, 0.7, 1.0],
}


def feature_columns(time=False):
    """Design columns of the forest. The static variant has no time feature at all."""
    return list(config.CATEGORICAL_FEATURES) + STATIC_NUMERIC + ([TIME] if time else [])


def build_forest_pipeline(time=False, **params):
    """Ordinal encoding (unknown category -> -1) + RandomForestRegressor."""
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    columns = ColumnTransformer(
        [('cat', encoder, list(config.CATEGORICAL_FEATURES)),
         ('num', 'passthrough', STATIC_NUMERIC + ([TIME] if time else []))],
        remainder='drop', verbose_feature_names_out=False)
    forest = RandomForestRegressor(n_jobs=-1, random_state=config.RANDOM_SEED, **params)
    return Pipeline([('prep', columns), ('model', forest)])


class DetrendedForest(RegressorMixin, BaseEstimator):
    """Log-linear month trend + forest on the residual.

    fit: y_log = a + b * t is fit by least squares on the rows given to `fit` (TRAIN or the
    training part of a CV fold); a static forest (no `t`) learns y_log - (a + b * t); predict
    adds the trend back, so the trend is extrapolated linearly beyond the training months.
    `X` must contain the column `t`; the forest never sees it.
    """

    def __init__(self, **forest_params):
        self.forest_params = forest_params

    def get_params(self, deep=True):
        return dict(self.forest_params)

    def set_params(self, **params):
        self.forest_params = {**self.forest_params, **params}
        return self

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        t = X[TIME].to_numpy(dtype=float)
        self.slope_, self.intercept_ = np.polyfit(t, y, 1)
        self.model_ = build_forest_pipeline(time=False, **self.forest_params)
        self.model_.fit(X, y - self.trend(t))
        return self

    def trend(self, t):
        return self.intercept_ + self.slope_ * np.asarray(t, dtype=float)

    def predict(self, X):
        return self.model_.predict(X) + self.trend(X[TIME].to_numpy(dtype=float))


def make_model(variant, **params):
    """variant: 'static', 'time' or 'detrended'."""
    if variant == 'static':
        return build_forest_pipeline(time=False, **params)
    if variant == 'time':
        return build_forest_pipeline(time=True, **params)
    if variant == 'detrended':
        return DetrendedForest(**params)
    raise ValueError(f'unknown variant: {variant}')


def to_original(pred_log):
    """Back-transform to wones; negative log predictions are floored at 0 (target >= 0)."""
    return inverse_target(np.maximum(pred_log, 0.0))


def wape_scorer(estimator, X, y_log):
    """Negative WAPE on the original scale (higher is better, as scikit-learn expects)."""
    return -wape(inverse_target(np.asarray(y_log, dtype=float)), to_original(estimator.predict(X)))


def tune(variant, X, y_log, folds, n_iter=20):
    """RandomizedSearchCV over expanding-window folds; score = WAPE on the original scale.

    Returns (best_params, cv_table sorted by mean WAPE). refit=False: the caller refits the
    final model on the full TRAIN.
    """
    prefix = '' if variant == 'detrended' else 'model__'
    distributions = {prefix + k: v for k, v in PARAM_DISTRIBUTIONS.items()}
    search = RandomizedSearchCV(
        make_model(variant), distributions, n_iter=n_iter, cv=folds,
        scoring=wape_scorer, refit=False, n_jobs=1, random_state=config.RANDOM_SEED)
    search.fit(X[feature_columns(time=variant != 'static')], y_log)
    res = search.cv_results_
    table = pd.DataFrame([{k[len(prefix):]: v for k, v in p.items()} for p in res['params']])
    table['cv_wape_mean'] = -res['mean_test_score']
    table['cv_wape_std'] = res['std_test_score']
    for k in range(len(folds)):
        table[f'fold{k + 1}'] = -res[f'split{k}_test_score']
    table = table.sort_values('cv_wape_mean').reset_index(drop=True)
    best = {k: (None if pd.isna(table.loc[0, k]) else table.loc[0, k]) for k in PARAM_DISTRIBUTIONS}
    best = {k: (v if v is None else (int(v) if k != 'max_features' else float(v))) for k, v in best.items()}
    return best, table


def fit_predict(variant, params, X_train, y_train_log, X_test):
    """Fit on TRAIN only; returns (fitted model, predictions in original wones)."""
    cols = feature_columns(time=variant != 'static')
    model = make_model(variant, **params).fit(X_train[cols], y_train_log)
    return model, to_original(model.predict(X_test[cols]))
