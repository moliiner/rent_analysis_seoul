"""Histogram gradient boosting on log1p(target) with early stopping on a TRAIN-only validation tail.

Categoricals are ordinal-encoded inside the estimator (fit on the rows given to `fit`) and passed
to HistGradientBoostingRegressor as native categorical features; NaN floors are kept natively.
`fit` receives rows of TRAIN (or of a CV fold's training part) only: the validation tail is the
last `tail_months` months of those rows, so no TEST or validation-fold row is ever used.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from . import config
from .forest import to_original, wape_scorer  # noqa: F401  (re-exported for the notebook)

TIME = config.TIME_FEATURE
DONG = config.LEGAL_DONG_FEATURE
NUMERIC = ['log_area', 'floor', 'floor_missing', 'building_age']
OTHER = '__other__'

VARIANTS = ('static', 'time', 'detrended', 'legal_dong')

PARAM_DISTRIBUTIONS = {
    'learning_rate': [0.03, 0.05, 0.1],
    'max_iter': [300, 600, 1000],            # upper cap: early stopping picks the actual number
    'max_leaf_nodes': [15, 31, 63, 127],
    'min_samples_leaf': [10, 20, 50, 100],
    'l2_regularization': [0.0, 0.1, 1.0, 10.0],
}


def feature_columns(variant, extra_categorical=()):
    """Columns that are model features. `t` is a feature of the 'time' variant only.

    `extra_categorical` appends caller-built categorical columns (for example a cluster label)."""
    cols = list(config.CATEGORICAL_FEATURES) + list(extra_categorical) + NUMERIC
    if variant == 'time':
        cols.append(TIME)
    if variant == 'legal_dong':
        cols.append(DONG)
    return cols


def required_columns(variant):
    """Columns the estimator needs in X: the features plus `t` (validation tail and trend)."""
    cols = feature_columns(variant)
    return cols if TIME in cols else cols + [TIME]


class RareGrouper(BaseEstimator, TransformerMixin):
    """Keeps the most frequent categories (count >= min_frequency, at most max_categories) and maps
    the rest, and categories unseen in fit, to one 'other' level. Needed because native categorical
    support in HistGradientBoosting is limited to 255 levels."""

    def __init__(self, min_frequency=30, max_categories=254):
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    def fit(self, X, y=None):
        counts = pd.Series(np.asarray(X).ravel()).value_counts()
        counts = counts[counts >= self.min_frequency].head(self.max_categories)
        self.keep_ = set(counts.index)
        return self

    def transform(self, X):
        values = pd.Series(np.asarray(X).ravel(), dtype=object)
        return np.where(values.isin(self.keep_), values, OTHER).reshape(-1, 1)


def build_preprocessor(variant, min_frequency=30, extra_categorical=()):
    """Ordinal encoding of the categoricals (unknown -> NaN, i.e. 'missing' for the booster)."""
    def encoder():
        return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)

    parts = [('cat', encoder(), list(config.CATEGORICAL_FEATURES) + list(extra_categorical))]
    if variant == 'legal_dong':
        parts.append(('dong', Pipeline([('group', RareGrouper(min_frequency)), ('ord', encoder())]), [DONG]))
    parts.append(('num', 'passthrough', NUMERIC + ([TIME] if variant == 'time' else [])))
    return ColumnTransformer(parts, remainder='drop', verbose_feature_names_out=False)


def _stage_loss(y_true, y_pred, quantile):
    if quantile is None:
        return float(np.mean((y_true - y_pred) ** 2))
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


class BoostedModel(RegressorMixin, BaseEstimator):
    """HistGradientBoostingRegressor with validation-tail early stopping.

    variant: 'static', 'time', 'detrended' (log-linear month trend fit on the rows given to `fit`,
    booster learns the residual, trend added back in `predict`) or 'legal_dong'.
    loss='quantile' with `quantile` in (0, 1) gives a quantile booster (stopping on pinball loss).
    X must contain `required_columns(variant)`.
    """

    def __init__(self, variant='static', learning_rate=0.1, max_iter=500, max_leaf_nodes=31,
                 min_samples_leaf=20, l2_regularization=0.0, quantile=None, tail_months=6,
                 min_frequency=30, extra_categorical=()):
        self.variant = variant
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.quantile = quantile
        self.tail_months = tail_months
        self.min_frequency = min_frequency
        self.extra_categorical = extra_categorical

    def _booster(self, max_iter, categorical):
        kwargs = dict(loss='squared_error') if self.quantile is None else dict(loss='quantile', quantile=self.quantile)
        return HistGradientBoostingRegressor(
            learning_rate=self.learning_rate, max_iter=max_iter, max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf, l2_regularization=self.l2_regularization,
            categorical_features=categorical, early_stopping=False,
            random_state=config.RANDOM_SEED, **kwargs)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        t = X[TIME].to_numpy(dtype=float)
        if self.variant == 'detrended':
            self.slope_, self.intercept_ = np.polyfit(t, y, 1)
            y_fit = y - (self.intercept_ + self.slope_ * t)
        else:
            y_fit = y

        self.pre_ = build_preprocessor(self.variant, self.min_frequency, self.extra_categorical)
        A = self.pre_.fit_transform(X[feature_columns(self.variant, self.extra_categorical)])
        n_cat = (len(config.CATEGORICAL_FEATURES) + len(self.extra_categorical)
                 + (1 if self.variant == 'legal_dong' else 0))
        categorical = np.arange(A.shape[1]) < n_cat

        tail = t > t.max() - self.tail_months
        if tail.any() and (~tail).sum() > 0:
            probe = self._booster(self.max_iter, categorical).fit(A[~tail], y_fit[~tail])
            losses = [_stage_loss(y_fit[tail], p, self.quantile) for p in probe.staged_predict(A[tail])]
            self.n_iter_ = int(np.argmin(losses)) + 1
        else:
            self.n_iter_ = int(self.max_iter)
        self.model_ = self._booster(self.n_iter_, categorical).fit(A, y_fit)
        return self

    def predict(self, X):
        pred = self.model_.predict(self.pre_.transform(X[feature_columns(self.variant, self.extra_categorical)]))
        if self.variant == 'detrended':
            pred = pred + self.intercept_ + self.slope_ * X[TIME].to_numpy(dtype=float)
        return pred


def tune(variant, X, y_log, folds, n_iter=20):
    """RandomizedSearchCV over expanding-window folds; score = WAPE on the original scale.

    Each candidate is fit with validation-tail early stopping inside the training part of its fold.
    Returns (best_params, cv_table sorted by mean WAPE).
    """
    search = RandomizedSearchCV(
        BoostedModel(variant=variant), PARAM_DISTRIBUTIONS, n_iter=n_iter, cv=folds,
        scoring=wape_scorer, refit=False, n_jobs=1, random_state=config.RANDOM_SEED)
    search.fit(X[required_columns(variant)], y_log)
    res = search.cv_results_
    table = pd.DataFrame(res['params'])
    table['cv_wape_mean'] = -res['mean_test_score']
    table['cv_wape_std'] = res['std_test_score']
    for k in range(len(folds)):
        table[f'fold{k + 1}'] = -res[f'split{k}_test_score']
    table = table.sort_values('cv_wape_mean').reset_index(drop=True)
    best = {k: (int(v) if k in ('max_iter', 'max_leaf_nodes', 'min_samples_leaf') else float(v))
            for k, v in ((k, table.loc[0, k]) for k in PARAM_DISTRIBUTIONS)}
    return best, table


def fit_predict(variant, params, X_train, y_train_log, X_test, **extra):
    """Fit on TRAIN only. Returns (fitted model, predictions in original wones)."""
    model = BoostedModel(variant=variant, **params, **extra).fit(X_train[required_columns(variant)], y_train_log)
    return model, to_original(model.predict(X_test[required_columns(variant)]))


def fit_interval(variant, params, X_train, y_train_log, X_test, low=0.1, high=0.9):
    """Quantile boosters (same hyperparameters) -> prediction interval on the original scale."""
    bounds = []
    for q in (low, high):
        _, pred = fit_predict(variant, params, X_train, y_train_log, X_test, quantile=q)
        bounds.append(pred)
    lower, upper = np.minimum(*bounds), np.maximum(*bounds)
    return lower, upper


def interval_coverage(y, lower, upper):
    """Share of observations inside [lower, upper] and mean interval width."""
    y = np.asarray(y, dtype=float)
    return float(np.mean((y >= lower) & (y <= upper))), float(np.mean(upper - lower))
