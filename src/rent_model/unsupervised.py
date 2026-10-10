"""Hierarchical (Ward) clustering of districts and a Gaussian mixture on contract-level structure.

Both are leakage-safe transformers: every statistic (district profiles, scaler, linkage, mixture) is
learned from the rows given to `fit` (TRAIN, or the training part of a CV fold) and applied to other
rows only through `transform` / `predict`. This module extends `clustering.py` and reuses it.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .boosting import BoostedModel, feature_columns
from .linear import build_preprocessor as build_linear_preprocessor
from .linear import make_estimator

BUILDING_TYPES = ['Apartment', 'Officetel', 'Row house / Villa (multi-unit)', 'Detached / Multi-household house']
TIER_COLUMN = 'district_tier'
TIER_RANGE = tuple(range(3, 7))
MIN_DISTRICTS_PER_TIER = 2
UNKNOWN_TIER = 'unknown'

GMM_NUMERIC = ['log_area', 'floor', 'building_age']
GMM_CATEGORICAL = ['building_type']
GMM_COVARIANCE_TYPES = ('full', 'tied', 'diag', 'spherical')
GMM_K_RANGE = tuple(range(2, 13))
GMM_REG_COVAR = 1e-3   # building-type dummies make some directions almost constant inside a component


# ----------------------------------------------------------------------------------------------
# Part A - district tiers
# ----------------------------------------------------------------------------------------------
def district_profiles(X, y_log):
    """One row per district from the given rows only: median log target (of the track), median
    log_area, median building_age and the share of each building type."""
    frame = pd.DataFrame({
        'district_name': X['district_name'].to_numpy(), 'log_area': X['log_area'].to_numpy(),
        'building_age': X['building_age'].to_numpy(), 'y': np.asarray(y_log, dtype=float)})
    profile = frame.groupby('district_name').agg(
        median_log_target=('y', 'median'), median_log_area=('log_area', 'median'),
        median_building_age=('building_age', 'median'))
    shares = pd.crosstab(X['district_name'].to_numpy(), X['building_type'].to_numpy(), normalize='index')
    shares = shares.reindex(columns=BUILDING_TYPES, fill_value=0.0)
    shares.columns = [f'share_{c}' for c in shares.columns]
    return profile.join(shares)


def _ward(profile):
    z = StandardScaler().fit_transform(profile)
    return z, linkage(z, method='ward')


def select_n_tiers(profile, tier_range=TIER_RANGE):
    """Silhouette of the Ward cut for each number of tiers (TRAIN profiles only).

    Rule: highest silhouette among the cuts in which every tier has at least 2 districts
    (interpretability: a tier of one district is just that district). Returns (n, table)."""
    z, tree = _ward(profile)
    rows = []
    for n in tier_range:
        labels = fcluster(tree, n, criterion='maxclust')
        sizes = np.bincount(labels)[1:]
        rows.append({'n_tiers': n, 'silhouette': float(silhouette_score(z, labels)),
                     'min_tier_size': int(sizes.min()), 'tier_sizes': sorted(sizes.tolist(), reverse=True)})
    table = pd.DataFrame(rows).set_index('n_tiers')
    table['eligible'] = table['min_tier_size'] >= MIN_DISTRICTS_PER_TIER
    pool = table[table['eligible']] if table['eligible'].any() else table
    return int(pool['silhouette'].idxmax()), table


class DistrictTierer(BaseEstimator, TransformerMixin):
    """Ward clustering of the district profiles; appends `district_tier` (T0 = lowest median target).

    fit(X, y_log) learns the profiles, the linkage and the district -> tier map from the given
    rows only. Districts not seen in fit get the tier 'unknown'."""

    def __init__(self, n_tiers=4):
        self.n_tiers = n_tiers

    def fit(self, X, y):
        self.profiles_ = district_profiles(X, y)
        self.scaler_ = StandardScaler().fit(self.profiles_)
        self.linkage_ = linkage(self.scaler_.transform(self.profiles_), method='ward')
        raw = fcluster(self.linkage_, self.n_tiers, criterion='maxclust')
        order = pd.Series(self.profiles_['median_log_target'].to_numpy()).groupby(raw).mean().sort_values().index
        rank = {cluster: f'T{i}' for i, cluster in enumerate(order)}
        self.mapping_ = {d: rank[c] for d, c in zip(self.profiles_.index, raw)}
        return self

    def transform(self, X):
        return X.assign(**{TIER_COLUMN: X['district_name'].map(self.mapping_).fillna(UNKNOWN_TIER).to_numpy()})


def tier_table(tierer):
    """Profile of each tier (medians over its districts) for printing."""
    prof = tierer.profiles_.assign(tier=pd.Series(tierer.mapping_))
    table = prof.groupby('tier').agg(
        n_districts=('median_log_target', 'size'), median_log_target=('median_log_target', 'median'),
        median_log_area=('median_log_area', 'median'), median_building_age=('median_building_age', 'median'),
        **{c: (c, 'median') for c in prof.columns if c.startswith('share_')})
    table['median_target'] = np.expm1(table['median_log_target'])
    table['districts'] = prof.groupby('tier').apply(lambda g: ', '.join(g.index.str.replace('-gu', '')), include_groups=False)
    return table


def name_tiers(table):
    """Plain-language names from the printed tier table: price rank and apartment share."""
    labels = {1: ['all'], 3: ['lower-price', 'mid-price', 'higher-price'],
              4: ['lowest-price', 'lower-mid-price', 'upper-mid-price', 'highest-price'],
              5: ['lowest-price', 'lower-price', 'mid-price', 'higher-price', 'highest-price'],
              6: ['lowest-price', 'lower-price', 'lower-mid-price', 'upper-mid-price', 'higher-price', 'highest-price']}
    order = table['median_log_target'].sort_values().index
    apt = table['share_Apartment']
    names = {}
    for i, tier in enumerate(order):
        tag = labels[len(order)][i]
        extra = ', apartment-heavy' if apt[tier] == apt.max() else ', apartment-light' if apt[tier] == apt.min() else ''
        names[tier] = f'{tier} {tag}{extra}'
    return names


# ----------------------------------------------------------------------------------------------
# Part B - Gaussian mixture on contract structure
# ----------------------------------------------------------------------------------------------
def build_gmm_preprocessor():
    """Median imputation of floor, one-hot of building_type, then standardization of every column."""
    numeric = Pipeline([('impute', SimpleImputer(strategy='median'))])
    columns = ColumnTransformer(
        [('num', numeric, GMM_NUMERIC),
         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), GMM_CATEGORICAL)])
    return Pipeline([('columns', columns), ('scale', StandardScaler())])


def prob_columns(n_components):
    return [f'gmm_p{i}' for i in range(n_components)]


class GMMLabeler(BaseEstimator, TransformerMixin):
    """Preprocessing + GaussianMixture fit on the given rows. `transform` appends the soft
    assignment probabilities `gmm_p0 .. gmm_p{k-1}`; `components` gives the hard assignment."""

    def __init__(self, n_components=6, covariance_type='full', reg_covar=GMM_REG_COVAR, n_init=2,
                 random_state=config.RANDOM_SEED):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X, y=None):
        self.pre_ = build_gmm_preprocessor()
        A = self.pre_.fit_transform(X[GMM_NUMERIC + GMM_CATEGORICAL])
        self.gmm_ = GaussianMixture(
            n_components=self.n_components, covariance_type=self.covariance_type, reg_covar=self.reg_covar,
            n_init=self.n_init, random_state=self.random_state, max_iter=300).fit(A)
        return self

    def embed(self, X):
        return self.pre_.transform(X[GMM_NUMERIC + GMM_CATEGORICAL])

    def probabilities(self, X):
        return self.gmm_.predict_proba(self.embed(X))

    def components(self, X):
        return self.gmm_.predict(self.embed(X))

    def transform(self, X):
        probs = pd.DataFrame(self.probabilities(X), columns=prob_columns(self.n_components), index=X.index)
        return pd.concat([X, probs], axis=1)


def select_gmm(X_train, k_range=GMM_K_RANGE, covariance_types=GMM_COVARIANCE_TYPES, n_init=2):
    """BIC and AIC of every (covariance type, k) on TRAIN. Chosen = minimum BIC.

    Returns (covariance_type, k, table) with one row per (covariance type, k)."""
    pre = build_gmm_preprocessor()
    A = pre.fit_transform(X_train[GMM_NUMERIC + GMM_CATEGORICAL])
    rows = []
    for cov, k in itertools.product(covariance_types, k_range):
        gmm = GaussianMixture(n_components=k, covariance_type=cov, reg_covar=GMM_REG_COVAR, n_init=n_init,
                              random_state=config.RANDOM_SEED, max_iter=300).fit(A)
        rows.append({'covariance_type': cov, 'k': k, 'bic': float(gmm.bic(A)), 'aic': float(gmm.aic(A)),
                     'converged': bool(gmm.converged_)})
    table = pd.DataFrame(rows).set_index(['covariance_type', 'k'])
    best = table['bic'].idxmin()
    return best[0], int(best[1]), table


class GMMMedian(BaseEstimator, RegressorMixin):
    """Hard assignment to the most probable component; predicts the TRAIN median of that component
    (original wones). Components empty under hard assignment fall back to the global median."""

    def __init__(self, n_components=6, covariance_type='full'):
        self.n_components = n_components
        self.covariance_type = covariance_type

    def fit(self, X, y_original):
        self.labeler_ = GMMLabeler(self.n_components, self.covariance_type).fit(X)
        y = pd.Series(np.asarray(y_original, dtype=float))
        self.global_ = float(y.median())
        self.medians_ = y.groupby(self.labeler_.components(X)).median()
        return self

    def predict(self, X):
        return self.medians_.reindex(self.labeler_.components(X)).fillna(self.global_).to_numpy()


# ----------------------------------------------------------------------------------------------
# Models that use a label / probabilities as features
# ----------------------------------------------------------------------------------------------
BASE_RIDGE_COLUMNS = (list(config.CATEGORICAL_FEATURES)
                      + ['log_area', 'floor', 'floor_missing', 'building_age', 'month_sin', 'month_cos', config.TIME_FEATURE])


def build_ridge_pipeline(labeler, alpha, extra_categorical=(), extra_numeric=()):
    """`ridge_time` configuration plus the columns produced by `labeler` (fit inside the pipeline)."""
    parts = [('base', build_linear_preprocessor(time=True), BASE_RIDGE_COLUMNS)]
    if extra_categorical:
        parts.append(('extra_cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(extra_categorical)))
    if extra_numeric:
        parts.append(('extra_num', StandardScaler(), list(extra_numeric)))
    return Pipeline([('label', labeler), ('prep', ColumnTransformer(parts, remainder='drop')),
                     ('model', make_estimator('ridge', alpha=alpha))])


class BoostedWithLabeler(BaseEstimator, RegressorMixin):
    """`hgb_time` plus the columns produced by `labeler`; labeler and booster are fit on the rows
    given to `fit`. Categorical extras are native categoricals, numeric extras are plain features."""

    def __init__(self, labeler=None, extra_categorical=(), extra_numeric=(), learning_rate=0.1, max_iter=500,
                 max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0):
        self.labeler = labeler
        self.extra_categorical = extra_categorical
        self.extra_numeric = extra_numeric
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization

    def _columns(self):
        return feature_columns('time', self.extra_categorical, self.extra_numeric)

    def fit(self, X, y):
        self.labeler_ = clone(self.labeler).fit(X, y)
        self.model_ = BoostedModel(
            variant='time', learning_rate=self.learning_rate, max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes, min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization, extra_categorical=tuple(self.extra_categorical),
            extra_numeric=tuple(self.extra_numeric)).fit(self.labeler_.transform(X)[self._columns()], y)
        self.n_iter_ = self.model_.n_iter_
        return self

    def predict(self, X):
        return self.model_.predict(self.labeler_.transform(X)[self._columns()])
