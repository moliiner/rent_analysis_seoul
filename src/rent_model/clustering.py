"""K-Means on structural descriptors, used as a cluster-median predictor and as a feature generator.

Leakage rules: the scaler, the encoder and K-Means are fit only on the rows passed to `fit` (TRAIN, or
the training part of a CV fold); the target is never an input of the clustering. Descriptors are
structural only: log_area, floor (+ floor_missing), building_age and building_type. District identity
is not a clustering input (it is already a feature of the supervised models).
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .boosting import BoostedModel, required_columns
from .linear import build_preprocessor as build_linear_preprocessor
from .linear import make_estimator

NUMERIC_DESCRIPTORS = ['log_area', 'floor', 'floor_missing', 'building_age']
CATEGORICAL_DESCRIPTORS = ['building_type']
DESCRIPTORS = NUMERIC_DESCRIPTORS + CATEGORICAL_DESCRIPTORS
CLUSTER_COLUMN = 'cluster'

K_RANGE = tuple(range(2, 13))
MIN_STABILITY_ARI = 0.80
MIN_K = 4   # one cluster per building type is the minimum structure the descriptors can express


def build_descriptor_preprocessor():
    """Median imputation + standardization of the numeric descriptors, one-hot of building_type."""
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    return ColumnTransformer(
        [('num', numeric, NUMERIC_DESCRIPTORS),
         ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_DESCRIPTORS)],
        remainder='drop')


class StructuralKMeans(BaseEstimator, TransformerMixin):
    """Descriptor preprocessing + K-Means. `fit` learns scaler, encoder and centroids from the given
    rows only; `predict` assigns unseen rows to the nearest TRAIN centroid."""

    def __init__(self, k=6, n_init=10, random_state=config.RANDOM_SEED):
        self.k = k
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X, y=None):
        self.pre_ = build_descriptor_preprocessor()
        A = self.pre_.fit_transform(X[DESCRIPTORS])
        self.kmeans_ = KMeans(n_clusters=self.k, n_init=self.n_init, random_state=self.random_state).fit(A)
        self.labels_ = self.kmeans_.labels_
        return self

    def embed(self, X):
        return self.pre_.transform(X[DESCRIPTORS])

    def predict(self, X):
        return self.kmeans_.predict(self.embed(X))

    def transform(self, X):
        return self.kmeans_.transform(self.embed(X))


def select_k(X_train, k_range=K_RANGE, n_seeds=10, silhouette_rows=6_000):
    """Diagnostics on TRAIN only: inertia (elbow), silhouette and stability across seeds.

    For every k, K-Means is fit with `n_seeds` different seeds (n_init=1). Stability is the mean
    pairwise adjusted Rand index between the labelings of different seeds. Silhouette is the mean
    over seeds on one fixed TRAIN subsample. The chosen k is the one with the highest mean
    silhouette among the k that are stable (ARI >= 0.80) and k >= 4. Returns (k, table).
    """
    pre = build_descriptor_preprocessor()
    A = pre.fit_transform(X_train[DESCRIPTORS])
    rng = np.random.default_rng(config.RANDOM_SEED)
    sample = rng.choice(len(A), size=min(silhouette_rows, len(A)), replace=False)
    seeds = [config.RANDOM_SEED + i for i in range(n_seeds)]

    rows = []
    for k in k_range:
        labelings, inertias, silhouettes = [], [], []
        for seed in seeds:
            km = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(A)
            labelings.append(km.labels_)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(A[sample], km.labels_[sample]))
        ari = [adjusted_rand_score(a, b) for a, b in itertools.combinations(labelings, 2)]
        rows.append({'k': k, 'inertia': float(np.mean(inertias)), 'silhouette': float(np.mean(silhouettes)),
                     'silhouette_std': float(np.std(silhouettes, ddof=1)),
                     'stability_ari': float(np.mean(ari)), 'stability_ari_min': float(np.min(ari))})
    table = pd.DataFrame(rows).set_index('k')
    table['inertia_drop_pct'] = -table['inertia'].pct_change() * 100
    table['eligible'] = (table['stability_ari'] >= MIN_STABILITY_ARI) & (table.index >= MIN_K)
    pool = table[table['eligible']] if table['eligible'].any() else table[table.index >= MIN_K]
    return int(pool['silhouette'].idxmax()), table


def describe_clusters(X, labels, y_original=None):
    """Median profile of each cluster (structural descriptors; target only for description)."""
    frame = X[['log_area', 'floor', 'building_age']].assign(
        area_sqm=np.exp(X['log_area']), cluster=np.asarray(labels))
    profile = frame.groupby('cluster').agg(
        n=('log_area', 'size'), median_area_sqm=('area_sqm', 'median'),
        median_floor=('floor', 'median'), median_building_age=('building_age', 'median'))
    profile['share_pct'] = profile['n'] / profile['n'].sum() * 100
    types = pd.crosstab(np.asarray(labels), X['building_type'].to_numpy(), normalize='index') * 100
    types.columns = [f'pct_{c}' for c in types.columns]
    profile = profile.join(types)
    if y_original is not None:
        profile['median_target'] = pd.Series(np.asarray(y_original)).groupby(np.asarray(labels)).median()
    return profile


def name_clusters(profile):
    """Plain-language names from the profile: dominant building type, size tier and age tier.

    Size and age tiers compare each cluster with the median of the cluster medians
    (below / above / equal -> small|large, newer|older; the middle third is left unlabeled).
    """
    type_cols = [c for c in profile.columns if c.startswith('pct_')]
    short = {'Apartment': 'Apartment', 'Officetel': 'Officetel',
             'Row house / Villa (multi-unit)': 'Row house/Villa',
             'Detached / Multi-household house': 'Detached house'}
    area_lo, area_hi = profile['median_area_sqm'].quantile([1 / 3, 2 / 3])
    age_lo, age_hi = profile['median_building_age'].quantile([1 / 3, 2 / 3])
    names = {}
    for c, row in profile.iterrows():
        kind = row[type_cols].astype(float).idxmax().removeprefix('pct_')
        size = 'small' if row['median_area_sqm'] <= area_lo else 'large' if row['median_area_sqm'] >= area_hi else 'mid-size'
        age = 'newer' if row['median_building_age'] <= age_lo else 'older' if row['median_building_age'] >= age_hi else 'mid-age'
        names[c] = f"C{c} {short.get(kind, kind)}, {size}, {age}"
    return names


class ClusterMedian(BaseEstimator, RegressorMixin):
    """Predicts the TRAIN median of the target in each cluster (original wones)."""

    def __init__(self, k=6):
        self.k = k

    def fit(self, X, y_original):
        self.clusterer_ = StructuralKMeans(k=self.k).fit(X)
        y = pd.Series(np.asarray(y_original, dtype=float))
        self.global_ = float(y.median())
        self.medians_ = y.groupby(self.clusterer_.labels_).median()
        return self

    def predict(self, X):
        labels = self.clusterer_.predict(X)
        return self.medians_.reindex(labels).fillna(self.global_).to_numpy()


class ClusterLabeler(BaseEstimator, TransformerMixin):
    """Pipeline step: fits StructuralKMeans on the rows given to `fit` and appends a string
    `cluster` column (C0, C1, ...) to X. Inside a Pipeline this keeps the clustering inside the
    training rows of every fit / CV fold."""

    def __init__(self, k=6):
        self.k = k

    def fit(self, X, y=None):
        self.clusterer_ = StructuralKMeans(k=self.k).fit(X)
        return self

    def transform(self, X):
        labels = self.clusterer_.predict(X)
        return X.assign(**{CLUSTER_COLUMN: pd.Series(labels, index=X.index).map(lambda c: f'C{c}')})


def build_ridge_kmeans_pipeline(k, alpha):
    """ridge_time configuration (given alpha) plus the one-hot cluster label."""
    base_columns = list(config.CATEGORICAL_FEATURES) + ['log_area', 'floor', 'floor_missing', 'building_age',
                                                        'month_sin', 'month_cos', config.TIME_FEATURE]
    design = ColumnTransformer(
        [('base', build_linear_preprocessor(time=True), base_columns),
         ('cluster', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [CLUSTER_COLUMN])],
        remainder='drop')
    return Pipeline([('label', ClusterLabeler(k)), ('prep', design), ('model', make_estimator('ridge', alpha=alpha))])


class BoostedWithClusters(BaseEstimator, RegressorMixin):
    """hgb_time plus the cluster label as a native categorical feature.

    Clustering and boosting are both fit on the rows given to `fit`."""

    def __init__(self, k=6, learning_rate=0.1, max_iter=500, max_leaf_nodes=31, min_samples_leaf=20,
                 l2_regularization=0.0):
        self.k = k
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization

    def _columns(self):
        return required_columns('time') + [CLUSTER_COLUMN]

    def fit(self, X, y):
        self.labeler_ = ClusterLabeler(self.k).fit(X)
        self.model_ = BoostedModel(
            variant='time', learning_rate=self.learning_rate, max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes, min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization, extra_categorical=(CLUSTER_COLUMN,)
        ).fit(self.labeler_.transform(X)[self._columns()], y)
        self.n_iter_ = self.model_.n_iter_
        return self

    def predict(self, X):
        return self.model_.predict(self.labeler_.transform(X)[self._columns()])
