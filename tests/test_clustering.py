import numpy as np
import pandas as pd
import pytest

from rent_model import config
from rent_model.clustering import (CLUSTER_COLUMN, DESCRIPTORS, BoostedWithClusters, ClusterLabeler,
                                   ClusterMedian, StructuralKMeans, build_ridge_kmeans_pipeline,
                                   describe_clusters, name_clusters, select_k)
from rent_model.features import build_features, build_target, forbidden_columns
from rent_model.splits import temporal_split

FAST_HGB = dict(learning_rate=0.1, max_iter=40, max_leaf_nodes=15, min_samples_leaf=20)


@pytest.fixture(scope='module')
def small(clean_df):
    d = clean_df[clean_df['lease_type'] == config.TRACKS['jeonse']['lease_type']]
    d = d.sample(12_000, random_state=config.RANDOM_SEED).sort_values('contract_date')
    train, test = temporal_split(d)
    X = lambda f: build_features(f, 'jeonse', include_time=True)
    return (X(train), build_target(train, 'jeonse'), train['deposit_10k_krw'].to_numpy(),
            X(test), test['deposit_10k_krw'].to_numpy())


def test_descriptors_are_structural_and_target_free():
    assert not forbidden_columns(DESCRIPTORS)
    assert not set(DESCRIPTORS) & set(config.TARGETS)
    assert set(DESCRIPTORS) == {'log_area', 'floor', 'floor_missing', 'building_age', 'building_type'}


def test_fit_uses_train_only_and_predict_labels_unseen_rows(small):
    Xtr, _, _, Xte, _ = small
    model = StructuralKMeans(k=5).fit(Xtr)
    scaler = model.pre_.named_transformers_['num'].named_steps['scale']
    assert scaler.n_samples_seen_ == len(Xtr) and len(model.labels_) == len(Xtr)
    labels = model.predict(Xte)
    assert labels.shape == (len(Xte),) and labels.min() >= 0 and labels.max() < 5
    np.testing.assert_array_equal(labels, model.predict(Xte))
    # TEST rows do not move the centroids: refitting on the same TRAIN gives identical centers
    again = StructuralKMeans(k=5).fit(Xtr.copy())
    np.testing.assert_allclose(model.kmeans_.cluster_centers_, again.kmeans_.cluster_centers_, atol=1e-9)


def test_select_k_uses_train_and_respects_rule(small):
    Xtr, _, _, _, _ = small
    k, table = select_k(Xtr, k_range=range(2, 7), n_seeds=3, silhouette_rows=1000)
    assert list(table.index) == [2, 3, 4, 5, 6] and 4 <= k <= 6
    assert {'inertia', 'silhouette', 'stability_ari', 'eligible'} <= set(table.columns)
    assert (table['stability_ari'] <= 1.0).all() and table['inertia'].is_monotonic_decreasing


def test_cluster_median_predicts_train_median_and_handles_all_rows(small):
    Xtr, _, ytr, Xte, _ = small
    model = ClusterMedian(k=4).fit(Xtr, ytr)
    train_pred = model.predict(Xtr)
    for c in range(4):
        rows = model.clusterer_.labels_ == c
        assert np.allclose(train_pred[rows], np.median(ytr[rows]))
    pred = model.predict(Xte)
    assert pred.shape == (len(Xte),) and np.isfinite(pred).all() and (pred > 0).all()


def test_ridge_kmeans_pipeline_design_matrix(small):
    Xtr, y, _, Xte, _ = small
    pipe = build_ridge_kmeans_pipeline(k=4, alpha=0.01).fit(Xtr, y)
    labeled = pipe.named_steps['label'].transform(Xte)
    assert CLUSTER_COLUMN in labeled and labeled[CLUSTER_COLUMN].nunique() <= 4
    design_columns = pipe.named_steps['prep'].transformers_[0][2]
    assert not forbidden_columns(design_columns) and not set(config.TARGETS) & set(design_columns)
    assert len(pipe.predict(Xte)) == len(Xte)


def test_boosted_with_clusters_runs_and_is_deterministic(small):
    Xtr, y, _, Xte, _ = small
    a = BoostedWithClusters(k=4, **FAST_HGB).fit(Xtr, y).predict(Xte)
    b = BoostedWithClusters(k=4, **FAST_HGB).fit(Xtr, y).predict(Xte)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_profile_and_names(small):
    Xtr, _, ytr, _, _ = small
    model = StructuralKMeans(k=4).fit(Xtr)
    profile = describe_clusters(Xtr, model.labels_, ytr)
    names = name_clusters(profile)
    assert profile['n'].sum() == len(Xtr) and len(set(names.values())) == 4
