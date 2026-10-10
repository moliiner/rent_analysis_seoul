import numpy as np
import pandas as pd
import pytest

from rent_model import config
from rent_model.features import build_features, build_target, forbidden_columns
from rent_model.splits import temporal_split
from rent_model.unsupervised import (GMM_CATEGORICAL, GMM_NUMERIC, TIER_COLUMN, UNKNOWN_TIER, BoostedWithLabeler,
                                     DistrictTierer, GMMLabeler, GMMMedian, build_ridge_pipeline, district_profiles,
                                     name_tiers, prob_columns, select_gmm, select_n_tiers, tier_table)

FAST_HGB = dict(learning_rate=0.1, max_iter=40, max_leaf_nodes=15, min_samples_leaf=20)


@pytest.fixture(scope='module')
def small(clean_df):
    d = clean_df[clean_df['lease_type'] == config.TRACKS['jeonse']['lease_type']]
    d = d.sample(14_000, random_state=config.RANDOM_SEED).sort_values('contract_date')
    train, test = temporal_split(d)
    X = lambda f: build_features(f, 'jeonse', include_time=True)
    return X(train), build_target(train, 'jeonse'), train['deposit_10k_krw'].to_numpy(), X(test), test


def test_profiles_use_only_given_rows_and_have_expected_columns(small):
    Xtr, ytr, _, Xte, _ = small
    prof = district_profiles(Xtr, ytr)
    assert prof.index.nunique() == Xtr['district_name'].nunique()
    assert {'median_log_target', 'median_log_area', 'median_building_age'} <= set(prof.columns)
    shares = prof[[c for c in prof.columns if c.startswith('share_')]]
    np.testing.assert_allclose(shares.sum(axis=1), 1.0)
    half = Xtr.iloc[: len(Xtr) // 2]
    assert not prof.equals(district_profiles(half, ytr.iloc[: len(Xtr) // 2]))


def test_tierer_fit_on_given_rows_only_and_unseen_district(small):
    Xtr, ytr, _, Xte, _ = small
    tierer = DistrictTierer(n_tiers=4).fit(Xtr, ytr)
    assert set(tierer.mapping_) == set(Xtr['district_name'])
    assert len(set(tierer.mapping_.values())) == 4
    # fitting again on the same TRAIN, with TEST rows around, gives the same map
    again = DistrictTierer(n_tiers=4).fit(Xtr.copy(), ytr.copy())
    assert again.mapping_ == tierer.mapping_
    probe = Xte.head(3).copy()
    probe['district_name'] = ['Nowhere-gu', *probe['district_name'].iloc[1:]]
    out = tierer.transform(probe)
    assert out[TIER_COLUMN].iloc[0] == UNKNOWN_TIER and out[TIER_COLUMN].iloc[1:].isin(set(tierer.mapping_.values())).all()


def test_tiers_are_ordered_by_price_and_selection_rule(small):
    Xtr, ytr, _, _, _ = small
    tierer = DistrictTierer(n_tiers=4).fit(Xtr, ytr)
    table = tier_table(tierer)
    assert table['median_log_target'].is_monotonic_increasing and list(table.index) == ['T0', 'T1', 'T2', 'T3']
    assert len(set(name_tiers(table).values())) == 4
    n, sel = select_n_tiers(tierer.profiles_)
    assert n in (3, 4, 5, 6) and sel.loc[n, 'min_tier_size'] >= 2


def test_gmm_probabilities_and_hard_assignment(small):
    Xtr, _, ytr_orig, Xte, _ = small
    labeler = GMMLabeler(n_components=4, covariance_type='full', n_init=1).fit(Xtr)
    out = labeler.transform(Xte)
    probs = out[prob_columns(4)]
    assert probs.shape == (len(Xte), 4) and np.allclose(probs.sum(axis=1), 1.0)
    comps = labeler.components(Xte)
    assert comps.min() >= 0 and comps.max() < 4
    assert set(GMM_NUMERIC + GMM_CATEGORICAL) == {'log_area', 'floor', 'building_age', 'building_type'}
    scaler = labeler.pre_.named_steps['scale']
    assert scaler.n_samples_seen_ == len(Xtr)
    model = GMMMedian(n_components=4).fit(Xtr, ytr_orig)
    pred = model.predict(Xte)
    assert pred.shape == (len(Xte),) and np.isfinite(pred).all() and (pred > 0).all()


def test_select_gmm_returns_min_bic(small):
    Xtr, _, _, _, _ = small
    cov, k, table = select_gmm(Xtr, k_range=(2, 3), covariance_types=('diag', 'spherical'), n_init=1)
    assert (cov, k) == table['bic'].idxmin() and len(table) == 4 and {'bic', 'aic'} <= set(table.columns)


def test_ridge_pipelines_with_tier_and_gmm(small):
    Xtr, ytr, _, Xte, _ = small
    tier = build_ridge_pipeline(DistrictTierer(4), 0.01, extra_categorical=[TIER_COLUMN]).fit(Xtr, ytr)
    gmm = build_ridge_pipeline(GMMLabeler(3, n_init=1), 0.01, extra_numeric=prob_columns(3)).fit(Xtr, ytr)
    for pipe in (tier, gmm):
        assert len(pipe.predict(Xte)) == len(Xte)
        cols = pipe.named_steps['prep'].transformers_[0][2]
        assert not forbidden_columns(cols) and not set(config.TARGETS) & set(cols)


def test_boosted_with_labeler_tier_and_gmm_deterministic(small):
    Xtr, ytr, _, Xte, _ = small
    a = BoostedWithLabeler(DistrictTierer(4), extra_categorical=[TIER_COLUMN], **FAST_HGB).fit(Xtr, ytr)
    b = BoostedWithLabeler(DistrictTierer(4), extra_categorical=[TIER_COLUMN], **FAST_HGB).fit(Xtr, ytr)
    np.testing.assert_allclose(a.predict(Xte), b.predict(Xte), rtol=1e-9)
    g = BoostedWithLabeler(GMMLabeler(3, n_init=1), extra_numeric=prob_columns(3), **FAST_HGB).fit(Xtr, ytr)
    assert np.isfinite(g.predict(Xte)).all()
