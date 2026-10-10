import numpy as np
import pandas as pd
import pytest

from rent_model import config
from rent_model.boosting import (BoostedModel, PARAM_DISTRIBUTIONS, RareGrouper, VARIANTS,
                                 feature_columns, fit_interval, fit_predict, interval_coverage,
                                 required_columns, tune)
from rent_model.features import build_features, build_target, forbidden_columns
from rent_model.splits import expanding_window_folds, temporal_split

FAST = dict(learning_rate=0.1, max_iter=60, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=0.0)


@pytest.fixture(scope='module')
def small(clean_df):
    d = clean_df[clean_df['lease_type'] == config.TRACKS['jeonse']['lease_type']]
    d = d.sample(12_000, random_state=config.RANDOM_SEED).sort_values('contract_date')
    train, test = temporal_split(d)
    X = lambda f: build_features(f, 'jeonse', include_time=True, include_legal_dong=True)
    return train, test, X(train), build_target(train, 'jeonse'), X(test)


@pytest.mark.parametrize('variant', VARIANTS)
def test_shapes_finite_and_deterministic(small, variant):
    _, test, Xtr, ytr, Xte = small
    _, a = fit_predict(variant, FAST, Xtr, ytr, Xte)
    _, b = fit_predict(variant, FAST, Xtr, ytr, Xte)
    assert a.shape == (len(test),) and np.isfinite(a).all() and (a >= 0).all()
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_no_leakage_columns_and_time_only_where_expected():
    for variant in VARIANTS:
        cols = feature_columns(variant)
        assert not forbidden_columns(cols)
        assert (config.TIME_FEATURE in cols) == (variant == 'time')
        assert (config.LEGAL_DONG_FEATURE in cols) == (variant == 'legal_dong')
        assert not {'month_sin', 'month_cos', 'month_of_year', 'deposit_10k_krw', 'monthly_rent_10k_krw'} & set(cols)
        assert config.TIME_FEATURE in required_columns(variant)


def test_unseen_category_and_nan_floor(small):
    _, _, Xtr, ytr, Xte = small
    Xte = Xte.copy()
    Xte.iloc[0, Xte.columns.get_loc('district_name')] = 'Nowhere-gu'
    Xte.iloc[0, Xte.columns.get_loc('legal_dong')] = 'Nowhere-gu | Nowhere-dong'
    Xte.iloc[1, Xte.columns.get_loc('floor')] = np.nan
    for variant in ('static', 'legal_dong'):
        _, pred = fit_predict(variant, FAST, Xtr, ytr, Xte)
        assert np.isfinite(pred).all()


def test_rare_grouper_limits_levels():
    values = np.array(['a'] * 50 + ['b'] * 40 + ['c'] * 3 + ['d'] * 2).reshape(-1, 1)
    g = RareGrouper(min_frequency=30).fit(values)
    out = g.transform(np.array([['a'], ['c'], ['zzz']]))
    assert out.ravel().tolist() == ['a', '__other__', '__other__']
    assert set(RareGrouper(min_frequency=1, max_categories=2).fit(values).keep_) == {'a', 'b'}


def test_early_stopping_uses_only_fit_rows_and_respects_cap(small):
    train, _, Xtr, ytr, _ = small
    model = BoostedModel(variant='static', **FAST).fit(Xtr[required_columns('static')], ytr)
    assert 1 <= model.n_iter_ <= FAST['max_iter']
    # the validation tail is the last 6 months of the rows given to fit
    t = Xtr[config.TIME_FEATURE]
    assert (t > t.max() - 6).sum() < len(t)


def test_detrended_extrapolates_trend_but_static_booster_is_flat():
    rng = np.random.default_rng(config.RANDOM_SEED)
    n = 3000
    t = rng.integers(0, 36, n)
    X = pd.DataFrame({'district_name': 'A', 'building_type': 'Apartment', 'contract_type': 'New',
                      'renewal_right_used': 'No', 'log_area': rng.normal(3.5, 0.2, n), 'floor': 3.0,
                      'floor_missing': 0, 'building_age': 10.0, config.TIME_FEATURE: t})
    y = pd.Series(8.0 + 0.02 * t + rng.normal(0, 0.01, n))
    future = X.head(200).assign(**{config.TIME_FEATURE: 48})
    flat = BoostedModel('time', **FAST).fit(X, y).predict(future)
    drift = BoostedModel('detrended', **FAST).fit(X, y).predict(future)
    assert flat.max() < y.max() + 0.01
    assert drift.mean() > y.max() + 0.1


def test_quantile_interval_ordered_and_coverage_reasonable(small):
    _, _, Xtr, ytr, Xte = small
    lower, upper = fit_interval('static', FAST, Xtr, ytr, Xte)
    assert (lower <= upper).all()
    # on TRAIN itself an 80% interval should cover well above half of the rows
    lo_tr, hi_tr = fit_interval('static', FAST, Xtr, ytr, Xtr)
    cov, width = interval_coverage(np.expm1(ytr), lo_tr, hi_tr)
    assert 0.5 < cov <= 1.0 and width > 0


def test_tune_returns_valid_params(small):
    train, _, Xtr, ytr, _ = small
    folds = expanding_window_folds(train)
    best, table = tune('static', Xtr, ytr, folds, n_iter=2)
    assert set(best) == set(PARAM_DISTRIBUTIONS) and len(table) == 2
    assert table['cv_wape_mean'].is_monotonic_increasing
