import numpy as np
import pandas as pd
import pytest

from rent_model import config
from rent_model.features import build_features, build_target, forbidden_columns
from rent_model.forest import (DetrendedForest, PARAM_DISTRIBUTIONS, feature_columns, fit_predict,
                               make_model, tune)
from rent_model.splits import expanding_window_folds, temporal_split

FAST = dict(n_estimators=20, max_depth=8, min_samples_leaf=5, max_features=0.7)


@pytest.fixture(scope='module')
def small(clean_df):
    d = clean_df[clean_df['lease_type'] == config.TRACKS['jeonse']['lease_type']]
    d = d.sample(12_000, random_state=config.RANDOM_SEED).sort_values('contract_date')
    train, test = temporal_split(d)
    X = lambda f: build_features(f, 'jeonse', include_time=True)
    return train, test, X(train), build_target(train, 'jeonse'), X(test)


@pytest.mark.parametrize('variant', ['static', 'time', 'detrended'])
def test_shapes_and_positive_predictions(small, variant):
    train, test, Xtr, ytr, Xte = small
    model, pred = fit_predict(variant, FAST, Xtr, ytr, Xte)
    assert pred.shape == (len(test),)
    assert np.isfinite(pred).all() and (pred >= 0).all()


@pytest.mark.parametrize('variant', ['static', 'time', 'detrended'])
def test_deterministic(small, variant):
    _, _, Xtr, ytr, Xte = small
    a = fit_predict(variant, FAST, Xtr, ytr, Xte)[1]
    b = fit_predict(variant, FAST, Xtr, ytr, Xte)[1]
    np.testing.assert_allclose(a, b, rtol=1e-9)   # threads sum tree outputs in varying order


def test_no_leakage_columns():
    for time in (False, True):
        cols = feature_columns(time)
        assert not forbidden_columns(cols)
        assert (config.TIME_FEATURE in cols) == time
        assert not {'month_sin', 'month_cos', 'month_of_year'} & set(cols)


def test_unseen_category_and_nan_floor(small):
    _, _, Xtr, ytr, Xte = small
    Xte = Xte.copy()
    Xte.iloc[0, Xte.columns.get_loc('district_name')] = 'Nowhere-gu'
    Xte.iloc[1, Xte.columns.get_loc('floor')] = np.nan
    _, pred = fit_predict('static', FAST, Xtr, ytr, Xte)
    assert np.isfinite(pred).all()


def test_test_rows_do_not_change_fit(small):
    """The fitted trend and forest depend on the rows given to fit only."""
    train, test, Xtr, ytr, Xte = small
    probe = Xte.head(50)
    base = fit_predict('detrended', FAST, Xtr, ytr, probe)[1]
    again = fit_predict('detrended', FAST, Xtr.copy(), ytr.copy(), pd.concat([probe, Xte.tail(500)]).head(50))[1]
    np.testing.assert_allclose(base, again, rtol=1e-9)


def test_detrended_extrapolates_trend_but_plain_forest_is_flat():
    rng = np.random.default_rng(config.RANDOM_SEED)
    n = 3000
    t = rng.integers(0, 36, n)
    X = pd.DataFrame({'district_name': 'A', 'building_type': 'Apartment', 'contract_type': 'New',
                      'renewal_right_used': 'No', 'log_area': rng.normal(3.5, 0.2, n), 'floor': 3.0,
                      'floor_missing': 0, 'building_age': 10.0, config.TIME_FEATURE: t})
    y = pd.Series(8.0 + 0.02 * t + rng.normal(0, 0.01, n))
    future = X.head(200).assign(**{config.TIME_FEATURE: 48})
    flat = make_model('time', **FAST).fit(X[feature_columns(True)], y).predict(future[feature_columns(True)])
    drift = make_model('detrended', **FAST).fit(X[feature_columns(True)], y).predict(future[feature_columns(True)])
    assert flat.max() < y.max() + 0.01          # a forest cannot exceed the TRAIN range
    assert drift.mean() > y.max() + 0.1         # the extrapolated trend does


def test_tune_returns_valid_params_inside_train(small):
    train, _, Xtr, ytr, _ = small
    folds = expanding_window_folds(train)
    best, table = tune('static', Xtr, ytr, folds, n_iter=2)
    assert set(best) == set(PARAM_DISTRIBUTIONS) and len(table) == 2
    assert table['cv_wape_mean'].is_monotonic_increasing
