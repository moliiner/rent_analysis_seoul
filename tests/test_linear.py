import numpy as np
import pytest

from rent_model import config
from rent_model.features import build_features, build_target, forbidden_columns
from rent_model.linear import build_pipeline, feature_names, fit_predict, tune
from rent_model.splits import expanding_window_folds, temporal_split


@pytest.fixture(scope='module')
def small(clean_df):
    """Deterministic subsample of the Jeonse track."""
    d = clean_df[clean_df['lease_type'] == config.TRACKS['jeonse']['lease_type']]
    d = d.sample(12_000, random_state=config.RANDOM_SEED).sort_values('contract_date')
    return temporal_split(d)


def matrices(frame, time):
    return build_features(frame, 'jeonse', include_time=time), build_target(frame, 'jeonse')


@pytest.mark.parametrize('time,rich', [(False, False), (True, False), (True, True)])
def test_design_matrix_has_no_forbidden_columns(small, time, rich):
    train, _ = small
    X, y = matrices(train, time)
    pipe = build_pipeline('ridge', time=time, rich=rich, alpha=1.0).fit(X, y)
    names = feature_names(pipe)
    allowed = (set(config.CATEGORICAL_FEATURES) | {'district_x_type', 'log_area', 'floor',
                                                   'floor_missing', 'building_age', 'month_sin',
                                                   'month_cos', 't'})
    assert not forbidden_columns(names)
    assert all(any(n == p or n.startswith(p + '_') or n.startswith(p + ' ') for p in allowed)
               for n in names)
    assert any(n in ('t', 'month_sin', 'month_cos') for n in names) == time
    assert len(names) == pipe.named_steps['model'].coef_.shape[0]


def test_predictions_positive_and_unseen_categories_ok(small):
    train, test = small
    Xtr, ytr = matrices(train, True)
    Xte, _ = matrices(test, True)
    Xte = Xte.copy()
    Xte.iloc[0, Xte.columns.get_loc('district_name')] = 'Nowhere-gu'
    for family, params in [('ols', {}), ('ridge', {'alpha': 1.0}), ('lasso', {'alpha': 1e-3})]:
        _, pred = fit_predict(build_pipeline(family, time=True, **params), Xtr, ytr, Xte)
        assert np.isfinite(pred).all() and (pred > 0).all()


def test_deterministic(small):
    train, test = small
    Xtr, ytr = matrices(train, False)
    Xte, _ = matrices(test, False)
    pipe = build_pipeline('elasticnet', alpha=1e-3, l1_ratio=0.5)
    a = fit_predict(pipe, Xtr, ytr, Xte)[1]
    b = fit_predict(pipe, Xtr, ytr, Xte)[1]
    np.testing.assert_array_equal(a, b)


def test_scaler_fit_on_train_only(small):
    train, _ = small
    Xtr, ytr = matrices(train, False)
    pipe = build_pipeline('ridge', alpha=1.0).fit(Xtr, ytr)
    scaler = pipe.named_steps['prep'].named_transformers_['num'].named_steps['scale']
    assert scaler.n_samples_seen_ == len(train)


def test_tune_returns_sorted_grid_inside_train(small):
    train, _ = small
    X, y = matrices(train, False)
    folds = expanding_window_folds(train)
    best, table = tune('ridge', X, y, folds, grid={'alpha': [1.0, 100.0]})
    assert set(table['alpha']) == {1.0, 100.0} and best['alpha'] == table.loc[0, 'alpha']
    assert table['cv_wape_mean'].is_monotonic_increasing
    for _, valid_pos in folds:
        assert train['contract_date'].iloc[valid_pos].max() < config.TEST_START
