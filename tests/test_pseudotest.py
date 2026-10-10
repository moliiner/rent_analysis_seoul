import numpy as np
import pandas as pd
import pytest

from rent_model import config
from rent_model.compare import classify_projection
from rent_model.features import build_features, select_track
from rent_model.projection import FUTURE
from rent_model.pseudotest import (WINDOWS, check_windows, duan_factor, pool, run_windows, select, smeared_prediction,
                                   window_split)
from rent_model.recipes import ModelRecipe, projected_path, recipe_ids

FAST = dict(learning_rate=0.1, max_iter=40, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=0.0)
TRACK = 'jeonse'


@pytest.fixture(scope='module')
def jeonse(clean_df):
    return select_track(clean_df, TRACK).sample(frac=0.3, random_state=config.RANDOM_SEED)


def features(frame):
    return build_features(frame, TRACK, include_time=True, include_legal_dong=True)


class SpyRecipe:
    """Records what reaches the fit and predicts the training mean."""
    track = TRACK

    def fit(self, train, cutoff):
        self.train_max, self.cutoff, self.n = train['contract_date'].max(), cutoff, len(train)
        self.mean = train[config.TRACKS[TRACK]['target']].mean()
        return self

    def predict(self, frame, months=None):
        return np.full(len(frame), self.mean)


def test_windows_are_ordered_and_end_before_the_test_year():
    check_windows()
    assert [w['fit_end'] for w in WINDOWS] == [pd.Timestamp('2023-01-01'), pd.Timestamp('2024-01-01')]
    assert all(pd.Timestamp(f"{w['test_year']}-12-31") < config.TEST_START for w in WINDOWS)
    with pytest.raises(AssertionError):
        check_windows(WINDOWS[::-1])
    with pytest.raises(AssertionError):
        check_windows(({'name': '2025', 'fit_end': pd.Timestamp('2025-01-01'), 'test_year': 2025},))


def test_no_row_at_or_after_the_window_start_is_used_in_fit(jeonse):
    spy = SpyRecipe()
    out = run_windows(spy, jeonse, features)
    assert set(out) == {'2023', '2024'}
    for window in WINDOWS:
        train, test = window_split(jeonse, window)
        assert train['contract_date'].max() < window['fit_end'] <= test['contract_date'].min()
        assert (test['contract_date'].dt.year == window['test_year']).all()
        assert out[window['name']]['n_train'] == len(train) and out[window['name']]['n_test'] == len(test)
    assert out['2023']['n_train'] < out['2024']['n_train']          # expanding window
    assert spy.train_max < pd.Timestamp('2024-01-01')               # the last window never saw 2024
    assert all(pd.DatetimeIndex(r['months']).max() < config.TEST_START for r in out.values())


def test_a_recipe_refuses_rows_at_or_after_the_cutoff(jeonse):
    recipe = ModelRecipe('hgb_static', TRACK, params=FAST)
    with pytest.raises(AssertionError):
        recipe.fit(jeonse, pd.Timestamp('2024-01-01'))               # the sample contains 2024 and 2025 rows


def test_pool_and_select_use_only_pseudo_test_rows(jeonse):
    results = {m: run_windows(SpyRecipe(), jeonse, features) for m in ('hgb_static', 'hgb_time')}
    y, preds, months, districts = pool(results)
    assert pd.DatetimeIndex(months).max() < config.TEST_START and len(y) == len(districts) == len(preds['hgb_static'])
    table, leader, selected = select(results, n_boot=50)
    assert set(table['model_id']) == {'hgb_static', 'hgb_time'} and leader in {'hgb_static', 'hgb_time'}


@pytest.mark.parametrize('model_id', ['baseline_district_type_area', 'hgb_static', 'hgb_static_median', 'hybrid_hgb', 'hybrid_hgb_median'])
def test_recipes_predict_finite_non_negative_values_on_both_windows(jeonse, model_id):
    recipe = ModelRecipe(model_id, TRACK, params={} if model_id.startswith('baseline') else FAST, index_candidate='naive')
    out = run_windows(recipe, jeonse, features)
    for result in out.values():
        assert np.isfinite(result['y_pred']).all() and (result['y_pred'] >= 0).all()
        assert result['metrics']['wape_pct'] > 0
    if recipe.is_hybrid:    # the index vintage stops before the window start
        assert recipe.index_.index.max() < pd.Timestamp('2024-01-01')


def test_median_variants_use_the_median_loss_and_the_ids_are_registered(jeonse):
    train = jeonse[jeonse['contract_date'] < '2024-01-01']
    base = ModelRecipe('hgb_static', TRACK, params=FAST).fit(train, pd.Timestamp('2024-01-01'))
    median = ModelRecipe("hgb_static_median", TRACK, params=FAST).fit(train, pd.Timestamp('2024-01-01'))
    assert base.model_.quantile is None and median.model_.quantile == 0.5
    for suffix in ('hgb_static', 'hgb_time', 'hgb_detrended', 'hgb_legal_dong', 'hybrid_hgb'):
        assert f'{suffix}_median' in recipe_ids()


def test_duan_smearing_factor():
    z = np.log1p(np.array([100.0, 200.0, 400.0]))
    assert duan_factor(z, z) == pytest.approx(1.0)
    np.testing.assert_allclose(smeared_prediction(z, 1.0), np.expm1(z))
    resid_pred = z - np.array([0.1, -0.1, 0.0])
    factor = duan_factor(z, resid_pred)
    assert factor == pytest.approx(np.mean(np.exp([0.1, -0.1, 0.0]))) and factor > 1.0
    assert (smeared_prediction(np.array([-3.0]), 1.0) >= 0).all()


@pytest.mark.parametrize('model_id, index_candidate, expected', [
    ('hybrid_hgb', 'naive', 'level persistence'),      # a naive index forecast is a flat level
    ('hgb_time', None, 'level persistence'),           # trees are flat beyond the last month seen
    ('hgb_static', None, 'level persistence'),
    ('hgb_detrended', None, 'extrapolates'),           # the log-linear trend is added back
    ('hybrid_hgb', 'drift', 'extrapolates'),           # a drift index forecast moves
])
def test_behavioural_extrapolation_classification(jeonse, model_id, index_candidate, expected):
    train = jeonse[jeonse['contract_date'] < config.TEST_START]
    recipe = ModelRecipe(model_id, TRACK, params=FAST, index_candidate=index_candidate).fit(train, config.TEST_START)
    path = projected_path(recipe, jeonse, FUTURE)
    assert len(path) == 36 and (path > 0).all()
    assert classify_projection(path) == expected
