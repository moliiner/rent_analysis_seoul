"""Neighbourhood-aware trend models: grouped legal_dong, recalibration factor, repeat-unit flag and tiers."""
import numpy as np
import pandas as pd
import pytest

from rent_model import config
from rent_model.boosting import BoostedModel, EXTENDED_PARAM_DISTRIBUTIONS, adjusted_wape_scorer, required_columns, tune_extended
from rent_model.compare import complexity_of, extrapolation_kind, family_of
from rent_model.diagnostics import error_table, lot_available, price_tier, repeat_unit_flag, unit_key
from rent_model.features import MAX_NEIGHBOURHOOD_LEVELS, NeighbourhoodGrouper, build_features, build_target, select_track
from rent_model.pseudotest import WINDOWS, run_windows, window_split
from rent_model.recipes import ModelRecipe, all_recipe_ids, ld_recipe_ids, projected_path
from rent_model.splits import window_folds

FAST = dict(learning_rate=0.1, max_iter=40, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=0.0, min_frequency=30)


# ------------------------------------------------------------------ grouping
def test_grouping_uses_only_fit_rows_and_maps_unseen_levels_to_the_district_other():
    fit = pd.Series(['A | x'] * 40 + ['A | y'] * 3 + ['B | z'] * 35 + ['B | w'] * 2)
    grouper = NeighbourhoodGrouper(min_count=30).fit(fit.to_numpy().reshape(-1, 1))
    assert grouper.kept_ == {'A | x', 'B | z'}
    # rare fit levels and levels never seen in fit go to the OTHER level of their own district
    out = grouper.group(np.array(['A | x', 'A | y', 'A | new', 'B | w', 'C | q']).reshape(-1, 1))
    assert list(out) == ['A | x', 'A | OTHER', 'A | OTHER', 'B | OTHER', 'OTHER']
    codes = grouper.transform(np.array(['A | x', 'A | new', 'C | q']).reshape(-1, 1)).ravel()
    assert codes[0] == grouper.codes_['A | x'] and codes[1] == grouper.codes_['A | OTHER'] and codes[2] == len(grouper.levels_)
    # evaluation rows cannot change the levels: a second fit on the same fit rows is identical
    again = NeighbourhoodGrouper(min_count=30).fit(fit.to_numpy().reshape(-1, 1))
    assert again.levels_ == grouper.levels_ and grouper.n_fit_rows_ == len(fit)


def test_grouping_caps_the_number_of_levels():
    fit = pd.Series([f'D{i % 5} | n{i}' for i in range(600) for _ in range(3)])
    grouper = NeighbourhoodGrouper(min_count=2, max_levels=20).fit(fit.to_numpy().reshape(-1, 1))
    assert len(grouper.levels_) <= 20 and len(grouper.kept_) == 20 - 5
    assert len(NeighbourhoodGrouper().fit(fit.to_numpy().reshape(-1, 1)).levels_) <= MAX_NEIGHBOURHOOD_LEVELS


def test_window_folds_work_inside_short_fit_windows(clean_df):
    d = select_track(clean_df, 'jeonse')
    for window, first in zip(WINDOWS, (6, 12)):
        train, _ = window_split(d, window)
        folds = window_folds(train)
        assert len(folds) == 3
        months = train['contract_date'].dt.to_period('M').astype('int64').to_numpy()
        for train_pos, valid_pos in folds:
            assert months[train_pos].max() < months[valid_pos].min()
        assert (months[folds[0][0]].max() - months.min() + 1) == first


# ------------------------------------------------------------------ boosting
@pytest.fixture(scope='module')
def small(clean_df):
    d = select_track(clean_df, 'jeonse').sample(12_000, random_state=config.RANDOM_SEED).sort_values('contract_date')
    return d


def test_boosted_model_with_neighbourhood_handles_unseen_levels(small):
    train = small[small['contract_date'] < '2024-01-01']
    test = small[small['contract_date'] >= '2024-01-01'].copy()
    X = lambda f: build_features(f, 'jeonse', include_time=True, include_legal_dong=True)
    cols = required_columns('detrended', neighbourhood=True)
    assert config.LEGAL_DONG_FEATURE in cols
    model = BoostedModel(variant='detrended', neighbourhood=True, quantile=0.5, **FAST).fit(X(train)[cols], build_target(train, 'jeonse'))
    Xt = X(test)[cols].copy()
    Xt.iloc[0, Xt.columns.get_loc(config.LEGAL_DONG_FEATURE)] = f'{test.iloc[0]["district_name"]} | never seen'
    Xt.iloc[1, Xt.columns.get_loc(config.LEGAL_DONG_FEATURE)] = 'Nowhere-gu | never seen'
    pred = model.predict(Xt)
    assert np.isfinite(pred).all() and len(pred) == len(test)


def test_extended_grid_goes_beyond_127_leaves_and_tunes_the_minimum_count():
    assert max(EXTENDED_PARAM_DISTRIBUTIONS['max_leaf_nodes']) > 127 and 'min_frequency' in EXTENDED_PARAM_DISTRIBUTIONS


def test_tune_extended_scores_a_hybrid_structure_model_on_its_own_target(small):
    train = small[small['contract_date'] < '2024-01-01']
    X = build_features(train, 'jeonse', include_time=True, include_legal_dong=True)
    y = build_target(train, 'jeonse')
    X = X.assign(adj_log=0.0)
    grid = {'learning_rate': [0.1], 'max_iter': [30], 'max_leaf_nodes': [15, 31], 'min_frequency': [10, 30]}
    best, table, n = tune_extended('static', X, y, window_folds(train), n_iter=2, quantile=0.5, neighbourhood=True,
                                   scorer=adjusted_wape_scorer, distributions=grid)
    assert n == len(train) and set(best) == set(grid) and len(table) == 2 and (table['cv_wape_mean'] > 0).all()


# ------------------------------------------------------------------ recipes
def test_ld_ids_are_registered_and_classified():
    assert set(ld_recipe_ids()) <= set(all_recipe_ids())
    for model_id in ld_recipe_ids():
        assert complexity_of(model_id) >= 14 and family_of(model_id) in ('boosting', 'time-series/hybrid')
    assert complexity_of('hgb_detrended_ld_median_recal') == complexity_of('hgb_detrended_ld_median') + 1
    assert extrapolation_kind('hgb_detrended_ld_median_recal')[1] and extrapolation_kind('hybrid_hgb_ld')[1]


@pytest.mark.parametrize('model_id', ['hgb_detrended_ld_median_recal', 'hybrid_hgb_ld_median_recal'])
def test_recalibration_factor_uses_only_fit_window_rows(small, model_id):
    window = WINDOWS[1]
    train, test = window_split(small, window)
    recipe = ModelRecipe(model_id, 'jeonse', params=FAST, index_candidate='naive').fit(train, window['fit_end'])
    assert recipe.recal and np.isfinite(recipe.recal_factor_) and recipe.recal_factor_ > 0
    assert pd.Timestamp(recipe.recal_info_['valid_start']) < window['fit_end']
    plain = ModelRecipe(model_id.removesuffix('_recal'), 'jeonse', params=FAST, index_candidate='naive').fit(train, window['fit_end'])
    X = build_features(test, 'jeonse', include_time=True, include_legal_dong=True)
    assert np.allclose(recipe.predict(X, test['contract_month']), plain.predict(X, test['contract_month']) * recipe.recal_factor_)
    # the factor is a function of the fit window only: dropping the evaluation rows changes nothing
    again = ModelRecipe(model_id, 'jeonse', params=FAST, index_candidate='naive').fit(train.copy(), window['fit_end'])
    assert again.recal_factor_ == pytest.approx(recipe.recal_factor_)


def test_recipe_per_window_and_projection_behaviour(small):
    recipes = {w['name']: ModelRecipe('hgb_detrended_ld_median', 'jeonse', params=FAST) for w in WINDOWS}
    out = run_windows(recipes, small, lambda f: build_features(f, 'jeonse', include_time=True, include_legal_dong=True))
    assert set(out) == {'2023', '2024'} and all(np.isfinite(r['y_pred']).all() for r in out.values())
    path = projected_path(recipes['2024'], small, pd.date_range('2026-01-01', '2028-12-01', freq='MS'))
    annual = path.groupby(path.index.year).mean()
    assert annual.iloc[-1] != annual.iloc[0]           # the trend is added back: the path is not constant


# ------------------------------------------------------------------ diagnostics
def test_repeat_flag_looks_only_at_fit_rows_and_marks_known_units():
    cols = dict(district_name='D', legal_dong_name='n', main_lot_number=1.0, sub_lot_number=0.0, building_type='Apartment',
                leased_area_sqm=59.9, floor=3.0, year_built=2000)
    fit = pd.DataFrame([cols, {**cols, 'floor': 4.0}])
    ev = pd.DataFrame([cols, {**cols, 'floor': 9.0}, {**cols, 'main_lot_number': np.nan}])
    assert list(repeat_unit_flag(fit, ev)) == [True, False, False]
    assert list(lot_available(ev)) == [True, True, False]
    assert unit_key(ev).nunique() == 3


def test_price_tiers_come_from_predictions_and_error_table():
    pred = np.arange(1.0, 301.0)
    tiers = price_tier(pred)
    assert tiers.value_counts().to_dict() == {'T1 low': 100, 'T2 mid': 100, 'T3 high': 100}
    y = pred * 1.1
    table = error_table(tiers, y, pred)
    assert list(table.index) == ['T1 low', 'T2 mid', 'T3 high']
    assert table['mdape_pct'].between(9.0, 9.2).all() and (table['mae'].diff().dropna() > 0).all()
