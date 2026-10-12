"""Renewal-only study: plausibility rule, models fit on fit-window rows only, isolation from the main protocol."""
import json

import numpy as np
import pandas as pd
import pytest

from rent_model import boosting, config, forest, linear
from rent_model.compare import load_runs
from rent_model.features import build_features, forbidden_columns
from rent_model.renewal import (CarryForward, GroupRatioMedian, PRICE_CAP, RATIO_BOUNDS, SCOPE, flag_previous, previous_price,
                                ratio_wape_scorer)


def toy(track='jeonse', n=8):
    prev_col = {'jeonse': 'previous_deposit_10k_krw', 'wolse': 'previous_monthly_rent_10k_krw'}[track]
    target = config.TRACKS[track]['target']
    cap = PRICE_CAP[track]
    return pd.DataFrame({
        'contract_type': ['Renewal'] * 6 + ['New', 'Renewal'][:n - 6] if n >= 8 else ['Renewal'] * n,
        prev_col: [1000.0, 1000.0, 0.0, np.nan, cap * 2, 1000.0, 1000.0, 1000.0],
        target: [1050.0, 1049.0, 1000.0, 1000.0, 1000.0, 100.0, 1000.0, 6000.0],
        'renewal_right_used': ['Yes', 'No/Not reported'] * 4, 'building_type': ['Apartment'] * 8}).iloc[:n]


@pytest.mark.parametrize('track', ['jeonse', 'wolse'])
def test_plausibility_flags(track):
    d = toy(track)
    f = flag_previous(d, track)
    assert list(f['is_renewal']) == [True] * 6 + [False, True]
    assert list(f['previous_missing']) == [False, False, True, True, False, False, False, False]      # zero and NaN
    assert list(f['above_cap']) == [False, False, False, False, True, False, False, False]
    assert list(f['ratio_outside']) == [False, False, False, False, True, True, False, True]            # ratios far below 0.2 (rows 4 and 5) and 6.0; NaN/0 previous are not ratio-flagged
    assert list(f['input_ok']) == [True, True, False, False, False, True, False, True]                   # renewals with a usable previous price
    assert list(f['fully_ok']) == [True, True, False, False, False, False, False, False]
    assert RATIO_BOUNDS[0] < 1 < RATIO_BOUNDS[1]


def test_rule_uses_the_new_price_only_for_the_ratio_flag():
    d = toy('jeonse')
    changed = d.assign(deposit_10k_krw=d['deposit_10k_krw'] * 3)
    a, b = flag_previous(d, 'jeonse'), flag_previous(changed, 'jeonse')
    assert (a['input_ok'] == b['input_ok']).all()           # the input-only rule does not look at the outcome
    assert not (a['fully_ok'] == b['fully_ok']).all()


def test_carry_forward_and_group_median_use_only_fit_rows():
    fit = pd.DataFrame({'contract_type': 'Renewal', 'previous_deposit_10k_krw': np.full(100, 1000.0),
                        'deposit_10k_krw': np.full(100, 1100.0), 'renewal_right_used': ['Yes'] * 50 + ['No/Not reported'] * 50,
                        'building_type': 'Apartment'})
    fit.loc[fit['renewal_right_used'] == 'Yes', 'deposit_10k_krw'] = 1050.0
    ev = fit.iloc[[0, 99]].assign(deposit_10k_krw=[10_000.0, 10_000.0])           # evaluation prices are irrelevant to the fit
    model = GroupRatioMedian().fit(fit, 'jeonse')
    assert np.allclose(model.predict(ev), [1050.0, 1100.0])
    assert np.allclose(CarryForward().fit(fit, 'jeonse').predict(ev), [1000.0, 1000.0])
    unseen = ev.assign(building_type='Officetel')                                    # group not seen in fit: falls back to the renewal-right median
    assert np.allclose(model.predict(unseen), [1050.0, 1100.0])
    assert model.n_fit_ == 100


def test_ratio_scorer_rebuilds_the_new_price():
    class Perfect:
        def predict(self, X):
            return np.log(np.array([1.05, 0.9]))
    X = pd.DataFrame({'log_prev': np.log([1000.0, 2000.0])})
    assert ratio_wape_scorer(Perfect(), X, np.log([1.05, 0.9])) == pytest.approx(0.0)
    assert ratio_wape_scorer(Perfect(), X, np.log([1.05 * 1.1, 0.9])) < 0


def test_previous_price_selects_the_column_of_the_track():
    d = toy('wolse')
    assert previous_price(d, 'wolse').iloc[0] == 1000.0 and 'previous_monthly_rent_10k_krw' in d.columns


# ---------------------------------------------------------------------------------------------
# Isolation from the main protocol
# ---------------------------------------------------------------------------------------------
def test_main_feature_lists_contain_no_previous_column(clean_df):
    lists = {'config.CATEGORICAL_FEATURES': config.CATEGORICAL_FEATURES, 'config.NUMERIC_FEATURES': config.NUMERIC_FEATURES,
             'config.SEASONAL_FEATURES': config.SEASONAL_FEATURES, 'config.BASE_FEATURES': config.BASE_FEATURES,
             'forest.feature_columns(False)': forest.feature_columns(False), 'forest.feature_columns(True)': forest.feature_columns(True),
             'linear.numeric_columns(False)': linear.numeric_columns(False), 'linear.numeric_columns(True)': linear.numeric_columns(True)}
    for variant in boosting.VARIANTS:
        lists[f'boosting.feature_columns({variant})'] = boosting.feature_columns(variant)
        lists[f'boosting.required_columns({variant})'] = boosting.required_columns(variant)
    for name, columns in lists.items():
        assert not [c for c in columns if 'previous' in c], f'{name} contains a previous_* column'
    for track in config.TRACKS:
        X = build_features(clean_df.sample(2000, random_state=config.RANDOM_SEED), track, include_time=True, include_legal_dong=True)
        assert not [c for c in X.columns if 'previous' in c or c == 'log_prev']
    assert forbidden_columns(['previous_deposit_10k_krw', 'previous_monthly_rent_10k_krw', 'log_area']) == ['previous_deposit_10k_krw', 'previous_monthly_rent_10k_krw']


def test_renewal_only_runs_are_skipped_by_the_main_leaderboard(tmp_path):
    for model_id, scope in [('baseline_global_median', None), ('renewal_carry_forward', SCOPE)]:
        run = {'track': 'jeonse', 'model_id': model_id, 'n': 10, 'wape_pct': 10.0, 'meta': ({'scope': scope} if scope else {})}
        (tmp_path / f'jeonse__{model_id}.json').write_text(json.dumps([run]), encoding='utf-8')
    runs = load_runs(tmp_path)
    assert list(runs['model_id']) == ['baseline_global_median']
