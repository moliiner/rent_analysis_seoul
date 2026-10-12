"""Split conformal intervals: finite-sample quantile, coverage on exchangeable data, time-ordered calibration."""
import math

import numpy as np
import pandas as pd
import pytest

from rent_model import config
from rent_model.conformal import (CALIBRATION_MONTHS, calibrate, conformal_quantile, coverage, coverage_by_group,
                                  difficulty_by_group, intervals, split_calibration)
from rent_model.features import select_track
from rent_model.pseudotest import WINDOWS, window_split
from rent_model.recipes import ModelRecipe

FAST = dict(learning_rate=0.1, max_iter=40, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=0.0, min_frequency=30)


def test_quantile_is_the_ceil_n_plus_one_times_level_th_smallest_score():
    scores = np.arange(1.0, 10.0)                       # n = 9
    assert conformal_quantile(scores, 0.8) == 8.0       # ceil(10 * 0.8) = 8
    assert conformal_quantile(scores[::-1], 0.8) == 8.0     # order does not matter
    assert conformal_quantile(scores, 0.5) == 5.0       # ceil(10 * 0.5) = 5
    assert conformal_quantile(scores, 0.95) == math.inf     # ceil(9.5 + 0.5) = 10 > n: not enough scores for that level
    assert conformal_quantile(np.arange(1.0, 20.0), 0.95) == 19.0       # n = 19, rank ceil(20 * 0.95) = 19
    with pytest.raises(ValueError):
        conformal_quantile([], 0.8)


@pytest.mark.parametrize('level', [0.8, 0.95])
def test_coverage_on_exchangeable_data_is_close_to_nominal(level):
    rng = np.random.default_rng(0)
    covered = []
    for _ in range(40):
        calibration, test = np.abs(rng.standard_t(5, 400)), np.abs(rng.standard_t(5, 4000))
        covered.append(np.mean(test <= conformal_quantile(calibration, level)))
    assert level - 0.005 <= np.mean(covered) <= level + 0.02        # finite-sample guarantee: at least nominal (on average)


def test_scaled_scores_even_out_the_coverage_across_groups_with_different_noise():
    rng = np.random.default_rng(1)
    group = rng.integers(0, 2, 20_000)
    sigma = np.where(group == 0, 0.1, 0.5)
    residual = rng.normal(0, sigma)
    cal, ev = slice(0, 10_000), slice(10_000, None)
    q_plain = conformal_quantile(np.abs(residual[cal]), 0.8)
    q_scaled = conformal_quantile(np.abs(residual[cal]) / sigma[cal], 0.8)
    plain = [np.mean(np.abs(residual[ev])[group[ev] == g] <= q_plain) for g in (0, 1)]
    scaled = [np.mean(np.abs(residual[ev])[group[ev] == g] <= q_scaled * sigma[ev][group[ev] == g]) for g in (0, 1)]
    assert max(plain) - min(plain) > 0.2 and max(scaled) - min(scaled) < 0.03


def test_calibration_rows_are_the_most_recent_and_never_overlap_with_the_evaluation_rows(clean_df):
    d = select_track(clean_df, 'jeonse')
    for window in WINDOWS:
        fit, evaluation = window_split(d, window)
        proper, calibration = split_calibration(fit)
        assert calibration['contract_month'].nunique() == CALIBRATION_MONTHS
        assert proper['contract_date'].max() < calibration['contract_date'].min()
        assert calibration['contract_date'].max() < window['fit_end'] <= evaluation['contract_date'].min()
        assert not (set(calibration.index) & set(evaluation.index)) and not (set(proper.index) & set(calibration.index))


def test_intervals_are_ordered_non_negative_and_back_transformed_with_expm1():
    prediction = np.array([10.0, 100.0, 1000.0])
    calibration = type('C', (), {'levels': (0.8, 0.95), 'q': {0.8: 0.2, 0.95: 0.5}, 'q_scaled': {0.8: 2.0, 0.95: 4.0},
                                 'sigma': {'A': 0.1, 'B': 0.3, None: 0.2}})()
    out = intervals(prediction, calibration)
    assert np.allclose(out[0.8][0], np.expm1(np.log1p(prediction) - 0.2)) and np.allclose(out[0.8][1], np.expm1(np.log1p(prediction) + 0.2))
    assert (out[0.95][0] <= out[0.8][0]).all() and (out[0.95][1] >= out[0.8][1]).all() and (out[0.95][0] >= 0).all()
    scaled = intervals(prediction, calibration, groups=['A', 'B', 'C'], scaled=True)
    assert np.allclose(scaled[0.8][1], np.expm1(np.log1p(prediction) + 2.0 * np.array([0.1, 0.3, 0.2])))


def test_coverage_helpers():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert coverage(y, y - 0.5, y + 0.5) == (1.0, 1.0)
    assert coverage(y, y + 0.5, y + 1.0)[0] == 0.0
    table = coverage_by_group(['a', 'a', 'b', 'b'], y, np.array([0, 2.5, 0, 0.0]), np.array([1, 3.0, 3.5, 3.5]))
    assert table.loc['a', 'coverage'] == 0.5 and table.loc['b', 'coverage'] == 0.5


def test_difficulty_uses_only_the_given_rows(clean_df):
    d = select_track(clean_df, 'jeonse').sample(5000, random_state=config.RANDOM_SEED)
    sigma = difficulty_by_group(d, np.log1p(d['deposit_10k_krw']))
    assert set(d['building_type']) <= set(sigma) and None in sigma and all(v > 0 for v in sigma.values())


def test_calibrate_end_to_end_uses_only_the_fit_window(clean_df):
    d = select_track(clean_df, 'jeonse').sample(12_000, random_state=config.RANDOM_SEED).sort_values('contract_date')
    window = WINDOWS[1]
    fit, evaluation = window_split(d, window)
    calibration = calibrate(lambda: ModelRecipe('hgb_detrended_ld_median', 'jeonse', params=FAST), fit, 'jeonse')
    assert pd.Timestamp(calibration.last) < window['fit_end'] and calibration.n > 0
    assert all(np.isfinite(calibration.q[lv]) and calibration.q[lv] > 0 for lv in calibration.levels)
    assert calibration.q[0.95] >= calibration.q[0.8] and calibration.q_scaled[0.95] >= calibration.q_scaled[0.8]
