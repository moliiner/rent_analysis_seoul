import json

import numpy as np
import pandas as pd
import pytest

from rent_model.compare import (FAMILIES, _base_id, annual_cash_cost, area_buckets, bootstrap_best_share, classify_projection,
                                complexity_of, error_by_group, extrapolation_kind, family_of, family_table, interval_multipliers,
                                load_runs, one_se_selection, profile_frame, ranking, representative_profile)


def write_run(directory, track, model_id, wape, valid=None, bias=0.0, n=100):
    meta = {} if valid is None else {'valid': valid}
    row = {'track': track, 'model_id': model_id, 'n': n, 'wape_pct': wape, 'mae': 1.0, 'mdape_pct': 1.0,
           'median_bias_pct': bias, 'aggregate_bias_pct': bias, 'rmse_log': 0.1, 'r2_log': 0.9,
           'mape_dq_median_pct': 1.0, 'mape_dq_max_pct': 2.0, 'meta': meta}
    (directory / f'{track}__{model_id}.json').write_text(json.dumps([row]), encoding='utf-8')


def test_ranking_excludes_invalid_runs(tmp_path):
    write_run(tmp_path, 'jeonse', 'ridge_static', 24.0)
    write_run(tmp_path, 'jeonse', 'hybrid_hgb', 17.4, valid=True)
    write_run(tmp_path, 'jeonse', 'hybrid_hgb_oracle', 10.0, valid=False)   # better, but invalid
    (tmp_path / 'ts_index_jeonse.json').write_text('{}', encoding='utf-8')   # not a model run
    runs = load_runs(tmp_path)
    valid, invalid = ranking(runs)
    assert set(runs['model_id']) == {'ridge_static', 'hybrid_hgb', 'hybrid_hgb_oracle'}
    assert list(valid['model_id']) == ['hybrid_hgb', 'ridge_static'] and list(invalid['model_id']) == ['hybrid_hgb_oracle']
    assert valid['rank'].tolist() == [1, 2]
    assert family_table(valid).set_index('family').loc['time-series/hybrid', 'best_wape'] == 17.4


def test_families_and_extrapolation_labels():
    expected = {'baseline_district': 'baseline', 'ridge_time': 'linear', 'ridge_time_rich': 'linear', 'rf_detrended': 'forest',
                'hgb_legal_dong': 'boosting', 'hgb_time_gmm': 'unsupervised-assisted', 'ridge_time_kmeans': 'unsupervised-assisted',
                'kmeans_median': 'pure unsupervised', 'gmm_median': 'pure unsupervised', 'hybrid_hgb_oracle': 'time-series/hybrid'}
    for model_id, family in expected.items():
        assert family_of(model_id) == family and family in FAMILIES
    assert extrapolation_kind('hybrid_hgb') == ('forecasted index', True)
    assert extrapolation_kind('hgb_detrended')[1] and extrapolation_kind('ridge_time_gmm')[1]
    assert not extrapolation_kind('hgb_time')[1] and not extrapolation_kind('hgb_legal_dong')[1]
    assert not extrapolation_kind('baseline_drift')[1] and not extrapolation_kind('rf_time')[1]
    assert complexity_of('hybrid_hgb_oracle') == complexity_of('hybrid_hgb')


def toy_problem(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    y = rng.lognormal(3, 0.5, n)
    preds = {'leader': y * rng.lognormal(0, 0.10, n), 'tie_complex': y * rng.lognormal(0, 0.10, n),
             'tie_simple': y * rng.lognormal(0, 0.10, n), 'bad': y * rng.lognormal(0, 0.40, n)}
    runs = pd.DataFrame({'model_id': list(preds), 'complexity': [5, 9, 1, 0], 'aggregate_bias_pct': [0.0, 0.0, 0.0, 0.0]})
    return y, preds, runs


def test_one_se_tie_logic_prefers_simpler_model_among_ties():
    y, preds, runs = toy_problem()
    table, leader, selected = one_se_selection(y, preds, runs, n_boot=200)
    by_model = table.set_index('model_id')
    assert not bool(by_model.loc['bad', 'tie_with_leader'])
    ties = set(table[table['tie_with_leader']]['model_id'])
    assert leader in ties and selected in ties
    complexity = runs.set_index('model_id')['complexity']
    assert complexity[selected] == complexity[list(ties)].min()


def test_one_se_clear_leader_and_bias_tiebreak():
    rng = np.random.default_rng(1)
    y = rng.lognormal(3, 0.5, 3000)
    preds = {'a': y * 1.0, 'b': y * rng.lognormal(0, 0.3, 3000)}
    runs = pd.DataFrame({'model_id': ['a', 'b'], 'complexity': [9, 0], 'aggregate_bias_pct': [0.0, 0.0]})
    _, leader, selected = one_se_selection(y, preds, runs, n_boot=100)
    assert leader == 'a' and selected == 'a'        # b is simpler but clearly worse: not a tie
    preds2 = {'a': y * 1.01, 'b': y * 1.01}          # identical error: equal complexity, smaller |bias| wins
    runs2 = pd.DataFrame({'model_id': ['a', 'b'], 'complexity': [3, 3], 'aggregate_bias_pct': [-5.0, -1.0]})
    _, _, sel2 = one_se_selection(y, preds2, runs2, n_boot=50)
    assert sel2 == 'b'


def test_bootstrap_best_share_sums_to_one():
    y, preds, _ = toy_problem()
    share = bootstrap_best_share(y, preds, n_boot=200)
    assert share.sum() == pytest.approx(1.0) and share['bad'] == 0.0 and (share >= 0).all()


def test_error_by_group_and_area_buckets():
    y = np.array([100.0, 100.0, 200.0, 200.0])
    p = np.array([110.0, 90.0, 200.0, 220.0])
    t = error_by_group(['a', 'a', 'b', 'b'], y, p)
    assert t.loc['a', 'wape_pct'] == pytest.approx(10.0) and t.loc['b', 'wape_pct'] == pytest.approx(5.0)
    assert t['n'].tolist() == [2, 2]
    buckets = area_buckets(np.arange(1, 101, dtype=float), np.array([1.0, 50.0, 100.0, 500.0]))
    assert buckets.iloc[0].startswith('A1') and buckets.iloc[-1].startswith('A5') and buckets.nunique() <= 5


def test_representative_profile_and_frame():
    n = 60
    df = pd.DataFrame({'district_name': 'A-gu', 'building_type': 'Apartment', 'floor': np.arange(n) % 10 + 1.0,
                       'contract_date': pd.Timestamp('2024-05-01'), 'year_built': 2004,
                       'contract_type': ['New'] * 40 + ['Renewal'] * 20})
    prof, n_rows = representative_profile(df, 'A-gu', 'Apartment', 60.0)
    assert n_rows == 60 and prof['building_age'] == 20 and prof['contract_type'] == 'New'
    assert prof['log_area'] == pytest.approx(np.log(60))
    frame = profile_frame(prof, pd.date_range('2026-01-01', periods=3, freq='MS'))
    assert frame['t'].tolist() == [48, 49, 50] and len(frame) == 3
    rare, n_small = representative_profile(df, 'B-gu', 'Apartment', 60.0)   # unseen group: type-wide medians
    assert n_small == 0 and rare['building_age'] == 20


def test_annual_cost_and_multipliers():
    months = pd.date_range('2026-01-01', periods=24, freq='MS')
    values = pd.Series([50.0] * 12 + [60.0] * 12, index=months)
    assert annual_cash_cost(values, 'wolse').tolist() == [600.0, 720.0]
    assert annual_cash_cost(values, 'jeonse').tolist() == [50.0, 60.0]
    fc = pd.DataFrame({'mean': [100.0, 100.0], 'lo80': [90.0, 80.0], 'hi80': [110.0, 125.0]})
    lo, hi = interval_multipliers(fc)
    assert lo.tolist() == [0.9, 0.8] and hi.tolist() == [1.1, 1.25]


# ---------------------------------------------------------------------------------------------
# Robust selection protocol (CLAUDE.md rule 8, amended)
# ---------------------------------------------------------------------------------------------
def month_driven_models():
    """Two models whose WAPE difference is a month-level effect: rows say 'different', months say 'tie'."""
    months = np.repeat(np.arange(12), 500)
    y = np.full(months.size, 100.0)
    preds = {'a_complex': y + np.where(months < 7, 5.0, 7.0), 'b_simple': y + 6.0}
    runs = pd.DataFrame({'model_id': list(preds), 'complexity': [9, 1], 'aggregate_bias_pct': [0.0, 0.0]})
    districts = np.tile(np.arange(5), months.size // 5 + 1)[: months.size]
    return y, preds, runs, months, districts


def test_selection_uses_the_most_conservative_standard_error():
    y, preds, runs, months, districts = month_driven_models()
    row_table, row_leader, row_selected = one_se_selection(y, preds, runs, n_boot=300)
    block_table, block_leader, block_selected = one_se_selection(y, preds, runs, n_boot=300, months=months, districts=districts)
    assert row_leader == block_leader == 'a_complex'
    assert not bool(row_table.set_index('model_id').loc['b_simple', 'tie_with_leader'])      # rows: "clearly different"
    assert row_selected == 'a_complex'
    b = block_table.set_index('model_id').loc['b_simple']
    assert bool(b['tie_with_leader']) and b['se_source'] == 'month'                          # month blocks: a tie
    assert block_selected == 'b_simple'                                                      # tie -> fewer parameters
    assert b['se_diff'] > row_table.set_index('model_id').loc['b_simple', 'se_diff']


def test_classify_projection_is_behavioural():
    months = pd.date_range('2026-01-01', '2028-12-01', freq='MS')
    flat = pd.Series(100.0, index=months)
    seasonal = pd.Series(100.0 + 3 * np.sin(2 * np.pi * months.month / 12), index=months)
    trend = pd.Series(100.0 * 1.01 ** (np.arange(36) / 12), index=months)
    assert classify_projection(flat) == 'level persistence'
    assert classify_projection(seasonal) == 'level persistence'      # seasonal wiggles average out over a year
    assert classify_projection(trend) == 'extrapolates'
    assert classify_projection(flat * 1.0005) == 'level persistence'  # below the 0.1% tolerance


def test_median_variants_inherit_the_label_family_and_complexity():
    assert family_of('hgb_static_median') == 'boosting' and family_of('hybrid_hgb_median') == 'time-series/hybrid'
    assert complexity_of('hgb_detrended_median') == complexity_of('hgb_detrended')
    assert extrapolation_kind('hgb_detrended_median') == extrapolation_kind('hgb_detrended')
    assert extrapolation_kind('hgb_time_median')[1] is False and extrapolation_kind('hybrid_hgb_median')[1] is True


def test_behaviour_is_loaded_from_the_pseudotest_report(tmp_path):
    write_run(tmp_path, 'jeonse', 'hybrid_hgb', 17.4)
    write_run(tmp_path, 'jeonse', 'hgb_static', 18.6)
    write_run(tmp_path, 'jeonse', 'ridge_time', 24.0)
    (tmp_path / 'pseudotest_jeonse.json').write_text(json.dumps({'candidates': {
        'hybrid_hgb': {'behaviour': 'level persistence'}, 'ridge_time': {'behaviour': 'extrapolates'}}}), encoding='utf-8')
    runs = load_runs(tmp_path).set_index('model_id')
    assert runs.loc['hybrid_hgb', 'behaviour'] == 'level persistence' and not runs.loc['hybrid_hgb', 'can_extrapolate']
    assert runs.loc['ridge_time', 'can_extrapolate'] and runs.loc['hgb_static', 'behaviour'] == 'not evaluated'
    assert bool(runs.loc['hybrid_hgb', 'extrapolation_label_ok'])      # the label alone would have said "extrapolates"


def test_bootstrap_best_share_with_month_blocks_is_less_decisive_than_rows():
    y, preds, runs, months, districts = month_driven_models()
    rows = bootstrap_best_share(y, preds, n_boot=300)
    blocks = bootstrap_best_share(y, preds, n_boot=300, blocks=months)
    assert rows.sum() == pytest.approx(1.0) and blocks.sum() == pytest.approx(1.0)
    assert rows['a_complex'] > 0.99                    # rows: A is "always" the best
    assert 0.5 < blocks['a_complex'] < 0.95            # months: A is only usually the best


def test_native_median_ids_are_not_treated_as_median_loss_variants():
    for model_id in ('baseline_global_median', 'baseline_train_median', 'kmeans_median', 'gmm_median'):
        assert _base_id(model_id) == model_id
        assert complexity_of(model_id) >= 0
    assert _base_id('hybrid_hgb_median_oracle') == 'hybrid_hgb' and _base_id('hgb_time_median') == 'hgb_time'
    assert family_of('kmeans_median') == 'pure unsupervised' and extrapolation_kind('gmm_median') == ('none', False)
