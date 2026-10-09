import json

import numpy as np
import pandas as pd
import pytest

from rent_model import metrics as m
from rent_model.io import save_run

Y = np.array([100.0, 200.0, 400.0])
P = np.array([110.0, 150.0, 400.0])


def test_point_metrics_hand_computed():
    assert m.wape(Y, P) == pytest.approx(60 / 700 * 100)
    assert m.mae(Y, P) == pytest.approx(20.0)
    assert m.mdape(Y, P) == pytest.approx(10.0)
    assert m.median_bias_pct(Y, P) == pytest.approx(0.0)
    assert m.aggregate_bias_pct(Y, P) == pytest.approx(-40 / 700 * 100)


def test_log_metrics_hand_computed():
    y = np.expm1([1.0, 2.0, 3.0])
    p = np.expm1([2.0, 2.0, 1.0])
    assert m.rmse_log(y, p) == pytest.approx(np.sqrt(5 / 3))
    assert m.r2_log(y, p) == pytest.approx(-1.5)
    assert m.r2_log(y, y) == pytest.approx(1.0)


def test_zero_targets_are_skipped_in_percentage_errors():
    y = np.array([0.0, 100.0])
    p = np.array([5.0, 90.0])
    assert m.mdape(y, p) == pytest.approx(10.0)
    assert m.wape(y, p) == pytest.approx(15.0)


def test_mape_by_district_quarter():
    context = pd.DataFrame({
        'district_name': ['A', 'A', 'A', 'B'],
        'contract_date': pd.to_datetime(['2025-01-10', '2025-02-10', '2025-04-10', '2025-01-10']),
    })
    y = np.array([100.0, 100.0, 100.0, 200.0])
    p = np.array([110.0, 130.0, 100.0, 180.0])
    table = m.mape_by_district_quarter(context, y, p).set_index(['district_name', 'quarter'])
    assert table.loc[('A', '2025Q1'), 'mape_pct'] == pytest.approx(20.0)
    assert table.loc[('A', '2025Q1'), 'n'] == 2
    assert table.loc[('A', '2025Q2'), 'mape_pct'] == pytest.approx(0.0)
    assert table.loc[('B', '2025Q1'), 'mape_pct'] == pytest.approx(10.0)


def test_bootstrap_identical_predictions_and_reproducibility():
    rng = np.random.default_rng(0)
    y = rng.uniform(50, 500, 400)
    a = y * rng.uniform(0.8, 1.2, 400)
    same = m.paired_bootstrap_wape_diff(y, a, a, n=200)
    assert same['diff'] == 0 and same['se'] == 0 and same['ci_low'] == 0 and same['ci_high'] == 0

    b = y * rng.uniform(0.5, 1.5, 400)
    first = m.paired_bootstrap_wape_diff(y, a, b, n=200)
    again = m.paired_bootstrap_wape_diff(y, a, b, n=200)
    assert first == again
    assert first['diff'] < 0 and first['ci_high'] < 0 and first['prob_a_better'] > 0.99
    assert first['se'] > 0


def test_save_run_writes_predictions_and_appends_metrics(tmp_path):
    context = pd.DataFrame({
        'district_name': ['A'] * 3,
        'contract_date': pd.to_datetime(['2025-01-10', '2025-02-10', '2025-03-10']),
    })
    for _ in range(2):
        pred_path, metrics_path = save_run('wolse', 'toy_model', Y, P, meta={'note': 'toy'},
                                           context=context, reports_dir=tmp_path)
    assert pred_path.name == 'wolse__toy_model.parquet'
    assert metrics_path.name == 'wolse__toy_model.json'
    stored = pd.read_parquet(pred_path)
    assert list(stored['y_pred']) == list(P)
    rows = json.loads(metrics_path.read_text(encoding='utf-8'))
    assert len(rows) == 2
    assert rows[0]['wape_pct'] == pytest.approx(60 / 700 * 100)
    assert rows[0]['random_seed'] == 42


@pytest.mark.parametrize('track, model_id', [('other', 'x'), ('jeonse', 'a/b'), ('jeonse', 'a__b')])
def test_save_run_rejects_bad_names(track, model_id, tmp_path):
    with pytest.raises(ValueError):
        save_run(track, model_id, Y, P, reports_dir=tmp_path)
