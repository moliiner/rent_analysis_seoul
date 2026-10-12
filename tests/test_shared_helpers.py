"""Helpers moved from the notebooks to src/rent_model: they must behave as the notebook copies did."""
import json

import numpy as np
import pandas as pd

from rent_model import config, hybrid
from rent_model.diagnostics import bias_table, partial_dependence
from rent_model.io import last_run
from rent_model.projection import comparable_contracts


def test_last_run_returns_the_last_appended_run(tmp_path):
    runs = [{'track': 'jeonse', 'model_id': 'm', 'wape_pct': 20.0}, {'track': 'jeonse', 'model_id': 'm', 'wape_pct': 15.0}]
    (tmp_path / 'jeonse__m.json').write_text(json.dumps(runs), encoding='utf-8')
    assert last_run('jeonse', 'm', metrics_dir=tmp_path)['wape_pct'] == 15.0


def test_bias_table_median_and_aggregate_by_quarter():
    y = np.array([100.0, 100.0, 200.0, 200.0])
    pred = {'a': np.array([110.0, 90.0, 220.0, 220.0])}
    quarters = np.array([1, 1, 2, 2])
    median = bias_table(y, pred, quarters, 'median')
    aggregate = bias_table(y, pred, quarters, 'aggregate')
    assert median.loc[1, 'a'] == 0.0 and np.isclose(median.loc[2, 'a'], 10.0)
    assert aggregate.loc[1, 'a'] == 0.0 and np.isclose(aggregate.loc[2, 'a'], 10.0)


def test_partial_dependence_follows_the_model_and_keeps_other_columns():
    class Linear:
        def predict(self, X):
            return 0.1 * X['x'].to_numpy() + 0.0 * X['z'].to_numpy()
    X = pd.DataFrame({'x': np.linspace(0, 10, 200), 'z': 1.0})
    grid, values = partial_dependence(Linear(), X, 'x', ['x', 'z'], n_grid=5)
    assert len(grid) == len(values) == 5 and (np.diff(values) > 0).all()


def test_comparable_contracts_counts_the_latest_year_with_area_and_type_filters():
    d = pd.DataFrame({'district_name': 'A', 'building_type': 'Apartment', 'contract_type': ['New', 'New', 'New', 'Renewal', 'New'],
                      'contract_date': pd.to_datetime(['2025-01-05', '2025-02-05', '2025-03-05', '2025-03-05', '2024-03-05']),
                      'leased_area_sqm': [60.0, 70.0, 100.0, 60.0, 60.0], 'deposit_10k_krw': [1000.0, 2000.0, 9000.0, 5000.0, 7000.0]})
    n, median = comparable_contracts(d, 'jeonse', 'A', 'Apartment', 60.0, 'New')
    assert n == 2 and median == 1500.0
    assert comparable_contracts(d, 'jeonse', 'B', 'Apartment', 60.0, 'New')[0] == 0


def test_last_train_month_comes_from_the_config():
    assert config.TRAIN_LAST_MONTH == pd.Timestamp('2024-12-01') == hybrid.TRAIN_LAST_MONTH
