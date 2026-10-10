import numpy as np
import pandas as pd
import pytest

from rent_model import config
from rent_model.ts_index import (ALPHAS, CANDIDATES, MAX_HORIZON, adf_report, error_table, forecast, hedonic_index,
                                 independent_columns, log_index, rolling_origin_forecasts, rolling_origins,
                                 save_ts_report, select_candidate)

TRUE_EFFECT = 0.01   # log points per month in the synthetic market


def synthetic_market(n_months=30, per_month=300, seed=0):
    """Rows of the Jeonse track with a known month effect and known structural effects."""
    rng = np.random.default_rng(seed)
    months = pd.date_range('2022-01-01', periods=n_months, freq='MS')
    month_idx = np.repeat(np.arange(n_months), per_month)
    n = len(month_idx)
    district = rng.choice(['A-gu', 'B-gu', 'C-gu'], n)
    btype = rng.choice(['Apartment', 'Officetel', 'Row house / Villa (multi-unit)', 'Detached / Multi-household house'], n)
    area = rng.uniform(20, 120, n)
    floor = np.where(btype == 'Detached / Multi-household house', np.nan, rng.integers(1, 20, n)).astype(float)
    age = rng.integers(0, 40, n)
    effect = (TRUE_EFFECT * month_idx + 0.2 * (district == 'B-gu') + 0.4 * (district == 'C-gu') +
              0.5 * np.log(area) + 0.1 * (btype == 'Officetel') - 0.005 * age)
    deposit = np.expm1(7.0 + effect + rng.normal(0, 0.05, n))
    month = months[month_idx]
    return pd.DataFrame({
        'lease_type': config.TRACKS['jeonse']['lease_type'], 'deposit_10k_krw': deposit,
        'monthly_rent_10k_krw': 0.0, 'district_name': district, 'building_type': btype,
        'contract_type': rng.choice(['New', 'Renewal'], n), 'renewal_right_used': 'No/Not reported',
        'leased_area_sqm': area, 'floor': floor, 'year_built': month.year - age,
        'contract_date': month + pd.Timedelta(days=9), 'contract_month': month})


@pytest.fixture(scope='module')
def market():
    return synthetic_market()


def test_index_rebased_to_100_and_recovers_known_month_effect(market):
    ix = hedonic_index(market, 'jeonse')
    assert ix['index'].iloc[0] == pytest.approx(100.0) and len(ix) == 30
    expected = 100 * np.exp(TRUE_EFFECT * np.arange(30))
    np.testing.assert_allclose(ix['index'].to_numpy(), expected, rtol=0.01)
    assert (ix['index_low95'] <= ix['index'] + 1e-9).all() and (ix['index'] <= ix['index_high95'] + 1e-9).all()
    assert 'floor_missing' in ix.attrs['dropped_controls']   # identical to the Detached indicator


def test_train_only_fit_contains_no_future_month_and_ignores_future_rows(market):
    cut = pd.Timestamp('2023-12-01')
    ix = hedonic_index(market, 'jeonse', last_month=cut)
    assert ix.index.max() == cut and len(ix) == 24
    tampered = market.copy()
    future = tampered['contract_month'] > cut
    tampered.loc[future, 'deposit_10k_krw'] *= 5
    ix2 = hedonic_index(tampered, 'jeonse', last_month=cut)
    pd.testing.assert_series_equal(ix['index'], ix2['index'])
    assert hedonic_index(market, 'jeonse').loc[:cut, 'index'].iloc[0] == pytest.approx(100.0)


def test_independent_columns_drops_duplicates():
    frame = pd.DataFrame({'a': [1.0, 2, 3, 4], 'b': [2.0, 4, 6, 8], 'c': [1.0, 0, 1, 5]})
    kept = independent_columns(frame)
    assert len(kept) == 2 and 'c' in kept


def test_rolling_origins_respect_time():
    origins = rolling_origins(36, first_origin=18, step=3, max_horizon=12)
    assert [t for t, _ in origins] == [18, 21, 24, 27, 30, 33]
    for t, h in origins:
        assert t + h <= 36 and 1 <= h <= MAX_HORIZON and t >= 18


def test_rolling_forecasts_use_vintages_and_truth_after_origin(market):
    final = hedonic_index(market, 'jeonse', last_month='2023-12-01')
    fc = rolling_origin_forecasts(market, 'jeonse', final, candidates=['naive', 'drift'], first_origin=18, step=3)
    assert (fc['month'] > fc['origin']).all()
    assert (fc['month'] <= final.index.max()).all()
    assert fc.groupby('origin')['horizon'].max().max() <= 12
    table = error_table(fc)
    assert select_candidate(table) in ('naive', 'drift') and table[('MAPE', 'mean_h')].notna().all()


@pytest.mark.parametrize('name', CANDIDATES)
def test_forecast_shapes_and_interval_order(name):
    idx = pd.date_range('2022-01-01', periods=36, freq='MS')
    y = pd.Series(np.log(100) + 0.01 * np.arange(36) + np.random.default_rng(1).normal(0, 0.004, 36), index=idx)
    f = forecast(name, y, 12, ALPHAS)
    assert len(f) == 12 and f.index[0] == pd.Timestamp('2025-01-01') and np.isfinite(f.to_numpy()).all()
    assert (f['lo95'] <= f['lo80']).all() and (f['lo80'] <= f['mean']).all()
    assert (f['mean'] <= f['hi80']).all() and (f['hi80'] <= f['hi95']).all()


def test_simple_forecasts_values():
    idx = pd.date_range('2022-01-01', periods=24, freq='MS')
    y = pd.Series(np.log(100) + 0.01 * np.arange(24), index=idx)
    assert forecast('naive', y, 3)['mean'].tolist() == [y.iloc[-1]] * 3
    assert forecast('seasonal_naive', y, 2)['mean'].tolist() == [y.iloc[-12], y.iloc[-11]]
    np.testing.assert_allclose(forecast('drift', y, 2)['mean'], y.iloc[-1] + 0.01 * np.array([1, 2]))
    np.testing.assert_allclose(forecast('linear_trend', y, 2)['mean'], y.iloc[-1] + 0.01 * np.array([1, 2]))


def test_adf_report_and_save(tmp_path):
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(size=60))
    report = adf_report(walk)
    assert {'adf_stat', 'p_value', 'lags_used', 'n_obs', 'critical_5pct'} <= set(report)
    path = save_ts_report('jeonse', {'index': pd.Series([100.0, 101.0], index=pd.to_datetime(['2022-01-01', '2022-02-01']))},
                          reports_dir=tmp_path)
    assert path.name == 'ts_index_jeonse.json' and '"2022-01-01' in path.read_text(encoding='utf-8')


def test_log_index_has_monthly_frequency(market):
    y = log_index(hedonic_index(market, 'jeonse'))
    assert y.index.freqstr == 'MS' and y.iloc[0] == pytest.approx(np.log(100))
