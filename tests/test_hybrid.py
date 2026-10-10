import numpy as np
import pandas as pd
import pytest

from rent_model import config
from rent_model.features import build_features, build_target
from rent_model.hybrid import (HybridPredictor, LOG_BASE, StructureModel, TRAIN_LAST_MONTH, adjust_target, combine,
                               forecast_adjustment, index_adjustment, oracle_index, train_index)
from rent_model.splits import temporal_split
from rent_model.ts_index import hedonic_index, log_index
from tests.test_ts_index import synthetic_market

FAST_HGB = dict(learning_rate=0.1, max_iter=40, max_leaf_nodes=15, min_samples_leaf=20)


@pytest.fixture(scope='module')
def market():
    """Synthetic Jeonse market over 2022-01..2024-06 (30 months) with a known month effect."""
    return synthetic_market(n_months=30)


def test_train_index_has_no_test_month_and_oracle_is_longer(clean_df):
    d = clean_df.sample(30_000, random_state=config.RANDOM_SEED)
    tr, orc = train_index(d, 'jeonse'), oracle_index(d, 'jeonse')
    assert tr.index.max() == TRAIN_LAST_MONTH and tr.index.min() == pd.Timestamp('2022-01-01')
    assert orc.index.max() == pd.Timestamp('2025-12-01') and len(orc) == 48
    assert tr['index'].iloc[0] == pytest.approx(100.0) and orc['index'].iloc[0] == pytest.approx(100.0)


def test_train_index_ignores_test_rows(clean_df):
    d = clean_df.sample(30_000, random_state=config.RANDOM_SEED)
    tampered = d.copy()
    test_rows = tampered['contract_month'] >= pd.Timestamp('2025-01-01')
    tampered.loc[test_rows, 'deposit_10k_krw'] *= 7
    tampered.loc[test_rows, 'monthly_rent_10k_krw'] *= 7
    pd.testing.assert_series_equal(train_index(d, 'jeonse')['index'], train_index(tampered, 'jeonse')['index'])


def test_adjustment_matches_index_and_target_is_detrended(market):
    ix = hedonic_index(market, 'jeonse')
    months = market.loc[market['lease_type'] == config.TRACKS['jeonse']['lease_type'], 'contract_month']
    adj = index_adjustment(ix, months)
    np.testing.assert_allclose(adj, np.log(ix['index'].reindex(months).to_numpy() / 100.0))
    y = build_target(market, 'jeonse')
    y_adj = adjust_target(y, months, ix)
    np.testing.assert_allclose(y.to_numpy() - y_adj, adj)
    # after detrending, the month effect disappears (the means by month are flat up to noise)
    by_month = pd.Series(y_adj).groupby(months.to_numpy()).mean()
    assert by_month.std() < 0.5 * pd.Series(y.to_numpy()).groupby(months.to_numpy()).mean().std()


def test_back_transform_consistency():
    s = np.array([8.0, 9.5, 10.2])
    # index = 100 -> adjustment 0 -> plain expm1
    np.testing.assert_allclose(combine(s, 0.0), np.expm1(s))
    # a +x log shift multiplies (1 + prediction) by exp(x)
    np.testing.assert_allclose(combine(s, 0.05) + 1, (np.expm1(s) + 1) * np.exp(0.05))
    assert (combine(np.array([-5.0]), 0.0) >= 0).all()


def test_forecast_adjustment_uses_only_train_index(market):
    ix = hedonic_index(market, 'jeonse', last_month='2023-12-01')
    adj = forecast_adjustment(ix, 'naive', horizon=6)
    assert adj.index[0] == pd.Timestamp('2024-01-01') and len(adj) == 6
    np.testing.assert_allclose(adj.to_numpy(), np.log(ix['index'].iloc[-1]) - LOG_BASE)


@pytest.mark.parametrize('kind,params', [('ridge', {'alpha': 0.01}), ('hgb', FAST_HGB)])
def test_hybrid_predictor_and_oracle_flag(market, kind, params):
    d = market.copy()
    train, test = d[d['contract_month'] <= '2023-12-01'], d[d['contract_month'] > '2023-12-01']
    ix_train = hedonic_index(d, 'jeonse', last_month='2023-12-01')
    ix_all = hedonic_index(d, 'jeonse')
    Xtr = build_features(train, 'jeonse', include_time=True)
    y_adj = adjust_target(build_target(train, 'jeonse'), train['contract_month'], ix_train)
    structure = StructureModel(kind, params).fit(Xtr, y_adj)
    Xte = build_features(test, 'jeonse', include_time=True)
    fc = forecast_adjustment(ix_train, 'drift', horizon=6)
    oracle_adj = pd.Series(ix_all['log_coef'].loc[fc.index].to_numpy(), index=fc.index)
    hybrid = HybridPredictor(structure, fc, valid=True)
    oracle = HybridPredictor(structure, oracle_adj, valid=False)
    p, po = hybrid.predict(Xte, test['contract_month']), oracle.predict(Xte, test['contract_month'])
    assert p.shape == po.shape == (len(test),) and (p > 0).all() and (po > 0).all()
    assert hybrid.meta()['valid'] is True and oracle.meta(note='diagnostic')['valid'] is False
    # same structure model: the two differ only through the index level
    ratio = (po + 1) / (p + 1)
    np.testing.assert_allclose(ratio, np.exp((oracle_adj - fc).reindex(test['contract_month']).to_numpy()), rtol=1e-9)


def test_structure_model_has_no_time_feature(market):
    from rent_model.boosting import feature_columns
    assert config.TIME_FEATURE not in feature_columns('static')
    d = market.copy()
    Xtr = build_features(d, 'jeonse', include_time=True)
    y = build_target(d, 'jeonse')
    model = StructureModel('ridge', {'alpha': 0.01}).fit(Xtr, y)
    names = model.model_.named_steps['prep'].transformers_
    assert config.TIME_FEATURE not in names[1][2] and 'month_sin' not in names[1][2]
