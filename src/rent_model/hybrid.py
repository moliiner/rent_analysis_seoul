"""Hybrid model: structure model without time + forecasted quality-adjusted price index.

Training: y_adj = log1p(target) - log(index_t / 100), with the TRAIN-only hedonic index, and a
structure model (no time feature) is fit on y_adj. Prediction for a TEST month:
expm1(structure prediction + log(index_forecast_t / 100)).

The 'oracle' variant replaces the forecasted index by the index fitted on all months. It uses TEST
information and is therefore NOT a valid model: it is a diagnostic upper bound only.
"""
import numpy as np
import pandas as pd

from . import config
from .boosting import BoostedModel, required_columns
from .linear import build_pipeline as build_linear_pipeline
from .ts_index import MONTH_COLUMN, forecast, hedonic_index, log_index

TRAIN_LAST_MONTH = pd.Timestamp(config.TRAIN_END.year, config.TRAIN_END.month, 1)
LOG_BASE = np.log(100.0)
STRUCTURE_KINDS = ('ridge', 'hgb')


def train_index(df, track):
    """Hedonic index fitted on the TRAIN months only (2022-01..2024-12)."""
    table = hedonic_index(df, track, last_month=TRAIN_LAST_MONTH)
    assert table.index.max() <= TRAIN_LAST_MONTH, 'the TRAIN-only index contains a TEST month'
    return table


def oracle_index(df, track):
    """Hedonic index fitted on all months. Uses TEST months: diagnostic only, never a valid model."""
    return hedonic_index(df, track)


def index_adjustment(index_table, months):
    """log(index_t / 100) for each row's month (the month coefficient of the hedonic regression)."""
    adj = pd.Series(index_table['log_coef']).reindex(pd.DatetimeIndex(months))
    if adj.isna().any():
        raise ValueError('some months are not in the index table')
    return adj.to_numpy()


def forecast_adjustment(train_index_table, candidate, horizon=12):
    """log(forecast_index_t / 100) for the `horizon` months after the TRAIN-only index.

    The forecast is made with `candidate` on the log TRAIN-only index (the one selected before 2025)."""
    f = forecast(candidate, log_index(train_index_table), horizon)
    return pd.Series(f['mean'].to_numpy() - LOG_BASE, index=f.index)


def adjust_target(y_log, months, index_table):
    """y_adj = log1p(target) - log(index_t / 100)."""
    return np.asarray(y_log, dtype=float) - index_adjustment(index_table, months)


class StructureModel:
    """Structure model on y_adj with static features only (no time feature)."""

    def __init__(self, kind, params):
        if kind not in STRUCTURE_KINDS:
            raise ValueError(f'unknown structure model: {kind}')
        self.kind, self.params = kind, dict(params)

    def fit(self, X, y_adj):
        if self.kind == 'ridge':
            self.model_ = build_linear_pipeline('ridge', time=False, **self.params).fit(X, y_adj)
        else:
            self.model_ = BoostedModel(variant='static', **self.params).fit(X[required_columns('static')], y_adj)
        return self

    def predict_log(self, X):
        if self.kind == 'ridge':
            return self.model_.predict(X)
        return self.model_.predict(X[required_columns('static')])


def combine(structure_log, adjustment_log):
    """Back-transform structure prediction + log index level to original wones."""
    return np.expm1(np.maximum(np.asarray(structure_log, dtype=float) + np.asarray(adjustment_log, dtype=float), 0.0))


class HybridPredictor:
    """Structure model + index adjustment. `valid` is False when the index is the oracle."""

    def __init__(self, structure, adjustment_by_month, valid):
        self.structure, self.adjustment_by_month, self.valid = structure, adjustment_by_month, bool(valid)

    def predict(self, X, months):
        months = pd.DatetimeIndex(months)
        adjustment = self.adjustment_by_month.reindex(months)
        if adjustment.isna().any():
            raise ValueError('index adjustment missing for some months')
        return combine(self.structure.predict_log(X), adjustment.to_numpy())

    def meta(self, **extra):
        return {'valid': self.valid, **extra}
