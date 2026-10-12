"""Renewal-only study: "what will a tenant pay on a future RENEWAL, given the current price?"

SEPARATE PRODUCT. It uses `previous_deposit_10k_krw` / `previous_monthly_rent_10k_krw`, which rule 5 of the
modeling protocol forbids for the main models; it is an explicit, isolated exception documented in CLAUDE.md.
Nothing here is imported by the main models, the previous-price column never enters `features.py`, and the
runs saved by the study carry `scope = "renewal_only"` in their meta so that the main leaderboard skips them.

Availability of the previous price is a domain assumption: for a future renewal the parties know the current
price. It does not exist for new contracts and it is not part of the market projection.
"""
import numpy as np
import pandas as pd

from . import config
from .boosting import BoostedModel, required_columns
from .features import build_features
from .metrics import wape

SCOPE = 'renewal_only'
RENEWAL = 'Renewal'
PREVIOUS_COLUMN = {'jeonse': 'previous_deposit_10k_krw', 'wolse': 'previous_monthly_rent_10k_krw'}
# Hard caps (10,000 KRW): above the highest current price observed in each track, so a larger previous price is not plausible.
PRICE_CAP = {'jeonse': 1_000_000.0, 'wolse': 5_000.0}
# Domain-based multiple: a renewal whose new price is below a fifth or above five times the previous price is flagged.
RATIO_BOUNDS = (0.2, 5.0)
CAP_TOLERANCE = 0.005          # tolerance added to the 5% renewal cap when counting ratios "at or below 1.05"
RENEWAL_CAP = 1.05
GROUP_COLUMNS = ('renewal_right_used', 'building_type')
MIN_GROUP_ROWS = 30

SEARCH_SPACE = {
    'learning_rate': [0.05, 0.1],
    'max_iter': [200, 400],
    'max_leaf_nodes': [15, 31, 63],
    'min_samples_leaf': [20, 50, 100],
    'l2_regularization': [0.0, 1.0, 10.0],
}


def previous_price(df_track, track):
    """Previous price of the same target (deposit for Jeonse, monthly rent for Wolse), as float."""
    return df_track[PREVIOUS_COLUMN[track]].astype(float)


def flag_previous(df_track, track):
    """Plausibility flags of the previous price (rows of one track). Nothing is removed from the data.

    - `is_renewal`: contract_type == 'Renewal' (the previous columns are only filled for renewals);
    - `previous_missing`: previous price missing or not positive (a Wolse renewal whose previous contract had no monthly rent);
    - `above_cap`: previous price above the hard cap of the track;
    - `ratio_outside`: new / previous price outside `RATIO_BOUNDS` (needs the new price: it is used to clean TRAINING
      rows only and never to select evaluation rows);
    - `input_ok`: renewal with a positive previous price not above the cap (known at prediction time);
    - `fully_ok`: `input_ok` and the ratio inside the bounds (rows used to fit the models).
    """
    prev = previous_price(df_track, track)
    new = df_track[config.TRACKS[track]['target']].astype(float)
    flags = pd.DataFrame(index=df_track.index)
    flags['is_renewal'] = (df_track['contract_type'] == RENEWAL).to_numpy()
    flags['previous_missing'] = ~(prev > 0)
    flags['above_cap'] = prev > PRICE_CAP[track]
    ratio = new / prev.where(prev > 0)
    flags['ratio_outside'] = (ratio < RATIO_BOUNDS[0]) | (ratio > RATIO_BOUNDS[1])
    flags['input_ok'] = flags['is_renewal'] & ~flags['previous_missing'] & ~flags['above_cap']
    flags['fully_ok'] = flags['input_ok'] & ~flags['ratio_outside']
    return flags


def log_previous(df_track, track):
    return np.log(previous_price(df_track, track).clip(lower=1e-9))


def renewal_features(df_track, track):
    """Allowed features of the main protocol plus the log of the previous price (renewal-only exception)."""
    X = build_features(df_track, track, include_time=True)
    X['log_prev'] = log_previous(df_track, track).to_numpy()
    return X


def ratio_wape_scorer(estimator, X, y_log_ratio):
    """Negative WAPE of the new price, rebuilt as exp(log previous + log ratio), for RandomizedSearchCV."""
    log_prev = X['log_prev'].to_numpy(dtype=float)
    new_true = np.exp(log_prev + np.asarray(y_log_ratio, dtype=float))
    new_pred = np.exp(log_prev + estimator.predict(X))
    return -wape(new_true, new_pred)


class CarryForward:
    """New price = previous price."""

    def fit(self, df_track, track):
        self.track = track
        return self

    def predict(self, df_track):
        return previous_price(df_track, self.track).to_numpy()


class GroupRatioMedian:
    """New price = previous price x median ratio of the group (renewal right x building type) in the fit rows.

    Groups with fewer than `MIN_GROUP_ROWS` fit rows use the median of their renewal-right group, then the overall median."""

    def fit(self, df_track, track):
        self.track = track
        flags = flag_previous(df_track, track)
        fit = df_track[flags['fully_ok']]
        ratio = fit[config.TRACKS[track]['target']].astype(float) / previous_price(fit, track)
        frame = pd.DataFrame({'ratio': ratio.to_numpy(), **{c: fit[c].to_numpy() for c in GROUP_COLUMNS}})
        self.overall_ = float(frame['ratio'].median())
        self.by_right_ = frame.groupby(GROUP_COLUMNS[0])['ratio'].median().to_dict()
        counts = frame.groupby(list(GROUP_COLUMNS))['ratio'].agg(['median', 'size'])
        self.by_group_ = counts[counts['size'] >= MIN_GROUP_ROWS]['median'].to_dict()
        self.n_fit_ = int(len(fit))
        return self

    def ratios(self, df_track):
        out = []
        for right, btype in zip(df_track[GROUP_COLUMNS[0]], df_track[GROUP_COLUMNS[1]]):
            out.append(self.by_group_.get((right, btype), self.by_right_.get(right, self.overall_)))
        return np.array(out, dtype=float)

    def predict(self, df_track):
        return previous_price(df_track, self.track).to_numpy() * self.ratios(df_track)


class LogRatioBoost:
    """HistGradientBoosting with median loss on log(new / previous); prediction = previous x exp(prediction).

    Features: the allowed features of the main protocol (static variant, no time feature) plus the log of the previous price."""

    def __init__(self, params, quantile=0.5):
        self.params = dict(params)
        self.quantile = quantile

    def fit(self, df_track, track):
        self.track = track
        flags = flag_previous(df_track, track)
        fit = df_track[flags['fully_ok']]
        X = renewal_features(fit, track)
        y = np.log(fit[config.TRACKS[track]['target']].astype(float) / previous_price(fit, track))
        self.columns_ = required_columns('static') + ['log_prev']
        self.model_ = BoostedModel(variant='static', quantile=self.quantile, extra_numeric=('log_prev',), **self.params).fit(X[self.columns_], y)
        self.n_fit_ = int(len(fit))
        return self

    def predict(self, df_track):
        X = renewal_features(df_track, self.track)
        return previous_price(df_track, self.track).to_numpy() * np.exp(self.model_.predict(X[self.columns_]))
