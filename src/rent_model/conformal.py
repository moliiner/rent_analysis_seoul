"""Split conformal prediction intervals for a contract-level prediction, on the log1p scale.

Procedure (fit window = the rows available before the evaluation period):
  1. the last `months` months of the fit window are the calibration window; the rows before it are the
     proper training rows (time-ordered split, so the calibration rows are the most recent ones);
  2. the model is fit on the proper rows only and predicts the calibration rows (a backtest);
  3. the nonconformity score is the absolute residual on the log1p scale, |log1p(y) - log1p(prediction)|,
     optionally divided by a difficulty estimate sigma(building type) computed on the proper rows only;
  4. the interval radius is the finite-sample quantile of the scores: the ceil((n + 1) * level)-th smallest
     of n scores. For exchangeable scores the coverage is at least `level`;
  5. the interval of a new contract is [expm1(z_hat - q * sigma), expm1(z_hat + q * sigma)] around
     z_hat = log1p(point prediction), clipped at zero.

The model used for the evaluation rows is normally refit on the whole fit window, so it is slightly better
than the model that produced the calibration scores (the intervals are then slightly conservative). The
exchangeability between the calibration window and the evaluation period is an assumption: under a shift
of the market level or of the error scale the coverage can drop, which the notebooks measure.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .features import build_features

LEVELS = (0.8, 0.95)
CALIBRATION_MONTHS = 6


def conformal_quantile(scores, level):
    """Finite-sample split-conformal quantile: the ceil((n + 1) * level)-th smallest of n scores.

    Returns +inf when ceil((n + 1) * level) > n (too few calibration scores for that level)."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0 or not 0 < level < 1:
        raise ValueError('need at least one score and 0 < level < 1')
    rank = math.ceil((n + 1) * level)
    return float('inf') if rank > n else float(scores[rank - 1])


def split_calibration(df_fit, months=CALIBRATION_MONTHS):
    """(proper, calibration): the calibration rows are the last `months` calendar months of the fit window."""
    last = df_fit['contract_month'].max()
    start = last - pd.DateOffset(months=months - 1)
    calibration = df_fit[df_fit['contract_month'] >= start]
    proper = df_fit[df_fit['contract_month'] < start]
    assert len(proper) and len(calibration) and proper['contract_date'].max() < calibration['contract_date'].min()
    return proper, calibration


def difficulty_by_group(proper, y_log, group='building_type'):
    """Standard deviation of log1p(target) within each group of the proper rows (a TRAIN-only difficulty estimate).
    Groups unseen at predict time use the overall standard deviation (key `None`)."""
    y_log = pd.Series(np.asarray(y_log, dtype=float), index=proper.index)
    sigma = y_log.groupby(proper[group]).std().to_dict()
    sigma[None] = float(y_log.std())
    return sigma


def sigma_of(groups, sigma):
    return np.array([sigma.get(g, sigma[None]) for g in groups], dtype=float)


@dataclass
class Calibration:
    """Calibration result: radii on the log1p scale for each level, unscaled and scaled by sigma(group)."""
    levels: tuple
    q: dict = field(default_factory=dict)            # level -> radius (unscaled)
    q_scaled: dict = field(default_factory=dict)     # level -> radius of the normalised score (scaled)
    sigma: dict = field(default_factory=dict)        # group -> difficulty estimate
    start: str = ''                                  # first month of the calibration window
    last: str = ''                                   # last month of the calibration window
    n: int = 0
    median_abs_residual: float = 0.0
    mean_residual: float = 0.0                       # signed: log1p(actual) - log1p(prediction)


def calibrate(model_factory, df_fit, track, months=CALIBRATION_MONTHS, levels=LEVELS, group='building_type'):
    """Calibrate on the last `months` of `df_fit` (rows of one track). `model_factory()` returns an unfitted
    `ModelRecipe`; it is fit on the proper rows with the calibration start as cutoff."""
    proper, calibration = split_calibration(df_fit, months)
    start = calibration['contract_month'].min()
    model = model_factory().fit(proper, start)
    features = build_features(calibration, track, include_time=True, include_legal_dong=True)
    pred = model.predict(features, calibration['contract_month'])
    target = config.TRACKS[track]['target']
    residual = np.log1p(calibration[target].to_numpy(dtype=float)) - np.log1p(np.asarray(pred, dtype=float))
    sigma = difficulty_by_group(proper, np.log1p(proper[target]), group)
    scaled = np.abs(residual) / sigma_of(calibration[group], sigma)
    return Calibration(levels=tuple(levels), q={lv: conformal_quantile(np.abs(residual), lv) for lv in levels},
                       q_scaled={lv: conformal_quantile(scaled, lv) for lv in levels}, sigma=sigma,
                       start=str(start.date()), last=str(calibration['contract_month'].max().date()), n=int(len(calibration)),
                       median_abs_residual=float(np.median(np.abs(residual))), mean_residual=float(residual.mean()))


def intervals(prediction, calibration, groups=None, scaled=False):
    """{level: (lower, upper)} on the original scale around `prediction` (original scale, non-negative)."""
    z = np.log1p(np.asarray(prediction, dtype=float))
    out = {}
    for level in calibration.levels:
        if scaled:
            radius = calibration.q_scaled[level] * sigma_of(groups, calibration.sigma)
        else:
            radius = calibration.q[level]
        out[level] = (np.maximum(np.expm1(z - radius), 0.0), np.expm1(z + radius))
    return out


def coverage(y, lower, upper):
    """Share of observations inside [lower, upper] and the mean interval width."""
    y = np.asarray(y, dtype=float)
    return float(np.mean((y >= lower) & (y <= upper))), float(np.mean(np.asarray(upper) - np.asarray(lower)))


def coverage_by_group(groups, y, lower, upper):
    """n, coverage and mean width within each group."""
    frame = pd.DataFrame({'g': np.asarray(groups), 'y': np.asarray(y, dtype=float), 'lo': np.asarray(lower, dtype=float),
                          'hi': np.asarray(upper, dtype=float)})
    inside = (frame['y'] >= frame['lo']) & (frame['y'] <= frame['hi'])
    out = pd.DataFrame({'n': frame.groupby('g').size(), 'coverage': inside.groupby(frame['g']).mean(),
                        'mean_width': (frame['hi'] - frame['lo']).groupby(frame['g']).mean()})
    return out
