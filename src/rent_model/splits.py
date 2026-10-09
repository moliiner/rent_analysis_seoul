"""Temporal splits. A random split is never used (rule 3 of the protocol)."""
import numpy as np

from . import config


def temporal_split(df):
    """Return (train, test): TRAIN 2022-01-01..2024-12-31, TEST 2025."""
    dates = df['contract_date']
    train = df[(dates >= config.TRAIN_START) & (dates < config.TEST_START)]
    test = df[dates.dt.year == config.TEST_YEAR]
    return train, test


def month_index(dates):
    """Months elapsed since the start of TRAIN (2022-01 -> 0)."""
    origin = config.TIME_ORIGIN
    return ((dates.dt.year - origin.year) * 12 + (dates.dt.month - origin.month)).to_numpy()


def expanding_window_folds(df_train, n_splits=config.N_CV_SPLITS, first_train_months=12):
    """Expanding-window time CV inside TRAIN.

    The first fold trains on the first `first_train_months` months; the remaining months are cut
    into `n_splits` consecutive validation blocks of equal length (the last block absorbs the
    remainder). Fold k trains on everything before its validation block.

    Returns a list of (train_positions, valid_positions) arrays, positional with respect to
    `df_train`, so they can be passed to scikit-learn as `cv`.
    """
    months = month_index(df_train['contract_date'])
    first_month = months.min()
    months = months - first_month
    n_months = months.max() + 1
    block = (n_months - first_train_months) // n_splits
    if block < 1:
        raise ValueError('not enough months for the requested number of folds')

    folds = []
    for k in range(n_splits):
        valid_start = first_train_months + k * block
        valid_end = n_months if k == n_splits - 1 else valid_start + block
        train_pos = np.flatnonzero(months < valid_start)
        valid_pos = np.flatnonzero((months >= valid_start) & (months < valid_end))
        folds.append((train_pos, valid_pos))
    return folds
