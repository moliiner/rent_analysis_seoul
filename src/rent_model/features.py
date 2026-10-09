"""Feature and target construction. Nothing here is fit on data, so it cannot leak TEST information."""
import numpy as np
import pandas as pd

from . import config


def select_track(df, track):
    """Return the rows of one track (Jeonse or Wolse), keeping the original index."""
    spec = config.TRACKS[track]
    return df[df['lease_type'] == spec['lease_type']]


def forbidden_columns(columns, track=None):
    """Return the columns that must never be features (rule 5 of the protocol)."""
    return [
        col for col in columns
        if col in config.FORBIDDEN_FEATURES or col.startswith(config.FORBIDDEN_PREFIXES)
    ]


def build_features(df, track, include_time=False, include_legal_dong=False):
    """Build the allowed feature matrix of one track.

    Categorical columns are returned as strings and floor keeps its NaN: imputation and encoding
    belong to a pipeline fit on TRAIN only.
    """
    d = select_track(df, track)
    contract_year = d['contract_date'].dt.year
    month = d['contract_date'].dt.month

    X = pd.DataFrame(index=d.index)
    for col in config.CATEGORICAL_FEATURES:
        X[col] = d[col]
    X['log_area'] = np.log(d['leased_area_sqm'])
    X['floor'] = d['floor']
    X['floor_missing'] = d['floor'].isna().astype(int)
    X['building_age'] = (contract_year - d['year_built']).clip(lower=0)
    X['month_of_year'] = month
    X['month_sin'] = np.sin(2 * np.pi * month / 12)
    X['month_cos'] = np.cos(2 * np.pi * month / 12)

    if include_time:
        origin = config.TIME_ORIGIN
        X[config.TIME_FEATURE] = (contract_year - origin.year) * 12 + (month - origin.month)
    if include_legal_dong:
        X[config.LEGAL_DONG_FEATURE] = d['district_name'] + ' | ' + d['legal_dong_name']

    bad = forbidden_columns(X.columns, track)
    assert not bad, f'forbidden features in the feature matrix: {bad}'
    return X


def build_target(df, track, log=True):
    """Return the target of one track; log=True gives y = log1p(target)."""
    y = select_track(df, track)[config.TRACKS[track]['target']].astype(float)
    return np.log1p(y) if log else y


def inverse_target(y_log):
    """Back-transform predictions from the log1p scale."""
    return np.expm1(y_log)
