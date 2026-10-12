"""Feature and target construction. Nothing here is fit on data, so it cannot leak TEST information."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config

SEPARATOR = ' | '
OTHER_LEVEL = 'OTHER'
MAX_NEIGHBOURHOOD_LEVELS = 254      # native categorical support of HistGradientBoosting is limited to 255 levels


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


class NeighbourhoodGrouper(BaseEstimator, TransformerMixin):
    """Grouped `legal_dong` feature: 'district | dong' levels with at least `min_count` rows in the fit
    window keep their own level; every other level, and every level unseen at predict time, goes to the
    OTHER level of its own district ('district | OTHER'). At most `max_levels` levels are produced
    (kept neighbourhoods plus one OTHER level per fit district); a district never seen in fit maps to a
    single global OTHER level. Only the rows given to `fit` define the levels. `transform` returns
    integer codes (a native categorical feature of the booster); `group` returns the level names."""

    def __init__(self, min_count=30, max_levels=MAX_NEIGHBOURHOOD_LEVELS):
        self.min_count = min_count
        self.max_levels = max_levels

    @staticmethod
    def _district(values):
        return values.str.split(SEPARATOR, n=1).str[0]

    def fit(self, X, y=None):
        values = pd.Series(np.asarray(X, dtype=object).ravel(), dtype=object).astype(str)
        districts = sorted(set(self._district(values)))
        counts = values.value_counts()
        counts = counts[counts >= self.min_count]
        budget = max(self.max_levels - len(districts), 0)
        self.kept_ = set(counts.head(budget).index)                   # value_counts is sorted by frequency
        self.districts_ = set(districts)
        self.levels_ = sorted(self.kept_) + [f'{d}{SEPARATOR}{OTHER_LEVEL}' for d in districts]
        self.codes_ = {level: i for i, level in enumerate(self.levels_)}
        self.n_fit_rows_ = int(len(values))
        return self

    def group(self, X):
        values = pd.Series(np.asarray(X, dtype=object).ravel(), dtype=object).astype(str)
        district = self._district(values)
        other = np.where(district.isin(self.districts_), district + SEPARATOR + OTHER_LEVEL, OTHER_LEVEL)
        return np.where(values.isin(self.kept_), values, other)

    def transform(self, X):
        grouped = pd.Series(self.group(X))
        return grouped.map(self.codes_).fillna(len(self.levels_)).to_numpy(dtype=float).reshape(-1, 1)
