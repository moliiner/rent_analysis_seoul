"""Naive baselines. Each class is fit on TRAIN only and predicts in original wones (10,000 KRW units)."""
import numpy as np
import pandas as pd

from . import config

DISTRICT = 'district_name'
BUILDING = 'building_type'
AREA = 'leased_area_sqm'
N_AREA_BUCKETS = 5


def _lookup(table, keys):
    """Values of the group-median `table` for the rows of `keys`; NaN where the group is missing."""
    if isinstance(keys, pd.DataFrame):
        index = pd.MultiIndex.from_frame(keys)
    else:
        index = pd.Index(keys)
    return table.reindex(index).to_numpy(dtype=float)


class GlobalMedian:
    """Median of the TRAIN target for every row."""
    model_id = 'baseline_global_median'

    def fit(self, train, target):
        self.global_ = float(train[target].median())
        return self

    def predict(self, df):
        return np.full(len(df), self.global_)


class DistrictMedian(GlobalMedian):
    """Median by district_name; unseen district -> global median."""
    model_id = 'baseline_district'

    def fit(self, train, target):
        super().fit(train, target)
        self.district_ = train.groupby(DISTRICT)[target].median()
        return self

    def predict(self, df):
        pred = _lookup(self.district_, df[DISTRICT])
        return np.where(np.isnan(pred), self.global_, pred)


class DistrictTypeMedian(DistrictMedian):
    """Median by district_name x building_type; fallback: district, then global."""
    model_id = 'baseline_district_type'

    def fit(self, train, target):
        super().fit(train, target)
        self.district_type_ = train.groupby([DISTRICT, BUILDING])[target].median()
        return self

    def _fallback(self, df):
        return super().predict(df)

    def predict(self, df):
        pred = _lookup(self.district_type_, df[[DISTRICT, BUILDING]])
        return np.where(np.isnan(pred), self._fallback(df), pred)


class DistrictTypeAreaMedian(DistrictTypeMedian):
    """Median by district x building_type x area bucket (TRAIN quintiles).

    Fallback chain: district x type, district, global.
    """
    model_id = 'baseline_district_type_area'

    def fit(self, train, target):
        super().fit(train, target)
        edges = np.quantile(train[AREA], np.linspace(0, 1, N_AREA_BUCKETS + 1)[1:-1])
        self.area_edges_ = np.unique(edges)
        keys = self._keys(train).assign(y=train[target].to_numpy())
        self.group_ = keys.groupby([DISTRICT, BUILDING, 'area_bucket'])['y'].median()
        return self

    def _bucket(self, df):
        return np.searchsorted(self.area_edges_, df[AREA].to_numpy(), side='right')

    def _keys(self, df):
        return pd.DataFrame({DISTRICT: df[DISTRICT].to_numpy(),
                             BUILDING: df[BUILDING].to_numpy(),
                             'area_bucket': self._bucket(df)})

    def predict(self, df):
        pred = _lookup(self.group_, self._keys(df))
        return np.where(np.isnan(pred), self._fallback(df), pred)


class LastYearDrift(DistrictTypeMedian):
    """District x type median scaled by median(last TRAIN year) / median(TRAIN)."""
    model_id = 'baseline_drift'

    def fit(self, train, target):
        super().fit(train, target)
        last_year = train['contract_date'].dt.year == config.TRAIN_END.year
        self.ratio_ = float(train.loc[last_year, target].median() / train[target].median())
        return self

    def predict(self, df):
        return super().predict(df) * self.ratio_


BASELINES = [GlobalMedian, DistrictMedian, DistrictTypeMedian, DistrictTypeAreaMedian, LastYearDrift]
