import numpy as np
import pandas as pd
import pytest

from rent_model.baselines import (BASELINES, DistrictMedian, DistrictTypeAreaMedian,
                                  DistrictTypeMedian, GlobalMedian, LastYearDrift)


def make_train():
    rows = []
    for i in range(20):
        rows.append(('A', 'Apartment', 20.0 + i, 100.0 + i, '2023-05-01'))
        rows.append(('A', 'Officetel', 20.0 + i, 50.0, '2024-05-01'))
        rows.append(('B', 'Apartment', 20.0 + i, 200.0, '2024-06-01'))
    df = pd.DataFrame(rows, columns=['district_name', 'building_type', 'leased_area_sqm',
                                     'y', 'contract_date'])
    df['contract_date'] = pd.to_datetime(df['contract_date'])
    return df


def new_rows(*rows):
    return pd.DataFrame(rows, columns=['district_name', 'building_type', 'leased_area_sqm'])


def test_global_and_district():
    train = make_train()
    g = GlobalMedian().fit(train, 'y')
    assert g.predict(new_rows(('A', 'Apartment', 30.0)))[0] == train['y'].median()
    d = DistrictMedian().fit(train, 'y')
    assert d.predict(new_rows(('B', 'Officetel', 30.0)))[0] == 200.0
    assert d.predict(new_rows(('Z', 'Apartment', 30.0)))[0] == train['y'].median()


def test_district_type_fallbacks():
    train = make_train()
    m = DistrictTypeMedian().fit(train, 'y')
    pred = m.predict(new_rows(('A', 'Officetel', 30.0), ('B', 'Officetel', 30.0),
                              ('Z', 'Apartment', 30.0)))
    assert pred[0] == 50.0                       # group present
    assert pred[1] == 200.0                      # missing group -> district median
    assert pred[2] == train['y'].median()        # unseen district -> global median


def test_area_baseline_fallback_and_buckets():
    train = make_train()
    m = DistrictTypeAreaMedian().fit(train, 'y')
    assert len(m.area_edges_) == 4
    # Area far outside TRAIN range still lands in an edge bucket of a known group
    assert m.predict(new_rows(('B', 'Apartment', 500.0)))[0] == 200.0
    # (A, Officetel) exists, its top bucket may be empty -> falls back to district x type, not NaN
    pred = m.predict(new_rows(('A', 'Officetel', 39.0), ('B', 'Officetel', 30.0),
                              ('Z', 'Apartment', 30.0)))
    assert not np.isnan(pred).any()
    assert pred[0] == 50.0 and pred[1] == 200.0 and pred[2] == train['y'].median()


def test_drift_ratio():
    train = make_train()
    m = LastYearDrift().fit(train, 'y')
    in_2024 = train[train['contract_date'].dt.year == 2024]['y'].median()
    assert m.ratio_ == pytest.approx(in_2024 / train['y'].median())
    base = DistrictTypeMedian().fit(train, 'y').predict(new_rows(('A', 'Apartment', 30.0)))
    assert m.predict(new_rows(('A', 'Apartment', 30.0)))[0] == pytest.approx(base[0] * m.ratio_)


@pytest.mark.parametrize('cls', BASELINES)
def test_fit_ignores_test_rows(cls):
    """Adding extreme TEST-year rows to the data must not change a model fit on the TRAIN subset."""
    train = make_train()
    probe = new_rows(('A', 'Apartment', 25.0), ('B', 'Officetel', 30.0), ('Z', 'Officetel', 99.0))
    before = cls().fit(train, 'y').predict(probe)
    test = train.head(10).copy()
    test['y'] = 1e9
    test['contract_date'] = pd.Timestamp('2025-03-01')
    mixed = pd.concat([train, test])
    from rent_model.splits import temporal_split
    tr, _ = temporal_split(mixed)
    after = cls().fit(tr, 'y').predict(probe)
    np.testing.assert_allclose(before, after)


def test_area_baseline_independent_of_row_index():
    """TRAIN keeps its original (non-contiguous) index in the pipeline; the result must not change."""
    train = make_train()
    probe = new_rows(('A', 'Apartment', 25.0), ('A', 'Apartment', 38.0), ('B', 'Apartment', 22.0))
    expected = DistrictTypeAreaMedian().fit(train, 'y').predict(probe)
    shuffled = train.sample(frac=1.0, random_state=0)
    shuffled.index = shuffled.index * 7 + 1000
    np.testing.assert_allclose(DistrictTypeAreaMedian().fit(shuffled, 'y').predict(probe), expected)
    assert expected[0] != expected[1]   # buckets really separate the two areas
