import pytest

from rent_model import config
from rent_model.features import build_features, build_target, forbidden_columns, inverse_target
from rent_model.splits import temporal_split


@pytest.mark.parametrize('track', ['jeonse', 'wolse'])
@pytest.mark.parametrize('include_time', [False, True])
@pytest.mark.parametrize('include_legal_dong', [False, True])
def test_no_forbidden_column(clean_df, track, include_time, include_legal_dong):
    X = build_features(clean_df, track, include_time=include_time, include_legal_dong=include_legal_dong)
    assert forbidden_columns(X.columns, track) == []
    assert not set(config.TARGETS) & set(X.columns)
    assert ('t' in X.columns) == include_time
    assert ('legal_dong' in X.columns) == include_legal_dong
    expected = set(config.BASE_FEATURES)
    if include_time:
        expected.add('t')
    if include_legal_dong:
        expected.add('legal_dong')
    assert set(X.columns) == expected


def test_forbidden_detection():
    bad = ['lease_type', 'previous_deposit_10k_krw', 'zone', 'registration_year', 'contract_start', 'legal_dong_code']
    assert forbidden_columns(bad + ['log_area']) == bad


@pytest.mark.parametrize('track', ['jeonse', 'wolse'])
def test_features_are_track_pure_and_aligned(clean_df, track):
    X = build_features(clean_df, track)
    y = build_target(clean_df, track)
    assert X.index.equals(y.index)
    assert (clean_df.loc[X.index, 'lease_type'] == config.TRACKS[track]['lease_type']).all()
    assert len(X) == config.EXPECTED_ROWS_BY_TRACK[track]
    assert X['building_age'].ge(0).all()
    assert X['month_of_year'].between(1, 12).all()
    assert (X['floor_missing'] == X['floor'].isna().astype(int)).all()


def test_features_do_not_depend_on_other_rows(clean_df):
    """Features built on TRAIN alone equal the TRAIN rows of the full matrix (nothing is fit)."""
    full = build_features(clean_df, 'jeonse', include_time=True)
    train, _ = temporal_split(clean_df)
    part = build_features(train, 'jeonse', include_time=True)
    assert part.equals(full.loc[part.index])


def test_time_index_and_target_round_trip(clean_df):
    X = build_features(clean_df, 'wolse', include_time=True)
    assert X['t'].min() == 0
    assert X['t'].max() == 47
    y = build_target(clean_df, 'wolse', log=False)
    assert abs(inverse_target(build_target(clean_df, 'wolse')) - y).max() < 1e-6
