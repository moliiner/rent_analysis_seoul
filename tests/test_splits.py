import numpy as np
import pytest

from rent_model import config
from rent_model.features import select_track
from rent_model.splits import expanding_window_folds, temporal_split


@pytest.mark.parametrize('track, n_train, n_test', [('jeonse', 138_848, 37_935), ('wolse', 152_714, 57_829)])
def test_split_sizes(clean_df, track, n_train, n_test):
    train, test = temporal_split(select_track(clean_df, track))
    assert len(train) == n_train
    assert len(test) == n_test


def test_no_train_row_after_test_start(clean_df):
    train, test = temporal_split(clean_df)
    assert train['contract_date'].max() <= config.TRAIN_END
    assert train['contract_date'].max() < test['contract_date'].min()
    assert (test['contract_date'].dt.year == config.TEST_YEAR).all()
    assert len(train) + len(test) == len(clean_df)
    assert train.index.intersection(test.index).empty


@pytest.mark.parametrize('track', ['jeonse', 'wolse'])
def test_fold_boundaries_increase_in_time(clean_df, track):
    train, _ = temporal_split(select_track(clean_df, track))
    folds = expanding_window_folds(train, n_splits=3)
    assert len(folds) == 3

    dates = train['contract_date'].to_numpy()
    prev_train_end = prev_valid_start = prev_valid_end = None
    for train_pos, valid_pos in folds:
        train_end, valid_start, valid_end = dates[train_pos].max(), dates[valid_pos].min(), dates[valid_pos].max()
        assert train_end < valid_start
        assert np.intersect1d(train_pos, valid_pos).size == 0
        if prev_train_end is not None:
            assert train_end > prev_train_end
            assert valid_start > prev_valid_start
            assert valid_start > prev_valid_end
        prev_train_end, prev_valid_start, prev_valid_end = train_end, valid_start, valid_end

    assert dates[folds[-1][1]].max() == dates.max()
    assert all(len(v) > 0 for _, v in folds)
