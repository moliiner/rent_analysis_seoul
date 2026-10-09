import pytest

from rent_model.data import load_clean


@pytest.fixture(scope='session')
def clean_df():
    return load_clean()
