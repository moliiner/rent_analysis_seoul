"""Load and clean the dataset. Reproduces the cleaning of Notebook.ipynb (section 4) exactly."""
from pathlib import Path

import pandas as pd

from . import config


def load_clean(path=config.DATA_PATH, check=True):
    """Return the cleaned DataFrame used by the EDA notebook.

    Steps, in the order of the notebook: duplicates first (while building_name and the lot numbers
    are still present), NaN fills, drop of building_name and lot numbers, drop of missing year_built,
    type fixes, removal of the partial month (contract_month >= 2026-01-01), year_built range
    filter and floor filter.
    """
    df = pd.read_csv(Path(path))
    df = df.drop_duplicates()

    df['legal_dong_name'] = df['legal_dong_name'].fillna('Unknown')
    df = df.drop(['building_name', 'main_lot_number', 'sub_lot_number'], axis=1)
    df['lot_type_code'] = df['lot_type_code'].fillna(0.0)
    df['lot_type'] = df['lot_type'].fillna('No lot type')
    df['floor_reported'] = df['floor'].notna()
    df['contract_type'] = df['contract_type'].fillna('Not specified')
    df['renewal_right_used'] = df['renewal_right_used'].fillna('No/Not reported')

    df = df.dropna(subset=['year_built'])

    df['leased_area_sqm'] = df['leased_area_sqm'].str.replace(',', '.', regex=False).astype(float)
    df['contract_date'] = pd.to_datetime(df['contract_date'])
    df['contract_month'] = df['contract_date'].dt.to_period('M').dt.to_timestamp()

    df = df[df['contract_month'] < config.PARTIAL_MONTH_CUTOFF]
    df = df[df['year_built'].between(config.MIN_YEAR_BUILT, config.MAX_YEAR_BUILT)]
    df = df[df['floor'].isna() | (df['floor'] <= config.MAX_FLOOR)]
    df = df.reset_index(drop=True)

    if check:
        assert len(df) == config.EXPECTED_ROWS, f'unexpected row count: {len(df):,}'
        for track, spec in config.TRACKS.items():
            n = int((df['lease_type'] == spec['lease_type']).sum())
            assert n == config.EXPECTED_ROWS_BY_TRACK[track], f'unexpected {track} rows: {n:,}'
    return df
