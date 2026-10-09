from rent_model import config


def test_row_counts_after_cleaning(clean_df):
    assert len(clean_df) == 387_326
    assert (clean_df['lease_type'] == config.TRACKS['jeonse']['lease_type']).sum() == 176_783
    assert (clean_df['lease_type'] == config.TRACKS['wolse']['lease_type']).sum() == 210_543


def test_cleaning_rules(clean_df):
    assert clean_df['year_built'].between(1850, 2026).all()
    assert clean_df['floor'].dropna().le(100).all()
    assert clean_df['contract_date'].max() < config.PARTIAL_MONTH_CUTOFF
    assert clean_df['year_built'].notna().all()
    assert clean_df['leased_area_sqm'].dtype == float
    assert not {'building_name', 'main_lot_number', 'sub_lot_number'} & set(clean_df.columns)
