"""Error diagnostics for the model shortlist: repeat vs new units and price tiers.

Nothing here is a feature or a model. The unit key is an approximation of "the same dwelling"
(district, neighbourhood, lot, building type, area, floor, year built): the lot numbers are missing for part
of the rows, and then the key can merge different dwellings of the same neighbourhood. A row is a repeat
unit when its key appears among the rows of the fit window; the flag only looks at fit-window rows.
"""
import numpy as np
import pandas as pd

from .metrics import wape

UNIT_COLUMNS = ['district_name', 'legal_dong_name', 'main_lot_number', 'sub_lot_number', 'building_type',
                'leased_area_sqm', 'floor', 'year_built']


def unit_key(df):
    """One string per row built from `UNIT_COLUMNS` (missing values are written as 'NA')."""
    missing = [c for c in UNIT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f'columns missing for the unit key (load the data with keep_lots=True): {missing}')
    parts = []
    for col in UNIT_COLUMNS:
        values = df[col]
        if col == 'leased_area_sqm':
            values = values.round(2)
        parts.append(values.astype(object).where(values.notna(), 'NA').astype(str))
    key = parts[0]
    for part in parts[1:]:
        key = key + '|' + part
    return key


def lot_available(df):
    """True where the main lot number is known (the key is then a good proxy for the dwelling)."""
    return df['main_lot_number'].notna().to_numpy()


def repeat_unit_flag(fit_df, eval_df):
    """Boolean array: the unit of each evaluation row appears in the fit window (fit rows only define it)."""
    return eval_df.pipe(unit_key).isin(set(unit_key(fit_df))).to_numpy()


def price_tier(pred, n_tiers=3):
    """Tier labels from the terciles of the PREDICTED price of the evaluated rows (actuals are not used)."""
    pred = np.asarray(pred, dtype=float)
    edges = np.quantile(pred, np.linspace(0, 1, n_tiers + 1)[1:-1])
    labels = ['T1 low', 'T2 mid', 'T3 high'] if n_tiers == 3 else [f'T{i + 1}' for i in range(n_tiers)]
    return pd.Series(np.asarray(labels, dtype=object)[np.searchsorted(edges, pred, side='right')])


def error_table(groups, y, pred):
    """n, WAPE, MdAPE, MAE (original units) and aggregate bias of the predictions within each group."""
    frame = pd.DataFrame({'g': np.asarray(groups), 'y': np.asarray(y, dtype=float), 'p': np.asarray(pred, dtype=float)})

    def stats(g):
        return pd.Series({'n': len(g), 'wape_pct': wape(g['y'], g['p']),
                          'mdape_pct': float(np.median(np.abs(g['p'] - g['y']) / g['y']) * 100),
                          'mae': float(np.mean(np.abs(g['p'] - g['y']))),
                          'aggregate_bias_pct': float((g['p'].sum() - g['y'].sum()) / g['y'].sum() * 100)})
    out = frame.groupby('g').apply(stats, include_groups=False)
    out['n'] = out['n'].astype(int)
    return out
