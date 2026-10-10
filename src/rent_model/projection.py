"""Scenario projection 2026-2028 with the chosen models (extrapolation scenarios, not forecasts).

The logic mirrors Step 4 of notebooks/09_model_comparison_projection.ipynb: the chosen model of each
track (marked in reports/metrics/leaderboard.csv) is refit on all 2022-2025 rows through its recipe
(`recipes.py`) and projected over 2026-01..2028-12. Low and high scenarios scale the base path by the
80% interval of the quality-adjusted index forecast selected in notebook 07.
"""
import json

import numpy as np
import pandas as pd

from . import config
from .compare import interval_multipliers, profile_frame, representative_profile
from .features import select_track
from .hybrid import oracle_index
from .recipes import ModelRecipe
from .ts_index import ALPHAS, forecast, log_index

FUTURE = pd.date_range('2026-01-01', '2028-12-01', freq='MS')
FIT_CUTOFF = pd.Timestamp('2026-01-01')      # every 2022-2025 row is used in the final refit
LEADERBOARD_PATH = config.METRICS_DIR / 'leaderboard.csv'


def chosen_models(leaderboard_path=LEADERBOARD_PATH):
    """{track: model_id} of the models marked as chosen in the saved leaderboard."""
    board = pd.read_csv(leaderboard_path)
    chosen = board[board['is_chosen_model']]
    return dict(zip(chosen['track'], chosen['model_id']))


class ScenarioModel:
    """Chosen model of one track, refit on all months, plus the index forecast used for the bands."""

    def __init__(self, track, model_id, recipe, index_forecast, index_candidate, track_df):
        self.track, self.model_id, self.recipe = track, model_id, recipe
        self.index_forecast, self.index_candidate, self.track_df = index_forecast, index_candidate, track_df

    def profile(self, district, building_type, area_sqm):
        """Representative contract (median floor and age of the group) and the rows that support it."""
        profile, n_rows = representative_profile(self.track_df, district, building_type, area_sqm)
        group = self.track_df[(self.track_df['district_name'] == district) & (self.track_df['building_type'] == building_type)]
        source = group if len(group) else self.track_df[self.track_df['district_name'] == district]
        profile['legal_dong'] = (source['district_name'] + ' | ' + source['legal_dong_name']).mode().iloc[0]
        return profile, n_rows

    def project(self, profile, months=FUTURE):
        """Monthly base / low / high prediction (10,000 KRW) of a profile; Wolse is monthly rent."""
        months = pd.DatetimeIndex(months)
        base = pd.Series(self.recipe.predict(profile_frame(profile, months), months), index=months)
        m_lo, m_hi = interval_multipliers(self.index_forecast)
        return pd.DataFrame({'base': base, 'low': (base + 1) * m_lo.to_numpy() - 1,
                             'high': (base + 1) * m_hi.to_numpy() - 1})


def fit_scenario_model(df, track, model_id=None):
    """Refit the chosen model of `track` on all rows of `df` (the cleaned dataset)."""
    model_id = model_id or chosen_models()[track]
    df_t = select_track(df, track)
    recipe = ModelRecipe(model_id, track).fit(df_t, FIT_CUTOFF)

    index_all = oracle_index(df, track)           # index fitted on all months: legitimate for the final refit
    report = json.loads((config.METRICS_DIR / f'ts_index_{track}.json').read_text(encoding='utf-8'))
    candidate = report['rolling_origin']['selected']
    index_forecast = np.exp(forecast(candidate, log_index(index_all), len(FUTURE), ALPHAS))
    return ScenarioModel(track, model_id, recipe, index_forecast, candidate, df_t)
