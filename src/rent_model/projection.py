"""Scenario projection 2026-2028 with the chosen models (extrapolation scenarios, not forecasts).

The logic mirrors Step 4 of notebooks/09_model_comparison_projection.ipynb: the chosen model of each
track (marked in reports/metrics/leaderboard.csv) is refit on all 2022-2025 rows and projected over
2026-01..2028-12. Low and high scenarios scale the base path by the 80% interval of the quality-adjusted
index forecast selected in notebook 07.
"""
import json

import numpy as np
import pandas as pd

from . import config
from .boosting import BoostedModel, required_columns
from .compare import interval_multipliers, profile_frame, representative_profile
from .features import build_features, build_target, select_track
from .hybrid import StructureModel, adjust_target, oracle_index
from .ts_index import ALPHAS, forecast, log_index

FUTURE = pd.date_range('2026-01-01', '2028-12-01', freq='MS')
LEADERBOARD_PATH = config.METRICS_DIR / 'leaderboard.csv'


def chosen_models(leaderboard_path=LEADERBOARD_PATH):
    """{track: model_id} of the models marked as chosen in the saved leaderboard."""
    board = pd.read_csv(leaderboard_path)
    chosen = board[board['is_chosen_model']]
    return dict(zip(chosen['track'], chosen['model_id']))


def _last_run(track, model_id):
    path = config.METRICS_DIR / f'{track}__{model_id}.json'
    return json.loads(path.read_text(encoding='utf-8'))[-1]


class ScenarioModel:
    """Chosen model of one track, refit on all months, plus the index forecast used for the bands."""

    def __init__(self, track, model_id, model, index_forecast, index_candidate, track_df):
        self.track, self.model_id, self.model = track, model_id, model
        self.index_forecast, self.index_candidate, self.track_df = index_forecast, index_candidate, track_df

    def profile(self, district, building_type, area_sqm):
        """Representative contract (median floor and age of the group) and the rows that support it."""
        return representative_profile(self.track_df, district, building_type, area_sqm)

    def project(self, profile, months=FUTURE):
        """Monthly base / low / high prediction (10,000 KRW) of a profile; Wolse is monthly rent."""
        frame = profile_frame(profile, months)
        if self.model_id == 'hybrid_hgb':
            base_log = self.model.predict_log(frame) + np.log(self.index_forecast['mean'].to_numpy() / 100)
            base = np.expm1(np.maximum(base_log, 0.0))
        else:
            base = np.expm1(np.maximum(self.model.predict(frame[required_columns('detrended')]), 0.0))
        base = pd.Series(base, index=pd.DatetimeIndex(months))
        m_lo, m_hi = interval_multipliers(self.index_forecast)
        return pd.DataFrame({'base': base, 'low': (base + 1) * m_lo.to_numpy() - 1,
                             'high': (base + 1) * m_hi.to_numpy() - 1})


def fit_scenario_model(df, track, model_id=None):
    """Refit the chosen model of `track` on all rows of `df` (the cleaned dataset)."""
    model_id = model_id or chosen_models()[track]
    df_t = select_track(df, track)
    X_all = build_features(df_t, track, include_time=True)
    y_all = build_target(df_t, track)

    index_all = oracle_index(df, track)           # index fitted on all months: legitimate for the final refit
    report = json.loads((config.METRICS_DIR / f'ts_index_{track}.json').read_text(encoding='utf-8'))
    candidate = report['rolling_origin']['selected']
    index_forecast = np.exp(forecast(candidate, log_index(index_all), len(FUTURE), ALPHAS))

    if model_id == 'hybrid_hgb':
        params = _last_run(track, 'hybrid_hgb')['meta']['structure_params']
        y_adj = adjust_target(y_all, df_t['contract_month'], index_all)
        model = StructureModel('hgb', params).fit(X_all, y_adj)
    elif model_id == 'hgb_detrended':
        params = _last_run(track, 'hgb_detrended')['meta']['params']
        model = BoostedModel(variant='detrended', **params).fit(X_all[required_columns('detrended')], y_all)
    else:
        raise NotImplementedError(f'projection for {model_id} is not implemented')
    return ScenarioModel(track, model_id, model, index_forecast, candidate, df_t)
