import numpy as np
import pandas as pd
import pytest

from rent_model import config
from rent_model.compare import annual_cash_cost
from rent_model.projection import FUTURE, chosen_models, fit_scenario_model


@pytest.fixture(scope='module')
def sample(clean_df):
    """Deterministic 25% subsample covering all 48 months (fast to fit)."""
    return clean_df.sample(frac=0.25, random_state=config.RANDOM_SEED)


def test_chosen_models_are_read_from_the_saved_leaderboard():
    chosen = chosen_models()
    assert set(chosen) == {'jeonse', 'wolse'}
    assert all(isinstance(m, str) and m for m in chosen.values())


@pytest.mark.parametrize('track', ['jeonse', 'wolse'])
def test_scenarios_have_ordered_bands_and_correct_annual_cost(sample, track):
    scenario = fit_scenario_model(sample, track)
    profile, n_rows = scenario.profile('Mapo-gu', 'Apartment', 60.0)
    projection = scenario.project(profile)
    assert len(projection) == 36 and projection.index[0] == pd.Timestamp('2026-01-01') and projection.index[-1] == pd.Timestamp('2028-12-01')
    assert (projection['low'] <= projection['base']).all() and (projection['base'] <= projection['high']).all()
    assert (projection['base'] > 0).all() and n_rows > 0
    annual = annual_cash_cost(projection['base'], track)
    assert list(annual.index) == [2026, 2027, 2028]
    factor = 12 if track == 'wolse' else 1
    assert annual.iloc[0] == pytest.approx(projection['base'].iloc[:12].mean() * factor)
    assert list(FUTURE) == list(projection.index)


def test_larger_apartments_cost_more(sample):
    scenario = fit_scenario_model(sample, 'jeonse')
    small, _ = scenario.profile('Mapo-gu', 'Apartment', 40.0)
    large, _ = scenario.profile('Mapo-gu', 'Apartment', 110.0)
    assert scenario.project(large)['base'].mean() > scenario.project(small)['base'].mean()
