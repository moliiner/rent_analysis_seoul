"""Explorer of the 2026-2028 extrapolation scenarios.

Run with:  streamlit run app/streamlit_app.py
It needs no network access: it reads data/Seoul_Rentals_2022_2025.csv and reports/metrics, refits the
chosen model of each track on 2022-2025 (about a minute the first time, then cached) and projects the
selected profile. The results are extrapolation scenarios from 48 months of data, not forecasts.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd
import streamlit as st

from rent_model import config
from rent_model.compare import annual_cash_cost
from rent_model.data import load_clean
from rent_model.projection import FUTURE, LEADERBOARD_PATH, fit_scenario_model

TRACK_LABELS = {'jeonse': 'Jeonse (deposit)', 'wolse': 'Wolse (monthly rent)'}

st.set_page_config(page_title='Seoul rent scenarios 2026-2028', layout='wide')


@st.cache_data(show_spinner='Loading the cleaned dataset...')
def load_data():
    return load_clean()


@st.cache_resource(show_spinner='Fitting the chosen models on 2022-2025 (about a minute the first time)...')
def load_models():
    df = load_data()
    return {track: fit_scenario_model(df, track) for track in config.TRACKS}


@st.cache_data
def load_leaderboard():
    return pd.read_csv(LEADERBOARD_PATH)


def observed_median(track_df, target, district, building_type, area_sqm, contract_type):
    rows = track_df[(track_df['district_name'] == district) & (track_df['building_type'] == building_type)
                    & (track_df['contract_date'].dt.year == 2025) & (track_df['contract_type'] == contract_type)
                    & track_df['leased_area_sqm'].between(area_sqm * 0.8, area_sqm * 1.2)]
    return (rows[target].median() if len(rows) else float('nan')), len(rows)


st.title('Seoul rent scenarios 2026-2028')
st.warning('Extrapolation scenarios from 48 months of data (2022-2025), not forecasts. The bands only reflect the '
           'uncertainty of the price index level, not the dispersion of individual contracts.')

df = load_data()
models = load_models()
board = load_leaderboard()

districts = sorted(df['district_name'].unique())
building_types = sorted(df['building_type'].unique())
with st.sidebar:
    st.header('Profile')
    district = st.selectbox('District', districts, index=districts.index('Mapo-gu'))
    building_type = st.selectbox('Building type', building_types, index=building_types.index('Apartment'))
    area = st.slider('Leased area (m2)', min_value=15, max_value=150, value=60, step=5)
    st.caption('Floor, building age and contract type are the medians / most frequent values of the '
               'selected district and building type.')

columns = st.columns(2)
for column, (track, scenario) in zip(columns, models.items()):
    with column:
        st.subheader(TRACK_LABELS[track])
        row = board[(board['track'] == track) & (board['model_id'] == scenario.model_id)].iloc[0]
        st.caption(f"Chosen model: `{scenario.model_id}` (TEST 2025 WAPE {row['wape_pct']:.2f}%, "
                   f"index candidate for the bands: {scenario.index_candidate})")
        profile, n_group = scenario.profile(district, building_type, float(area))
        projection = scenario.project(profile)
        factor = 12 if track == 'wolse' else 1
        annual = pd.DataFrame({c: annual_cash_cost(projection[c], track) for c in ['low', 'base', 'high']})
        annual['base (million KRW)'] = annual['base'] / 100
        annual.index.name = 'year'
        st.markdown('**Annual cash cost** (Jeonse = deposit, Wolse = 12 x monthly rent), in 10,000 KRW')
        st.dataframe(annual.round(1))
        chart = projection[['low', 'base', 'high']] * factor
        st.line_chart(chart, y_label='10,000 KRW' + (' per year' if track == 'wolse' else ''))
        target = config.TRACKS[track]['target']
        median, n_comp = observed_median(scenario.track_df, target, district, building_type, float(area), profile['contract_type'])
        st.caption(f"Observed 2025 median of comparable contracts (same district, type and contract type, area within 20%): "
                   f"{median:,.1f} (10,000 KRW{' per month' if track == 'wolse' else ''}), {n_comp} contracts. "
                   f"Rows of the district and building type in 2022-2025: {n_group}.")
        if n_comp < 30 or n_group < 100:
            st.info('Thin support for this profile: treat the scenario with extra caution.')

with st.expander('How to read this'):
    st.markdown(
        '- Money columns are in units of 10,000 KRW (1,000 = 10 million KRW).\n'
        '- *base* is the chosen model projected over 2026-2028; *low* and *high* scale it by the 80% interval of the '
        'quality-adjusted index forecast.\n'
        '- The models underpredicted the 2025 level in the back-test, so the base scenario may be conservative.\n'
        f'- Projection months: {FUTURE[0]:%Y-%m} to {FUTURE[-1]:%Y-%m}.')
