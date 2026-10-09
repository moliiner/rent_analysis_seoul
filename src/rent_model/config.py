"""Central configuration of the modeling protocol (see "Modeling protocol" in CLAUDE.md)."""
from pathlib import Path

import pandas as pd

# Single source of the random seed. Every stochastic step imports it from here.
RANDOM_SEED = 42

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'data' / 'Seoul_Rentals_2022_2025.csv'
REPORTS_DIR = ROOT / 'reports'
PREDICTIONS_DIR = REPORTS_DIR / 'predictions'
METRICS_DIR = REPORTS_DIR / 'metrics'
FIGURES_DIR = REPORTS_DIR / 'figures'

# Temporal split: TRAIN 2022-01-01..2024-12-31, TEST 2025
TRAIN_START = pd.Timestamp('2022-01-01')
TRAIN_END = pd.Timestamp('2024-12-31')
TEST_YEAR = 2025
TEST_START = pd.Timestamp(f'{TEST_YEAR}-01-01')
TEST_END = pd.Timestamp(f'{TEST_YEAR}-12-31')
N_CV_SPLITS = 3

# Origin of the time index t (months since 2022-01)
TIME_ORIGIN = TRAIN_START

# Cleaning constants (identical to Notebook.ipynb, section 4)
PARTIAL_MONTH_CUTOFF = pd.Timestamp('2026-01-01')
MIN_YEAR_BUILT = 1850
MAX_YEAR_BUILT = 2026
MAX_FLOOR = 100

# Expected size of the cleaned dataset
EXPECTED_ROWS = 387_326
EXPECTED_ROWS_BY_TRACK = {'jeonse': 176_783, 'wolse': 210_543}

# Tracks: one model family per track
TRACKS = {
    'jeonse': {'lease_type': 'Jeonse (deposit only)', 'target': 'deposit_10k_krw'},
    'wolse': {'lease_type': 'Monthly rent', 'target': 'monthly_rent_10k_krw'},
}
TARGETS = [spec['target'] for spec in TRACKS.values()]

# Allowed features (rule 6 of the protocol)
CATEGORICAL_FEATURES = ['district_name', 'building_type', 'contract_type', 'renewal_right_used']
NUMERIC_FEATURES = ['log_area', 'floor', 'floor_missing', 'building_age']
SEASONAL_FEATURES = ['month_of_year', 'month_sin', 'month_cos']
TIME_FEATURE = 't'
LEGAL_DONG_FEATURE = 'legal_dong'
BASE_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + SEASONAL_FEATURES

# Forbidden features (rule 5 of the protocol). The targets are always forbidden as features;
# this also covers "the other track's target" and "deposit in the wolse track".
FORBIDDEN_FEATURES = {
    'lease_type',
    'deposit_10k_krw',
    'monthly_rent_10k_krw',
    'contract_period',
    'contract_start',
    'contract_end',
    'contract_date',
    'contract_month',
    'registration_year',
    'legal_dong_code',
    'legal_dong_name',
    'lot_type_code',
    'lot_type',
    'main_lot_number',
    'sub_lot_number',
    'floor_reported',
    'row_id',
}
FORBIDDEN_PREFIXES = ('previous_', 'zone', 'lot_')
