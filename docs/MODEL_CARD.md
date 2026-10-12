# Model card: Seoul rent models (Jeonse and Wolse)

Separate from the main models: the renewal-only study (`notebooks/13_renewal_study.ipynb`) is a different product and is not described here beyond the last section.

## Model details
- **Models.** One per track, chosen by the selection protocol below: `hgb_detrended_ld_median_recal` for Jeonse (deposit) and `hgb_detrended_ld_median` for Wolse (monthly rent). Both are HistGradientBoosting models with median loss (`quantile = 0.5`) on `log1p(target)` after removing a log-linear month trend (the trend is added back at prediction time), with a grouped neighbourhood feature (`legal_dong` within district; rare levels go to the OTHER level of their district). The Jeonse model also multiplies its predictions by one global scale factor estimated on the last CV fold of its fit window (`_recal`).
- **Targets and units.** Deposit of Jeonse contracts and monthly rent of Wolse contracts, in units of 10,000 KRW (1,000 = 10 million KRW). The annual cost of a Wolse contract is 12 x the monthly rent.
- **Random seed.** `RANDOM_SEED = 42`, defined in `src/rent_model/config.py`. Environment: Python 3.13.7, exact versions in `constraints.txt`.

## Intended use
- Market-level scenario analysis of Seoul rents and deposits for a representative contract (district, building type, area, median floor and age of the group), and a portfolio demonstration of a leakage-safe modeling workflow.
- A point estimate with an interval for a single contract (conformal interval, see Uncertainty), with the limits stated below.

## Out-of-scope use
- **Individual legal or financial decisions** (setting or negotiating a rent or deposit, credit or tax decisions, legal advice). The models describe a sample of contracts, not the price of a given dwelling.
- Forecasts of 2026-2028: the projections are extrapolation scenarios from 48 months of data.
- Other cities or other periods, new contracts of a specific building (the data has no building identifier), and renewals (the renewal-only study is a separate product that uses the previous price).

## Data
- **Source.** Seoul Open Data Plaza, "Seoul Real Estate Jeonse and Monthly Rent Price Information" (dataset OA-21276, <https://data.seoul.go.kr/dataList/OA-21276/S/1/datasetView.do>). The terms of use are not restated here; check them on the source page.
- **Sample.** 400,000 rows x 23 columns with about 100,000 records per year (2022-2025); it is not a proportional sample of the market, so counts per month or year are not market volume.
- **Cleaning.** 387,326 rows after the documented cleaning (12,674 rows, 3.17%, dropped): 176,783 Jeonse and 210,543 Wolse contracts. Price and area outliers are kept; only impossible values are removed with domain-based ranges. 24 exact duplicate rows remain (they differ only in dropped columns).

## Training and evaluation protocol
- **Features.** `district_name`, `building_type`, `contract_type`, `renewal_right_used`, `log_area`, `floor` (with a missing flag), `building_age`, the grouped neighbourhood and a time index used only to remove the trend. Forbidden: the other track's target, `monthly_rent` in the Jeonse track and the deposit in the Wolse track, every `previous_*` column, dates and periods, lot numbers and identifiers (enforced by tests).
- **Splits.** Strict temporal split: TRAIN 2022-01-01 to 2024-12-31 (138,848 Jeonse and 152,714 Wolse rows), TEST 2025 (37,935 and 57,829 rows). Everything (grouping, encoding, scaling, hyperparameters, recalibration factor) is fit on the fit window only; settings are tuned by random search with a 3-fold expanding-window time CV inside each fit window (12 candidates per search).
- **Selection protocol.** Candidates are compared on two rolling-origin pseudo-tests (fit before 2023 and test 2023; fit before 2024 and test 2024) with a one-standard-error rule on the most conservative of the row, month-block and district-block bootstrap standard errors; ties go to the model with fewer parameters, then to the smaller absolute bias. The winner must extrapolate: its projected path for a fixed profile must change beyond the last level. The final target year, 2025, is reported once after the choice.
- **Disclosure.** 2025 was inspected in earlier iterations of this project (it informed the candidate families and several design choices), so the 2025 figures are not those of a fully untouched hold-out. The final selection itself used only the pseudo-tests.

## Metrics
| | Jeonse (`hgb_detrended_ld_median_recal`) | Wolse (`hgb_detrended_ld_median`) |
|---|---:|---:|
| WAPE 2025 (%) | 14.14 | 40.80 |
| Median bias 2025 (%) | 2.35 | -5.76 |
| Aggregate bias 2025 (%) | 1.07 | -8.73 |
| MdAPE 2025 (%) | 10.70 | 27.34 |
| Pooled pseudo-test WAPE 2023-2024 (%) | 15.09 | 39.23 |
| Best baseline, WAPE 2025 (%) | 27.25 | 46.96 |
| Best linear model, WAPE 2025 (%) | 23.09 | 49.10 |

Source of every number: `reports/metrics/leaderboard.csv` (written by `notebooks/09_model_comparison_projection.ipynb`).

## Uncertainty
- **Contract-level interval (notebook 12).** Split conformal intervals on the `log1p` scale, calibrated on the last 6 months of the fit window. On 2025 they cover 0.8186 and 0.9595 (Jeonse) and 0.8095 and 0.9542 (Wolse) at the 80% and 95% levels. Coverage is not uniform: Jeonse detached houses are covered 0.5582 at the 80% level, and the Wolse pseudo-test of 2024 covers 0.7504 (unscaled) because the error scale grew after the calibration window. The Wolse intervals are wide (median relative width 1.7508 of the prediction for the scaled 80% interval).
- **Scenario bands.** The low/high scenarios scale the base path by the 80% interval of the quality-adjusted index forecast: they describe the uncertainty of the price level only, not the dispersion of individual contracts or the uncertainty of the model trend.

## Known failure modes
- **Cheap and low-rent Wolse contracts.** The Wolse WAPE is 40.80% against an MdAPE of 27.34%, and the median MAPE by district and quarter is 86.81%: relative errors are large for low rents, and the absolute error is concentrated in high-priced contracts (MAE 55.7511 against 14.6990 in the high and low predicted-price terciles, 10,000 KRW, 2025).
- **Level persistence in Jeonse.** The Jeonse hybrid models (structure model plus a naive index forecast) keep their last level (0.0000% change between the annual means of 2026 and 2028), so they cannot be used for projections; the chosen model extrapolates a log-linear trend (+15.0132% between 2026 and 2028 for a fixed profile, 8.17% per year), which is steeper than the flat index forecast that defines its band.
- **Extrapolation.** The trends rest on 48 months of data; the scenarios are not forecasts and the bands do not include the uncertainty of the trend.
- **Unseen or rare groups.** Neighbourhoods unseen in the fit window map to the OTHER level of their district; new units (not seen in the fit window) have a higher error than repeat units (2025 WAPE of the chosen models: Jeonse 15.2375% against 12.4268%, Wolse 43.2485% against 35.5252%).
- **Sample.** The data is not proportional to the market, has no building identifier for most rows, and the neighbourhood names are an automatic romanization.

## Not in the model
Macro variables (interest rates, policy, supply), building quality beyond type, floor and age, and the previous price (except in the separate renewal-only product).
