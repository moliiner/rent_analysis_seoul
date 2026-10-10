# Seoul Rental Market EDA

Portfolio project: exploratory data analysis of Jeonse and Monthly rent (Wolse) contracts in Seoul, 2022–2025. It is meant to be read by tech recruiters in South Korea, so it must look professional: clear business storytelling, reproducible code, and conclusions that match the data.

## Project files
- `Notebook.ipynb`: the analysis (English). Main deliverable.
- `data/Seoul_Rentals_2022_2025.csv`: dataset (never edit it, never commit changes to it).
- `Notebook_original_backup.ipynb`: backup of the notebook before the review. Do not modify it.
- `CLAUDE.md`: this file.

## Language
- Notebook content (code comments, markdown, plot titles): **English**.
- Chat replies to the user: **Spanish**, concise.

## Environment
- Python 3.13, recent pandas (string dtype is `str`, not `object`).
- Packages: pandas, numpy, matplotlib, seaborn, scipy, jupyter, nbconvert, nbformat.
- Run the notebook with: `jupyter nbconvert --to notebook --execute --inplace Notebook.ipynb`

## Dataset facts (do not guess beyond this)
- Source: Seoul Open Data Plaza, "Seoul Real Estate Jeonse and Monthly Rent Price Information" (OA-21276).
- 400,000 rows × 23 columns. It is a sample with about **100,000 records per year** (2022–2025), not a proportional random sample, so counts per month or year are NOT market volume.
- Money columns are in **10,000 KRW** (`deposit_10k_krw = 16000` means 160,000,000 KRW). Area is in m².
- `lease_type` exact values: `'Jeonse (deposit only)'` and `'Monthly rent'`.
- `monthly_rent_10k_krw` is 0 for every Jeonse contract: analyze it on Wolse rows only.
- `lease_type` was inferred from `monthly_rent == 0` in 2022, 2024 and 2025 (original value only in 2023).
- `building_type` values: `Apartment`, `Officetel`, `Row house / Villa (multi-unit)`, `Detached / Multi-household house`.
- `floor`: negative = basement; NaN mostly for detached houses (keep NaN, do not treat as error).
- `leased_area_sqm` arrives as text with comma decimals in the raw file and must be converted to float.
- `contract_date` arrives as text; use it for time analysis, not `registration_year`.
- `legal_dong_name` is an automatic romanization; dongs are only unique together with `district_name`.
- `building_name` exists only for 2023 rows. `previous_*` columns only exist for renewals.

## Rules for working on the notebook
1. **Numbers**: never write a number in markdown that does not appear in an executed output of the current notebook. No placeholders (`[X]`) and no values estimated by eye from charts.
2. **Verification**: before saying a task is done, re-run the whole notebook top to bottom in a fresh kernel. It must finish with no errors.
3. **Removing rows**: only with a documented reason and the exact number of rows dropped (markdown + printed count). Never drop rows just because they are IQR outliers.
4. **Outliers**: deposit, rent and area outliers are real market values, keep them (use log transforms later). Remove only impossible values using a domain-based range (as done with `year_built` and `floor`).
5. **Statistics**: use the median (not the mean) for skewed variables. Compare districts separately for Jeonse (deposit) and Wolse (monthly rent), because pooled deposits mix contract types.
6. **Order**: an "Interpretation" markdown goes AFTER the code cell it interprets.
7. **Voice**: neutral analytical English. No second person ("you"), no coaching notes inside the notebook.
8. **Structure**: keep the numbered `##` sections (Imports, Read dataset, Data exploration, Data cleaning, Univariate, Bivariate, Correlation, Temporal and geographic, Key EDA Insights, Limitations).
9. **Scope**: do not refactor working code for style (for example, do not rewrite cells into method chains). Only change what was asked.
10. **Safety**: make a backup before big edits. Do not commit or push unless explicitly asked. No personal paths (such as `C:\Users\...`) in the notebook; use `data/...`.

## Modeling notes (for later phases)
- Split train/test before scaling or encoding (avoid data leakage).
- Apply a log transform (`np.log1p`) to deposit, monthly rent and probably area.
- Model Jeonse and Wolse separately, or include `lease_type` explicitly.
- **Leakage warning**: do not use `monthly_rent_10k_krw` or `previous_monthly_rent_10k_krw` (and be careful with `previous_deposit_10k_krw`) as features to predict `lease_type`.
- Fix a random seed and document it.
- Likely main predictors: `district_name`, `building_type`, `leased_area_sqm`; secondary: `floor`, `year_built`.

## Skills (if installed in this project)
- `data-analysis-jupyter`: notebook conventions (data quality checks, documented assumptions, reproducibility).
- `verification-before-completion`: run it before reporting that work is finished.
- If a skill convention conflicts with an explicit instruction from the user, the user's instruction wins.

## Modeling protocol
1. Two tracks: `jeonse` (target `deposit_10k_krw`) and `wolse` (target `monthly_rent_10k_krw`). Models are trained per track. Annual cash cost for Wolse = 12 x monthly rent. Deposit is NOT a feature in the wolse track.
2. Target is `y = log1p(target)`; predictions are back-transformed with `expm1`.
3. Strict temporal split by `contract_date`: TRAIN 2022-01-01..2024-12-31, TEST 2025. Never a random split. Everything (imputation, encoding, scaling, clustering, hyperparameters) is fit on TRAIN only.
4. Tuning: 3-fold expanding-window time CV inside TRAIN only. Final projection refits on 2022-2025.
5. Forbidden features: `lease_type` within a track, the other track's target, deposit in the wolse track, `monthly_rent` and every `previous_*` column, `contract_period`/`contract_start`/`contract_end`, `registration_year`, raw `contract_date`, `legal_dong_code`, lot-number columns, zone columns, `row_id`.
6. Allowed features: `district_name`, `building_type`, `contract_type`, `renewal_right_used`, `log_area`, `floor`, `floor_missing`, `building_age` = max(contract_year - year_built, 0), `month_of_year` (sin/cos for linear models), `t` = months since 2022-01 (only in "time" variants), optional `legal_dong` (combined with `district_name`) for models that handle high cardinality.
7. Primary metric WAPE; also MAE, MdAPE, median bias %, aggregate bias %, MAPE by district x quarter, RMSE and R2 on log scale.
8. Model selection: lowest WAPE on TEST, one-standard-error rule with paired bootstrap, ties go to the simpler and less biased model. A projection model must be able to extrapolate a trend.
   - **Amended selection protocol (supersedes "on TEST" and the label-based extrapolation clause above).** Candidates are compared on two rolling-origin pseudo-tests: fit on rows before 2023-01-01 and test on 2023; fit on rows before 2024-01-01 and test on 2024. Settings are chosen by the 3-fold time CV inside each fit window. The one-standard-error rule uses the most conservative of the row, month-block and district-block bootstrap standard errors of the paired WAPE difference; ties go to the model with fewer parameters, then to the smaller absolute aggregate bias. The winner is then evaluated once on 2025. 2025 is never used to choose between candidates.
   - **Extrapolation is behavioural.** A model extrapolates if its projected path for a fixed profile over 2026-2028 changes beyond the last level (the annual means are not constant); otherwise it is labelled "level persistence". A projection model must extrapolate.
   - **Disclosure.** 2025 was inspected in earlier iterations of this project (it informed several modeling choices, for example the candidate families), so the 2025 figures are not those of a fully untouched hold-out. This is disclosed in the README.
9. Projections for 2026-2028 are extrapolation scenarios (only 48 months of data) and must be labelled as such.
10. Every model notebook ends by saving predictions and metrics under `reports/` (`reports/predictions/{track}__{model_id}.parquet`, `reports/metrics/{track}__{model_id}.json`), never by overwriting another model's files.

Code lives in `src/rent_model/` (`config`, `data`, `features`, `splits`, `metrics`, `io`); `RANDOM_SEED = 42` is defined only in `config.py`. Run the tests with `pytest -q`.
