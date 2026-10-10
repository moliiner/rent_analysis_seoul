# Seoul Rental Market: How Much Will a Tenant Pay in the Coming Years?

End-to-end data science project on 387,326 cleaned Seoul rental contracts (2022-2025): leakage-safe EDA and modeling of **Jeonse** (deposit-only) and **Wolse** (monthly rent) contracts, 36 valid models compared per track (baselines, linear, forests, boosting, K-Means, hierarchical clustering, Gaussian mixtures, time series and a hybrid), a quality-adjusted price index, and clearly labelled 2026-2028 extrapolation scenarios.

## Key results

- **Unconstrained choice** (selected on the 2023-2024 pseudo-tests, then evaluated once on 2025): `hgb_legal_dong` for Jeonse (2025 WAPE 15.84%) and `hgb_legal_dong_median` for Wolse (2025 WAPE 41.31%). Neither uses time information, so neither can extrapolate a trend.
- **Chosen model for future years** (must extrapolate): `rf_detrended` for Jeonse (WAPE 17.91%) and `hgb_detrended_median` for Wolse (WAPE 42.48%). Requiring extrapolation costs 2.0737 (Jeonse) and 1.1682 (Wolse) WAPE points against the unconstrained choice.
- **A median-aligned loss matters in Wolse.** The log of the Wolse rent is left-skewed (skewness -0.4006), so a squared loss on `log1p` predicts below the median that WAPE rewards. The median-loss variants take the first five places in Wolse, and the old Wolse choice `hgb_legal_dong` had an aggregate bias of -21.67% in 2025 against -9.57% for `hgb_legal_dong_median`.
- **Raw monthly medians mislead.** The quality-adjusted index ends 2025 at 103.96 (Jeonse) and 95.28 (Wolse), while the raw monthly median rebased to 2022-01 reaches 142.58 and 110.00: most of the apparent rise is a change in the sample mix.
- **Supervised beats unsupervised.** Boosting is the best family in both tracks; unsupervised features add little or nothing to boosting, and pure unsupervised predictors are the weakest family.
- **The 2026-2028 numbers are scenarios, not forecasts**: they extrapolate 48 months of data. The Wolse models underpredict in every evaluated year, not only in 2025 (see Limitations).

![Leaderboard](reports/figures/09_leaderboard.png)

## Business question

How much will a tenant pay in Seoul in the coming years? The answer differs by contract type: a **Jeonse** tenant pays a large lump-sum deposit and no monthly rent, while a **Wolse** tenant pays a smaller deposit plus monthly rent. The two are modeled separately, because pooled deposits mix both types. The annual cash cost is the deposit for Jeonse and 12 x the monthly rent for Wolse.

## Data

- **Source:** Seoul Open Data Plaza, "Seoul Real Estate Jeonse and Monthly Rent Price Information" (OA-21276), file `data/Seoul_Rentals_2022_2025.csv`.
- **Sample design:** 400,000 rows x 23 columns, with about 100,000 records per year (2022-2025). It is a sample and **not proportional to the market**, so counts per month or per year are not market volume.
- **Units:** money columns are in units of 10,000 KRW (a deposit of 16,000 means 160,000,000 KRW); area is in m2.
- **Contract type:** `lease_type` is `Jeonse (deposit only)` or `Monthly rent`; it was inferred from `monthly_rent == 0` in 2022, 2024 and 2025.

**Cleaning (documented in `Notebook.ipynb`; every dropped row is counted):**

| Step | Rows dropped | Rows left |
|---|---:|---:|
| Raw file | - | 400,000 |
| Exact duplicates | 10 | |
| Missing `year_built` (12,650 missing in the raw file) | | 387,342 |
| Partial month (contracts from 2026-01-01) | 11 | 387,331 |
| Impossible `year_built` (outside 1850-2026) | 4 | 387,327 |
| Impossible `floor` (above 100; one row with floor 302) | 1 | 387,326 |

In total 12,674 rows (3.17%) were dropped and 96.83% were retained: 176,783 Jeonse and 210,543 Wolse contracts. Price and area outliers are real market values and are **kept** (log transforms are used instead); only impossible values were removed, with domain-based ranges.

## Methodology

1. **EDA** (`Notebook.ipynb`): distributions, data quality, bivariate and temporal analysis, using medians for skewed variables and separate Jeonse / Wolse comparisons.
2. **Strict temporal split.** TRAIN is 2022-01-01 to 2024-12-31 (138,848 Jeonse and 152,714 Wolse rows) and TEST is 2025 (37,935 and 57,829 rows). Never a random split. Tuning uses a 3-fold expanding-window time CV inside TRAIN.
3. **Leakage controls.** Scaling, encoding, imputation, clustering and hyperparameters are fit on TRAIN only; the deposit is never a feature of the Wolse track; `monthly_rent`, every `previous_*` column, contract dates and period columns, lot numbers and `row_id` are forbidden (enforced by tests); the target never builds cluster features.
4. **Features:** `district_name`, `building_type`, `contract_type`, `renewal_right_used`, `log_area`, `floor` (with a missing flag), `building_age`, seasonality and time index `t` only in the "time" variants, and an optional neighbourhood (`legal_dong` within district).
5. **Target:** `log1p(target)`, back-transformed with `expm1`. **Primary metric: WAPE** (also MAE, MdAPE, median and aggregate bias, RMSE and R2 on the log scale, and MAPE by district x quarter).
6. **Models compared** (notebooks `01` to `08`, plus the median-loss variants of notebook `10`): naive baselines (global, district, district x type, + area quintile, last-year drift); OLS, Ridge, Lasso and Elastic Net (static and time variants, plus interactions); Random Forest (static, time, detrended); HistGradientBoosting (static, time, detrended, neighbourhood); K-Means, Ward clustering of districts and Gaussian mixtures, used as pure predictors and as features; a quality-adjusted (hedonic time-dummy) price index forecast with naive, drift, ETS, ARIMA, SARIMA and Theta; and a **hybrid** that adds a forecasted index to a structure model without time.
7. **Selection rule** (`notebooks 10 and 09`, `CLAUDE.md` rule 8): candidates are compared on two rolling-origin pseudo-tests (fit before 2023 and test 2023; fit before 2024 and test 2024), with settings chosen by 3-fold time CV inside each fit window. The rule is a one-standard-error rule on the pooled pseudo-test WAPE using the **largest of the row, month-block and district-block bootstrap standard errors**; ties go to the model with fewer parameters, then to the smaller absolute aggregate bias. The winner must **behave** as an extrapolating model: its projected path for a fixed profile over 2026-2028 must change beyond the last level (otherwise it is labelled level persistence). Runs that use TEST information (oracle diagnostics) are excluded.
8. **2025 is the final report year.** The selected models are evaluated once on 2025 and no 2025 row selects among the candidates (an assertion in notebook 10 checks this). Disclosure: the 2025 results were inspected in earlier iterations of this project, before the selection protocol was amended; the pseudo-tests remove that influence from the current choice, but the candidate list and the saved hyperparameters come from those earlier iterations.

## Results

The WAPE, bias and rank columns are the 2025 hold-out and are descriptive: they did not select the models. The pseudo-test column is the pooled 2023-2024 WAPE of the candidates that did (38 models x 2 tracks = 76 runs; 72 valid and 4 oracle diagnostics excluded). Full table: `reports/metrics/leaderboard.csv`.

### Jeonse (target: deposit)

| Rank | Model | Family | WAPE % | Median bias % | Aggregate bias % | Pseudo-test WAPE % | Handles time | Note |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | `hgb_legal_dong_median` | boosting | 15.54 | -6.36 | -7.70 | 16.60 | none |  |
| 2 | `hgb_legal_dong` | boosting | 15.84 | -7.37 | -8.57 | 16.38 | none | unconstrained choice |
| 3 | `hybrid_hgb_median` | time-series/hybrid | 17.09 | -2.15 | -3.93 | 17.95 | forecasted index |  |
| 4 | `hgb_time_median` | boosting | 17.22 | -2.53 | -3.52 | 17.34 | time feature in trees (flat beyond 2024) |  |
| 5 | `hgb_time` | boosting | 17.31 | -3.95 | -4.01 | 17.76 | time feature in trees (flat beyond 2024) |  |
| 6 | `hgb_time_kmeans` | unsupervised-assisted | 17.34 | -3.81 | -4.02 | - | time feature in trees (flat beyond 2024) |  |
| 7 | `hgb_time_tier` | unsupervised-assisted | 17.35 | -3.90 | -3.97 | - | time feature in trees (flat beyond 2024) |  |
| 8 | `hgb_time_gmm` | unsupervised-assisted | 17.38 | -3.90 | -4.05 | - | time feature in trees (flat beyond 2024) |  |
| 9 | `hybrid_hgb` | time-series/hybrid | 17.44 | -3.35 | -5.03 | 17.93 | forecasted index |  |
| 10 | `hgb_detrended` | boosting | 17.65 | 5.23 | 3.39 | 18.35 | log-linear trend |  |
| 13 | `rf_detrended` | forest | 17.91 | 5.25 | 2.75 | 18.53 | log-linear trend | chosen |
| 17 | `ridge_time_rich` | linear | 23.09 | -8.26 | -10.61 | 23.29 | linear time feature |  |
| 29 | `baseline_district_type_area` | baseline | 27.25 | -4.76 | -10.29 | 26.36 | none |  |

### Wolse (target: monthly rent)

| Rank | Model | Family | WAPE % | Median bias % | Aggregate bias % | Pseudo-test WAPE % | Handles time | Note |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | `hgb_legal_dong_median` | boosting | 41.31 | -6.49 | -9.57 | 39.75 | none | unconstrained choice |
| 2 | `hgb_detrended_median` | boosting | 42.48 | -6.10 | -10.23 | 40.18 | log-linear trend | chosen |
| 3 | `hgb_static_median` | boosting | 42.58 | -6.81 | -11.03 | 40.38 | none |  |
| 4 | `hgb_time_median` | boosting | 42.67 | -4.96 | -9.15 | 40.71 | time feature in trees (flat beyond 2024) |  |
| 5 | `hybrid_hgb_median` | time-series/hybrid | 43.05 | -9.58 | -13.69 | 40.72 | forecasted index |  |
| 6 | `hgb_legal_dong` | boosting | 44.33 | -16.91 | -21.67 | 41.49 | none |  |
| 7 | `hgb_detrended` | boosting | 45.33 | -16.92 | -22.48 | 41.58 | log-linear trend |  |
| 8 | `hgb_static` | boosting | 45.54 | -17.65 | -23.20 | 42.36 | none |  |
| 9 | `hgb_time_gmm` | unsupervised-assisted | 45.56 | -18.17 | -22.30 | - | time feature in trees (flat beyond 2024) |  |
| 10 | `hgb_time` | boosting | 45.74 | -18.20 | -22.46 | 42.37 | time feature in trees (flat beyond 2024) |  |
| 17 | `baseline_district_type_area` | baseline | 46.96 | -7.41 | -14.58 | 45.42 | none |  |
| 18 | `ridge_time_rich` | linear | 49.10 | -20.28 | -28.97 | 46.16 | linear time feature |  |

### Winner

- **Jeonse: `rf_detrended`**: a random forest on the detrended target with the log-linear trend added back. WAPE 17.91%, median bias 5.25%, aggregate bias 2.75%. It is the extrapolating model chosen on the pseudo-tests (it ties with `hgb_detrended` and has fewer parameters); the unconstrained choice `hgb_legal_dong` has 15.84% (difference 2.0737 points, 95% bootstrap interval 1.3463 to 2.7518). The previous choice, `hybrid_hgb`, was dropped because its projected path is constant (level persistence).
- **Wolse: `hgb_detrended_median`**: boosting with a median loss on the detrended target, with the trend added back. WAPE 42.48%, median bias -6.10%, aggregate bias -10.23%. The unconstrained choice `hgb_legal_dong_median` has 41.31% (difference 1.1682 points, interval 0.6377 to 1.6393).
- **What changed against the previous protocol** (selection on 2025 with row bootstrap only): the Jeonse unconstrained choice is unchanged; the Jeonse extrapolating choice changed from `hybrid_hgb` to `rf_detrended`; both Wolse choices changed to median-loss variants. The 2025 WAPE of the previous Wolse choices was 44.33% and 45.33%, against 41.31% and 42.48% now. Notebook 09 lists the changes.

### Supervised vs unsupervised

| Family | Jeonse best model | Jeonse WAPE % | Wolse best model | Wolse WAPE % |
|---|---|---:|---|---:|
| boosting | `hgb_legal_dong_median` | 15.54 | `hgb_legal_dong_median` | 41.31 |
| unsupervised-assisted | `hgb_time_kmeans` | 17.34 | `hgb_time_gmm` | 45.56 |
| forest | `rf_time` | 17.91 | `rf_detrended` | 45.76 |
| time-series/hybrid | `hybrid_hgb_median` | 17.09 | `hybrid_hgb_median` | 43.05 |
| linear | `ridge_time_rich` | 23.09 | `ridge_time_rich` | 49.10 |
| baseline | `baseline_district_type_area` | 27.25 | `baseline_district_type_area` | 46.96 |
| pure unsupervised | `gmm_median` | 37.10 | `gmm_median` | 51.87 |

Unsupervised features added to a supervised model (change of the WAPE against the same model without them; negative is better):

| Model | vs | Jeonse WAPE change (points) | Wolse WAPE change (points) |
|---|---|---:|---:|
| `ridge_time_gmm` | `ridge_time` | -0.65 | -0.90 |
| `ridge_time_kmeans` | `ridge_time` | -0.07 | -0.38 |
| `ridge_time_tier` | `ridge_time` | -0.00 | -0.00 |
| `hgb_time_gmm` | `hgb_time` | +0.07 | -0.18 |
| `hgb_time_kmeans` | `hgb_time` | +0.03 | +0.04 |
| `hgb_time_tier` | `hgb_time` | +0.04 | +0.00 |

Supervised boosting wins. Cluster, tier and mixture features do not help boosting beyond bootstrap noise (the only gain is `hgb_time_gmm` in Wolse), help ridge a little, and pure unsupervised predictors (`kmeans_median`, `gmm_median`) are the weakest family. The unsupervised analyses are still useful to describe market segments (see notebooks `05` and `06`).

### Quality-adjusted price index

A hedonic time-dummy index (month coefficients of a regression on structural controls, rebased to 100 at 2022-01) removes the composition effect of a non-proportional sample. At 2025-12 it is 103.96 (Jeonse) and 95.28 (Wolse), against 142.58 and 110.00 for the raw median rebased. See `notebooks/07_price_index_forecasting.ipynb`.

![Price index](reports/figures/07_price_index.png)

### Scenarios 2026-2028 (extrapolation, not forecasts)

The chosen model of each track is refit on 2022-2025. The base scenario is its projection; low and high scale it by the 80% interval of the index forecast. **They extrapolate only 48 months of data and are not forecasts.** Values are in units of 10,000 KRW (1,000 = 10 million KRW); Jeonse is the deposit, Wolse is the annual rent (12 x monthly rent) while the observed median is the monthly rent. The Wolse cost excludes the deposit.

**Jeonse (deposit):**

| Profile | Observed 2025 median | Comparable contracts | 2026 base (low - high) | 2026 base vs observed % | 2028 base (low - high) |
|---|---:|---:|---:|---:|---:|
| Apartment 60 m2, Gangnam-gu | 73,000.0 | 164 | 80,170.8 (74,499.9 - 86,316.5) | 9.8 | 93,804.0 (79,491.3 - 110,703.3) |
| Apartment 60 m2, Mapo-gu | 54,117.5 | 240 | 63,660.1 (59,157.0 - 68,540.1) | 17.6 | 74,485.5 (63,120.5 - 87,904.6) |
| Apartment 60 m2, Nowon-gu | 28,000.0 | 419 | 34,235.3 (31,813.6 - 36,859.7) | 22.3 | 40,057.1 (33,945.1 - 47,273.7) |
| Officetel 25 m2, Gangnam-gu | 24,500.0 | 44 | 24,183.3 (22,472.7 - 26,037.2) | -1.3 | 28,295.8 (23,978.3 - 33,393.6) |
| Officetel 25 m2, Mapo-gu | 22,000.0 | 33 | 27,501.6 (25,556.2 - 29,609.9) | 25.0 | 32,178.4 (27,268.5 - 37,975.7) |
| Officetel 25 m2, Nowon-gu | 23,500.0 | 2 | 16,272.2 (15,121.1 - 17,519.7) | -30.8 | 19,039.4 (16,134.3 - 22,469.7) |

**Wolse (annual rent):**

| Profile | Observed 2025 median | Comparable contracts | 2026 base (low - high) | 2026 base vs observed % | 2028 base (low - high) |
|---|---:|---:|---:|---:|---:|
| Apartment 60 m2, Gangnam-gu | 200.0 | 191 | 2,526.9 (2,127.5 - 3,010.8) | 5.3 | 2,612.5 (1,637.9 - 4,168.4) |
| Apartment 60 m2, Mapo-gu | 140.0 | 159 | 1,617.2 (1,360.9 - 1,927.7) | -3.7 | 1,672.1 (1,046.7 - 2,670.5) |
| Apartment 60 m2, Nowon-gu | 90.0 | 268 | 1,084.7 (912.1 - 1,293.7) | 0.4 | 1,121.6 (700.7 - 1,793.7) |
| Officetel 25 m2, Gangnam-gu | 80.0 | 276 | 959.4 (806.6 - 1,144.6) | -0.1 | 992.2 (619.3 - 1,587.5) |
| Officetel 25 m2, Mapo-gu | 73.0 | 235 | 918.1 (771.8 - 1,095.4) | 4.8 | 949.5 (592.4 - 1,519.4) |
| Officetel 25 m2, Nowon-gu | 58.0 | 29 | 673.0 (565.3 - 803.6) | -3.3 | 696.1 (433.2 - 1,115.9) |

For example, the 2026 base for an Apartment of 60 m2 in Gangnam-gu is 801.7 million KRW of Jeonse deposit. The Officetel of 25 m2 in Nowon-gu in Jeonse is supported by only 2 comparable contracts in 2025 and should not be relied on.

![Scenarios](reports/figures/09_projection_scenarios.png)

## Limitations and what I would do differently

- **Only 48 months of data.** Trends and seasonality are barely identifiable; the 12-month index forecast errors rest on a handful of overlapping origins, and the 2026-2028 results are extrapolations whose bands are wide, in particular for Wolse.
- **Underprediction is persistent in Wolse and not specific to 2025.** The aggregate bias of the squared-loss `hgb_legal_dong` in Wolse is -15.6548% (pseudo-test 2023), -15.7420% (2024) and -21.6673% (2025); with the median loss it is -6.2621%, -3.7336% and -9.5688%. In Jeonse the bias of `hgb_legal_dong` is 4.8101% in 2023, -8.3346% in 2024 and -8.5658% in 2025: it is present in the last two years but has the opposite sign in 2023, and the detrended models overpredict. Most of the Wolse bias is a retransformation effect: a Duan smearing factor or the median loss removes about half of it. A single global factor estimated on the pseudo-tests only improves Wolse (for `hgb_legal_dong`, factor 1.1493, 2025 WAPE 44.3335% to 41.9678%) but not Jeonse reliably, and it is not adopted. The Wolse scenarios are therefore probably conservative.
- **The Jeonse scenario mixes two trends.** The chosen Jeonse model adds back a log-linear trend of 8.17% per year fit on 48 months, while the band comes from a flat (naive) index forecast; the base path rises through the band. Treat the Jeonse base as an upper-trend extrapolation.
- **Row bootstrap understates uncertainty.** With month or district blocks the Jeonse difference between `hybrid_hgb` and `hgb_detrended` is a tie, and the selection uses the largest of the three standard errors.
- **The sample is not proportional to the market.** Absolute volumes cannot be inferred, and the index only partly corrects composition (neighbourhood and building quality are not controlled).
- **No macro variables.** Interest rates, policy changes, supply, and price indices of the sale market would be the first additions; they are the main drivers of Jeonse deposits.
- **Trees cannot extrapolate.** The most accurate models are therefore not the ones suitable for projection; the hybrid and detrended designs are a compromise that costs accuracy.
- **Scenario bands cover only the index level.** The dispersion of individual contracts around the model (the WAPE) is not included.
- **What I would do differently:** collect more years, add macroeconomic and building-level data, model Wolse deposit and rent jointly, calibrate prediction intervals for the target year, and re-evaluate with a longer rolling-origin backtest.

## Reproducibility

- **Environment:** Python 3.13.7; packages in `requirements.txt` (`pip install -r requirements.txt`). LightGBM and SHAP are not used; scikit-learn is the only boosting implementation.
- **Random seed:** `RANDOM_SEED = 42`, defined only in `src/rent_model/config.py`.
- **Tests:** `pytest -q` (160 tests: data cleaning, splits, leakage, metrics and every model module).
- **Run everything (tests and notebooks 00 to 08, 10 and 09, in that order):** `bash scripts/run_all.sh` (or `make all`). Each notebook can also be run with `python -m nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`. The notebooks must run in that order: later ones read the runs saved by earlier ones (notebook 10 writes the pseudo-test selection that notebook 09 reads). Re-running a notebook appends a new run to `reports/metrics/{track}__{model_id}.json`; every table reads the last run. `reports/predictions/` (parquet) is generated by the notebooks and not versioned.
- **Optional app:** `pip install streamlit` and `streamlit run app/streamlit_app.py` to choose a district, building type and area and see the scenarios (it refits the chosen models locally, about a minute the first time; no network needed).
- **CI:** `.github/workflows/ci.yml` installs the requirements and runs `pytest -q` only.

## Repository map

```
Notebook.ipynb                 EDA and cleaning (English)
data/                          Seoul_Rentals_2022_2025.csv (never edited)
notebooks/
  00_modeling_setup.ipynb      protocol, temporal split, folds, smoke-test baseline
  01_baselines.ipynb           naive baselines
  02_linear_models.ipynb       OLS, Ridge, Lasso, Elastic Net
  03_random_forest.ipynb       Random Forest (static, time, detrended)
  04_gradient_boosting.ipynb   HistGradientBoosting, quantile intervals
  05_kmeans_clustering.ipynb   K-Means predictor and features
  06_hierarchical_gmm.ipynb    Ward district tiers and Gaussian mixtures
  07_price_index_forecasting.ipynb   hedonic index and its forecast
  08_hybrid_model.ipynb        structure model + forecasted index
  10_selection_robustness.ipynb   pseudo-tests, block bootstrap, median-loss runs, bias by year
  09_model_comparison_projection.ipynb   leaderboard, selection, 2026-2028 scenarios
src/rent_model/                config, data, features, splits, metrics, io, baselines, linear,
                               forest, boosting, clustering, unsupervised, ts_index, hybrid,
                               recipes, pseudotest, compare, projection
tests/                         pytest suite
reports/metrics/               one JSON per model and track, ts_index_*.json, pseudotest_*.json, leaderboard.csv
reports/figures/               PNG charts
app/streamlit_app.py           optional scenario explorer
scripts/run_all.sh, Makefile   run the tests and the notebooks in order
.github/workflows/ci.yml       CI (tests only)
CLAUDE.md                      project conventions and modeling protocol
```
