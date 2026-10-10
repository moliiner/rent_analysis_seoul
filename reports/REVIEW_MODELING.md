# Senior review of the Seoul rental modeling project

Scope: read-only review of commit `1fab6fd` (CLAUDE.md, notebooks 00-09, `src/rent_model`, `reports/`, README, app). Evidence is cited as file and cell/line. Exploratory experiments were run in throwaway scripts outside the repository and are labelled "exploratory, not saved as a model run".

## 1. Executive verdict

**Overall rating: 7 / 10.** The engineering and the honesty of the reporting are well above a typical portfolio project: strict temporal split, a leakage guard that I verified on the real feature lists, flagged oracle runs, exact metric reproducibility, 118 passing tests and a README whose numbers are traceable. The rating is held back because several *conclusions* are weaker than they look and the models leave avoidable error on the table.

- **Trustworthy:** the family ranking (boosting > forest > linear > baselines; supervised > unsupervised) and the absence of leakage. `reports/metrics` reproduces exactly from the saved predictions (max difference 0).
- **Not trustworthy as stated:** (a) the Jeonse winner `hybrid_hgb` (its edge over `hgb_detrended` disappears with a month-block bootstrap and reverses on a 2024 pseudo-test); (b) the narrative of a "2025 level shift" (the underprediction is persistent and mostly calibratable: -8.33% in a 2024 pseudo-test vs -8.57% in 2025 for Jeonse); (c) the "price of extrapolation" of 1.6 / 1.0 WAPE points, which is an artifact of the extrapolating models omitting neighbourhood and a median-aligned loss.
- **Room for improvement is real:** exploratory trend-aware models with neighbourhood and median loss reach 14.15% (Jeonse) and 41.16% (Wolse) on TEST, against 17.44% and 45.33% for the chosen models (indicative; TEST looked at three times; a TRAIN-only 2024 pseudo-test points the same way).
- **Single most important fix:** redo the selection with the extrapolating models using `legal_dong` and a median-aligned loss, choosing among them on rolling-origin pseudo-tests with a block bootstrap, and rewrite the "level shift" narrative accordingly (branch `fix/selection-and-loss`, section 6).

## 2. Findings

Severity: Critical / High / Medium / Low. Areas: A validity and leakage, B metrics and conclusions, C modeling and statistics, D code and reproducibility, E portfolio. No Critical finding: I found no leakage and no invalid ranking.

| ID | Sev | Area | Location | Evidence | Recommended change | Effort |
|---|---|---|---|---|---|---|
| F-01 | High | A/B | `notebooks/09` cell 13 (`one_se_selection`, `src/rent_model/compare.py:123`); `metrics.py:125` | Row-level bootstrap gives `hybrid_hgb` vs `hgb_detrended` +0.2034 pts, SE 0.0590, CI 0.0835 to 0.3205. Month-block bootstrap (12 blocks): diff -0.203, SE 0.218, CI -0.609 to +0.235; district-block: SE 0.126, CI -0.479 to +0.020. Pseudo-test 2024 (fit on 2022-2023): `hgb_detrended` 17.16% beats `hybrid_hgb` 17.82%. If the two are declared tied, the tie rule selects `hgb_detrended` (complexity 13 vs 14, abs aggregate bias 3.39 vs 5.03) | Resample months (and districts) instead of rows; confirm with rolling-origin pseudo-tests before declaring a winner | M |
| F-02 | High | C | `README.md:157` (Limitations), `notebooks/04` md 33, `notebooks/08` md, `notebooks/09` summary | "A 2025 level shift no valid model anticipated". Evidence against: `hgb_legal_dong` aggregate bias -8.33% (Jeonse) and -15.74% (Wolse) on a 2024 pseudo-test vs -8.57% and -21.67% on 2025; rescaling predictions month by month lowers WAPE only 17.44 to 16.94 (`hybrid_hgb`, 2.9%) and 45.33 to 42.82 (`hgb_detrended`, 5.5%); the oracle index still leaves a -20.95% aggregate bias in Wolse (`notebooks/08`) | Reword as "persistent underprediction"; add a TRAIN-backtest recalibration and report it | S |
| F-03 | High | C/B | `src/rent_model/boosting.py:119` (`loss='squared_error'` default), all supervised notebooks | WAPE is minimized by a median, but models minimize squared error of `log1p`. In Wolse the log target is left-skewed (skew -0.401); in 92% of district x type x area groups (n>=30) mean(log1p y) < median(log1p y), median gap -0.132. Median loss (`quantile=0.5`) on TEST: `hgb_legal_dong` Wolse 44.33 to 41.31 (-3.02, CI -3.17 to -2.87), aggregate bias -21.67% to -9.57%; pseudo-test 2024: 43.32 to 41.73. No notebook discusses the retransformation bias of `expm1(E[log1p y])` | Make median loss an option and compare it systematically; document the retransformation | S |
| F-04 | High | C | `src/rent_model/compare.py:61` (`extrapolation_kind`), `notebooks/09` cells 13 and 20 | Extrapolation ability is assigned by model *label*. The Jeonse winner uses the `naive` index forecast, so its base projection is constant (69,726.1 for an Apartment of 60 m2 in Gangnam-gu in 2026, 2027 and 2028, cell 20). `hgb_time` (17.31%) is excluded as "cannot extrapolate" yet is behaviourally the same as the naive-index hybrid (17.44%) | Define extrapolation behaviourally (does the projected path move beyond the last level?) and state that Jeonse is "level persistence + band" | S |
| F-05 | High | C | `src/rent_model/hybrid.py:68`, `notebooks/09` md 15 ("cost of requiring extrapolation") | The extrapolating models use static features without `legal_dong`; `legal_dong` is worth 2.75 pts (Jeonse 18.59 to 15.84) and 1.21 pts (Wolse 45.54 to 44.33) for static models. Exploratory: `hybrid_hgb` + grouped `legal_dong` + median loss reaches 14.15% (Jeonse, vs 17.44%) and `hgb_detrended` + `legal_dong` + median loss 41.16% (Wolse, vs 45.33%): the "cost" of extrapolation (1.6 / 1.0 pts) is a design artifact | Add neighbourhood (TRAIN-grouped) and median loss to the trend-aware models, then redo selection | M |
| F-06 | Medium | A/B | `notebooks/09` cell 11; `notebooks/04` cell 12 (`best_id` by TEST WAPE) | 31 candidates per track are ranked and the winner is reported on the same 2025 TEST rows (winner's curse). It follows CLAUDE.md rule 8 literally, so it is a protocol weakness, not a violation | Select on 2024 pseudo-test / rolling-origin, then report 2025 once | M |
| F-07 | Medium | B | `notebooks/01` md 20; `notebooks/02` md 26 | md 20 says aggregate bias "from -14.58% to -31.84%"; the executed output (cell 19) has -30.63% (global median) as the worst; -31.84% is a stale number from the pre-fix run (the area baseline bug). md 26 says Detached highest "33.32%"; output 33.3251 (33.33%) | Correct both sentences; add a markdown-number check | S |
| F-08 | Medium | A | `src/rent_model/data.py` (no unit key); TEST vs TRAIN | With a strict unit key (district, dong, lot, type, area, floor, year built): 34.3% (Jeonse) and 32.0% (Wolse) of TEST rows are a unit already in TRAIN. WAPE matched vs unmatched for `hgb_legal_dong`: 14.55% vs 16.53% (Jeonse), 40.39% vs 45.99% (Wolse). Carrying forward the last price of the same unit gives 12.92% (Jeonse) and 34.96% (Wolse) on matched rows. Not temporal leakage, but TEST mixes known and new buildings | Report TEST error for repeat units and new units separately | S |
| F-09 | Medium | C | `notebooks/04` cell 10; `reports/metrics/*hgb_legal_dong.json` meta | The only contract-level interval belongs to `hgb_legal_dong` (not the chosen model) and covers 68.89% (Jeonse) and 64.93% (Wolse) vs 80% nominal; the chosen models have no interval. Scenario bands are index-only (stated) | Calibrated (conformal) intervals for the chosen model, with coverage on TEST | M |
| F-10 | Medium | C | `notebooks/08` cell 3 (`last_run(track, 'hgb_static')`), `notebooks/05` and `06` | `hybrid_hgb` reuses the hyperparameters of `hgb_static`; unsupervised-assisted models reuse `hgb_time`/`ridge_time`; tuning ran on 60,000 rows; `max_leaf_nodes=127` (grid edge) was chosen in 6 of 8 searches (`notebooks/04` md). The Jeonse winner got no tuning of its own | Tune each finalist on its own target; extend grids at the edge | M |
| F-11 | Medium | C | `notebooks/09` leaderboard cells | WAPE weights large contracts: the top decile of actual rents holds 34.0% of the Wolse total and 38.0% of the absolute error. Cheap contracts are poorly predicted: Wolse MdAPE 35.77% (`hgb_detrended`), median MAPE by district x quarter 77.07%; by predicted tercile MdAPE 37.92% (low), 29.51% (mid), 39.99% (high) | Report MdAPE and MAE in 10,000 KRW by price tier next to WAPE | S |
| F-12 | Medium | E | `README.md` (first screen) | Key results list models, not the answer to "how much will a tenant pay"; the scenario table is the last block of Results and its profile assumptions (median floor and age of the group, contract type `New`) appear only in `notebooks/09` cell 19 | Add a 3-line headline answer with one example and the assumptions | S |
| F-13 | Medium | D | `requirements.txt` | Only lower bounds (`pandas>=3.0`, `scikit-learn>=1.5`...) with no lock or caps; installed: pandas 3.0.3, numpy 2.5.1, scikit-learn 1.9.1, statsmodels 0.15.0. CI installs the latest versions. CI workflow never ran (not verified) | Add `constraints.txt` or `pip freeze` output and upper caps | S |
| F-14 | Medium | D | `notebooks/03` [9], `04` [9], `05` [3], `06` [3], `08` [3], `09` [27] (`last_run`); `03` [15] / `04` [17] (`bias_table`); `03` [21] / `04` [23] (`partial_dependence`); `07` cell 1 (`TRAIN_LAST = pd.Timestamp('2024-12-01')`) | Copy-pasted helpers in six notebooks; notebook 09 step 4 duplicates `src/rent_model/projection.py` (I checked they agree: 69,726.1 and 2,291.5); a hard-coded split date instead of `config.TRAIN_END` | Move helpers to `src`; read the date from config (CLAUDE.md rule 9 forbids style refactors, but this is divergence risk) | S/M |
| F-15 | Medium | C | whole project | No learning curve, no sensitivity to the training window, no residuals over time or calibration plot for the chosen models (`notebooks/09` has error by quarter only). Exploratory: Jeonse trained on 12 months 17.46% vs 24 months 16.10% (2024 pseudo-test); Wolse 43.74% vs 43.32% | Add a window/learning-curve section | M |
| F-16 | Medium | C | `README.md` Limitations; `notebooks/09` | Renewals are 39.8% (Jeonse) and 21.1% (Wolse) of TEST. With `renewal_right_used=Yes` 98.7% of Jeonse new/previous price ratios are <= 1.055 (the 5% rent cap); previous price as a feature cuts renewal-row WAPE 15.47% to 10.01% (Jeonse) and 46.88% to 22.22% (Wolse) in a TRAIN-only 2024 pseudo-test. Not exploited or discussed as policy information | Discuss in Limitations; decide on a labelled renewal-only model (section 4) | S (text) / M (model) |
| F-17 | Low | D | `src/rent_model/forest.py:8`, `hybrid.py:16`, `tests/test_clustering.py:2,6`, `test_forest.py:7`, `test_hybrid.py:9,10`, `test_projection.py:1`, `test_unsupervised.py:2` | pyflakes: unused imports (`clone`, `MONTH_COLUMN`, `pd`, `ClusterLabeler`, `DetrendedForest`, `temporal_split`, `log_index`, `np`) | Add pyflakes to CI | S |
| F-18 | Low | D | `tests/` | 84 test functions (118 collected). `tests/test_data.py` has 2 tests (counts and ranges). No test ties README or markdown numbers to `leaderboard.csv` (two stale numbers were found by hand), none checks that model notebooks only fit on TRAIN | Add a traceability test and a static check on the notebooks | M |
| F-19 | Low | D | `.gitignore`, `data/` | `.gitignore` lacks `.venv/`, `.idea/`, `.DS_Store`; the CSV is 60,620,429 bytes tracked in git (GitHub warns above 50 MB); data licence not stated in the README (not verified); `reports/predictions/` is ignored, so `notebooks/09` cannot run in a fresh clone before 01-08 (documented in the README) | Extend `.gitignore`; state the data licence; consider Git LFS | S |
| F-20 | Low | B | `notebooks/04` cell 20 | The permutation table lists `t` and `floor_missing` with 0.0000 for `hgb_legal_dong`, where `t` is not a feature (noise, partly explained in the markdown) | Print only the model's own features | S |
| F-21 | Low | A | `load_clean()` output | 24 exact duplicate rows remain (`df.duplicated().sum()` = 24): rows that differ only in the dropped lot/building columns | Mention or drop after the column drop | S |
| F-22 | Low | E | `reports/figures/09_leaderboard.png`, `09_projection_scenarios.png` | Titles state the method ("leaderboard (valid runs; dotted = invalid oracle diagnostics)", "EXTRAPOLATION SCENARIOS") instead of the message; units are present (10,000 KRW) | Message titles, e.g. "Boosting wins in both tracks; Wolse error stays above 44%" | S |

## 3. Critical and High findings in detail

**F-01 Selection is not robust to clustered uncertainty (High).**
- *What is wrong:* `paired_bootstrap_wape_diff` resamples 38,000 to 58,000 rows independently. Contracts of the same month, district or building are correlated and 2025 is a single year, so the standard errors (0.02 to 0.10 points) are tiny and the one-standard-error rule almost never produces a tie. With month blocks the SE of the key Jeonse comparison is 0.218 instead of 0.059 and the interval includes zero.
- *Why it matters:* the chosen Jeonse model rests on a 0.2-point difference. The 2024 pseudo-test reverses it (17.16% vs 17.82%), and the tie rule would then pick the simpler, less biased `hgb_detrended`. The README sentence "the ranking is stable" (99.9% of resamples) is only true under row resampling. The Wolse choice (`hgb_detrended` over `hybrid_hgb`) is robust: month-block CI -0.910 to -0.703.
- *Fix:* block bootstrap by month (and by district), plus a rolling-origin pseudo-test (train < 2023 and < 2024) as a second criterion.
- *Verify:* the tie set for Jeonse contains both detrended and hybrid; rerun `tests/test_compare.py` with a block option.

**F-02 The "2025 level shift" is a persistent, calibratable underprediction (High).**
- *What is wrong:* the models underpredict every year by a similar amount and the bias is mostly a constant scale, not a time-varying level. Evidence: pseudo-test 2024 vs TEST 2025 aggregate bias of `hgb_legal_dong` -8.33% vs -8.57% (Jeonse) and -15.74% vs -21.67% (Wolse); month-by-month rescaling reduces WAPE by only 0.50 (Jeonse chosen) and 2.51 points (Wolse chosen); the global scale that minimizes WAPE is 1.042 / 1.229 for the chosen models and 1.09 / 1.23 for the leader; the oracle index still leaves -20.95% aggregate bias in Wolse.
- *Why it matters:* the README and the summary blame an unforecastable 2025 shift, which suggests nothing can be done. A scale factor estimated only from a TRAIN backtest lowers TEST WAPE of `hgb_legal_dong` from 15.84% to 14.08% (Jeonse) and 44.33% to 41.87% (Wolse) (exploratory).
- *Fix:* rewrite the narrative; add recalibration from the last CV fold and report it as a model component.
- *Verify:* bias by year in a rolling-origin table; WAPE after recalibration on a pseudo-test not used to fit the factor.

**F-03 The training loss does not match the headline metric (High).**
- *What is wrong:* squared error on `log1p` estimates a geometric mean. For Wolse the log target is left-skewed (many small rents: 6.20% of TRAIN rents are 10 or less, in units of 10,000 KRW), so the geometric mean sits about 12% below the group median. `expm1` is applied once everywhere (checked in `forest.to_original`, `linear._to_original`, `hybrid.combine`), but the retransformation bias is never discussed.
- *Why it matters:* the Wolse aggregate bias of -21.67% and its WAPE are largely produced by the loss.
- *Fix:* median loss (`HistGradientBoostingRegressor(loss='quantile', quantile=0.5)`; already supported by `BoostedModel(quantile=0.5)`) as a compared variant for every boosting model.
- *Verify:* WAPE and bias on the rolling-origin pseudo-tests, then on 2025 once.

**F-04 "Can extrapolate" is a label, and the Jeonse winner is a flat line (High).**
- *What is wrong:* `extrapolation_kind` maps names to labels. The Jeonse winner extrapolates the *last level* (naive index forecast): the 2026-2028 base is constant and only the band moves. Tree models with a time feature are excluded although they behave the same way beyond TRAIN.
- *Why it matters:* the selection constraint decides the winner and the projection; a flat Jeonse base can be mistaken for a forecast of stable prices.
- *Fix:* define the criterion by behaviour and label Jeonse honestly as level persistence.
- *Verify:* a test that the projected path of each "extrapolating" model is not constant.

**F-05 The extrapolating models are handicapped by design (High).**
- *What is wrong:* `hybrid_hgb` and `hgb_detrended` use static features without `legal_dong`, and the squared loss.
- *Why it matters:* `notebooks/09` reports a "cost of requiring extrapolation" of 1.6094 (Jeonse) and 0.9958 (Wolse) points. With neighbourhood and median loss the trend-aware models beat the unconstrained leader (exploratory TEST: `hybrid_hgb` + `legal_dong` + median loss 14.15%; `hgb_detrended` + `legal_dong` + median loss 41.16%, against 15.84% and 44.33%).
- *Fix:* add a `legal_dong` option (TRAIN-grouped, 254 levels maximum) to the trend-aware models and rerun selection and projection.
- *Verify:* a rolling-origin table with the new models against the current ones.

## 4. Model improvement potential

### 4.1 Error decomposition on TEST 2025 (diagnostic: some columns use TEST actuals to measure, never to fit)

| Quantity | Jeonse | Wolse |
|---|---|---|
| Unconstrained leader `hgb_legal_dong`, WAPE % | 15.84 | 44.33 |
| Chosen model, WAPE % | 17.44 (`hybrid_hgb`) | 45.33 (`hgb_detrended`) |
| (a) Level: chosen model after rescaling each month so that aggregate bias is 0 | 16.94 (-0.50, 2.9% of the error) | 42.82 (-2.51, 5.5%) |
| Same with one global factor (WAPE-optimal factor) | 17.08 (factor 1.042) | 42.69 (factor 1.229) |
| Leader after one global factor | 14.07 (factor 1.090) | 41.65 (factor 1.226) |
| Rescaling by district x month / building type x month (chosen) | 16.79 / 16.88 | 42.52 / 42.38 |
| Index-forecast loss (`hybrid_hgb` minus invalid oracle `hybrid_hgb_oracle`) | 0.20 (17.44 vs 17.25) | 1.26 (46.14 vs 44.88) |
| (b) Structure: oracle hybrid WAPE (index known) | 17.25 | 44.88 |
| (c) Noise floor, coarse groups (district, type, 10% area bin, floor band, month), leave-one-out median, TRAIN | 24.25 (39.6% of rows covered) | 43.44 (36.1%) |
| (c) Noise floor with `legal_dong` and age band (small groups, n >= 5) | 13.89 (4.7% covered) | 34.02 (5.3%) |
| Share of the total absolute error from the top decile of actual values | 32.3% | 38.0% |

Reading: the time-varying level explains little (3% to 6% of the error); most of the gap is structure and a constant scale bias, plus noise. The crude floor is not a hard bound (groups of 5 or more near-identical contracts are the most standard ones and not representative of all rows, and the leave-one-out median itself is noisy). My estimate of the reachable region: **Jeonse roughly 14% to 15%** (the floor with neighbourhood and age is 13.89% to 14.82% on the covered subset), **Wolse roughly 40% to 42%**; the Wolse floor of 34% to 35% looks out of reach with the available features.

### 4.2 Ranked ideas

| # | Idea | Expected WAPE gain | Effort | Leakage risk | Can extrapolate | Evidence |
|---|---|---|---|---|---|---|
| 1 | Neighbourhood (`legal_dong`, TRAIN-grouped) in the trend-aware models | Large in Jeonse (about -2.8 to -3.3), medium in Wolse | S/M | Low (grouping from TRAIN only) | Yes | exploratory, section 4.3 |
| 2 | Median-aligned loss (`quantile=0.5` on `log1p`) | Large in Wolse (-2.9 to -3.0), small in Jeonse (-0.1 to -0.4) | S | Low | Yes | exploratory |
| 3 | Scale recalibration from a TRAIN backtest | Medium: leader Jeonse -1.76; Wolse -1.7 to -2.9 for squared-loss models; -0.2 to -0.3 for the chosen Jeonse models | S | Low if the factor comes from a pseudo-test | Yes (level adjustment, to be re-estimated yearly) | exploratory |
| 4 | Separate renewal-only model with the previous price | Large on renewals: 15.47% to 10.01% (Jeonse) and 46.88% to 22.22% (Wolse); overall pseudo-test 16.10% to 14.39% and 43.32% to 38.43% | M | Medium: forbidden by protocol, available only for renewals, `previous_*` has gross errors (max 2.2e9 vs max target 7e5) | Not for market projections | TRAIN-only pseudo-test |
| 5 | Average of `hybrid_hgb` and `hgb_detrended` | Small in Jeonse (17.03% vs 17.44%), none in Wolse (45.71% vs 45.33%) | S | Low | Yes | exploratory |
| 6 | More history (learning curve) | Medium in Jeonse (12 vs 24 months: 17.46% to 16.10%), little in Wolse (43.74% to 43.32%) | L (data) | Low | n/a | pseudo-test |
| 7 | Better index forecast (damped trend, ensembles) | Small: the whole index-forecast loss is 0.20 / 1.26 points | M | Low | Yes | `notebooks/08` |
| 8 | District or district x type indices | Not worth it: upper bound of 0.15 to 0.37 points (district x month rescaling 13.77 vs month 13.93; 41.38 vs 41.75) | M | Low | Yes | TEST-oracle bound |
| 9 | Unit price history (same lot, type, area, floor) | Marginal overall: 16.10% to 15.88% and 43.32% to 42.62% in the pseudo-test | M | Medium | n/a | pseudo-test |
| 10 | Hierarchical / mixed-effects shrinkage for `legal_dong`, interactions, per-type models | Unknown, probably small after #1 (not tested) | M | Low | Yes | not verified |
| 11 | Macro and policy information (base rate, rental-law dates, sale price indices) | Unknown for the level; the 5% renewal cap is already visible in the data (98.7% of ratios <= 1.055 when the renewal right is used) | M | Low if lagged | Only with forecasts | not verified |

Renewal-only model, honest assessment: for a *future renewal* the previous price is known to the parties, so it is available at prediction time, but it is not available for new contracts (74.4% of Wolse and 55.1% of Jeonse TEST rows) nor for market-level scenarios. It is a different product ("what will my renewal cost"). The `previous_*` columns need cleaning (values up to 2,200,000,000 against a maximum target of 700,000; 415 Jeonse rows with previous above 10 times the target). Recommendation: worth a separate, clearly labelled model; do not add to the main models; the decision belongs to the user.

### 4.3 Exploratory checks (indicative, not saved as model runs)

TEST was looked at **three times** (experiments A, B and C below) for about a dozen variants, plus the diagnostic decomposition of the saved predictions, so these TEST numbers are optimistic. Settings were not tuned on TEST: hyperparameters are those saved by the notebooks, and every scale factor comes from a pseudo-test 2024 (fit on 2022-2023). Caveat: the saved hyperparameters were tuned with CV folds that include 2024, so the pseudo-test is not fully independent of them. Intervals are the paired row bootstrap of `src/rent_model/metrics.py` (it understates uncertainty, see F-01).

| Experiment | Jeonse: pseudo-test 2024 / TEST 2025 (reference) | Wolse: pseudo-test 2024 / TEST 2025 (reference) |
|---|---|---|
| `hgb_legal_dong` median loss (A) | 16.01 vs 16.10 / 15.54 (15.84) | 41.73 vs 43.32 / 41.31 (44.33), diff -3.02 [-3.17, -2.87] |
| `hgb_legal_dong` x factor from 2024 (A) | - / 14.08 (15.84), diff -1.76 [-1.85, -1.67] | - / 41.87 (44.33), diff -2.46 [-2.58, -2.36] |
| `hgb_detrended` median loss (B) | 17.27 vs 17.16 / 17.70 (17.65) | 42.79 vs 43.76 / 42.48 (45.33), diff -2.85 |
| `hybrid_hgb` median loss x factor from 2024 (B) | 17.56 vs 17.82 / 17.01 (17.44), diff -0.43 [-0.53, -0.33] | 42.93 vs 44.75 / 42.37 (46.14), diff -3.77 [-3.91, -3.63] |
| `hgb_detrended` + `legal_dong` + median loss (C) | 14.32 vs 17.16 / 14.85 (17.65), diff -2.80 [-2.95, -2.64] | 41.53 vs 43.76 / 41.16 (45.33), diff -4.17 [-4.37, -3.97] |
| `hybrid_hgb` + `legal_dong` + median loss (C) | 14.93 vs 17.82 / 14.15 (17.44), diff -3.29 [-3.44, -3.13] | 41.74 vs 44.75 / 41.85 (46.14), diff -4.28 [-4.48, -4.10] |

The direction and most of the magnitude are confirmed by the TRAIN-only pseudo-test (for example Jeonse 17.82 to 14.93 and Wolse 44.75 to 41.74 for the hybrid), which does not use 2025. **Confirmation protocol before any claim enters the README:** rolling-origin pseudo-tests on 2023 and 2024 with settings chosen by the 3-fold CV on TRAIN, month-block bootstrap, and 2025 looked at once at the end.

### 4.4 Verdict per track

- **Jeonse: worth pursuing.** Expected gain larger than any bootstrap noise: chosen 17.44% to about 14% to 15% with neighbourhood and median loss; recalibration is a cheap extra. Best next experiment: `hybrid_hgb` / `hgb_detrended` with grouped `legal_dong` and median loss, tuned on `y_adj`, selected on the 2023 and 2024 pseudo-tests.
- **Wolse: worth pursuing, with a lower ceiling.** Chosen 45.33% to about 41% to 42.5%, mostly from the loss (bias -22.48% to about -10%). The remaining structure and noise (floor about 34% to 35%) is not reachable with these features. Best next experiment: median loss in `hgb_detrended` with `legal_dong`, then check the bias across the 2023 and 2024 pseudo-tests.
- **Renewal-only model:** worth a separate labelled study in both tracks (large gain on 18% to 40% of the rows), user decision.

## 5. What is already good

- Strict temporal split in every model notebook (all use `temporal_split`; no random split anywhere); scaling, encoding, clustering, tiers, mixtures and the index vintages are fit on TRAIN rows (checked in code and by cell).
- Feature lists checked at fit time for 12 model configurations: none contains a forbidden column (the full matrix passed to models has 13 columns, all allowed by the protocol).
- Oracle runs are flagged `valid=False`, excluded from rankings and shown separately (`tests/test_compare.py`, leaderboard CSV).
- Exact reproducibility: WAPE, MAE, median and aggregate bias recomputed from `reports/predictions` match `reports/metrics` with maximum difference 0.00e+00 for six runs; my independent re-derivation of the leader and tie sets (different bootstrap seed) agrees with the notebook and README.
- Index vintages: `rolling_origin_forecasts` asserts that no month after the origin enters a fit.
- Honest labelling: "extrapolation scenarios, not forecasts" and the 48-month limit appear in notebook 09, the README and the app; index-only bands are stated.
- Cleaning is documented with exact row counts, outliers are kept and only impossible values are removed.
- `pytest -q`: 118 passed (42.54 s); README numbers are generated from `leaderboard.csv` and notebook outputs and pass a numeric audit; notebook 00 runs without errors in a fresh kernel (temporary copy).

## 6. Suggested fix plan (nothing implemented)

1. **`fix/readme-and-notebook-numbers`**: correct the stale numbers (F-07), reword the level-shift narrative in the README and notebooks 04, 08 and 09 (F-02), add the numeric traceability test (F-18). Commit: `fix: correct stale markdown numbers and level-shift wording`.
2. **`fix/selection-and-loss`**: month/district block bootstrap and pseudo-test tie-break in `compare.py` (F-01, F-06), behavioural extrapolation criterion (F-04), median-loss option documented with the retransformation note (F-03). Commit: `fix: robust model selection and median-aligned loss`.
3. **`improve/model-error`**: `legal_dong` in the trend-aware models (F-05), tuning of each finalist (F-10), TRAIN-backtest recalibration, learning curve and window sensitivity (F-15), error by repeat vs new units (F-08), MdAPE and MAE by price tier (F-11). Commit: `feat: neighbourhood-aware trend models with median loss and recalibration`.
4. **`improve/uncertainty`**: conformal intervals for the chosen model with TEST coverage (F-09); scenario assumptions table. Commit: `feat: calibrated prediction intervals for the chosen model`.
5. **`improve/renewal-model`** (only if the user decides): a labelled renewal-only study with cleaned `previous_*` and the 5% cap discussion (F-16). Commit: `feat: renewal-only model study`.
6. **`polish/portfolio`**: README headline answer and assumptions (F-12), message chart titles (F-22), a model card, pinned environment (F-13), shared helpers in `src` and config-driven dates (F-14), unused imports and pyflakes in CI (F-17), `.gitignore` and data licence (F-19). Commit: `docs: README headline answer, model card and environment pins`.

## 7. Not verified

- The GitHub Actions workflow (`.github/workflows/ci.yml`) was never executed; the `Makefile` was not run (`make` is not installed here).
- Only notebook 00 was re-executed in a fresh kernel (in a temporary copy). Notebooks 01-09 were not re-run in this review; their outputs are those stored in the repository.
- True unit identity: lot numbers are missing for 23.1% of rows and are absent after `load_clean`; the strict unit key is an approximation.
- Availability of the previous price at prediction time is a domain assumption (a renewal is known to the parties).
- Data licence and terms of redistribution of the Seoul Open Data file.
- Gain from macroeconomic variables and policy dates: no external data were available to test it.
- The Streamlit app was tested headlessly only; its visual layout was not reviewed.
- Duan smearing was not computed; the retransformation bias is documented through group-level skew and the median-loss experiment instead.
- Platform and version sensitivity: only Python 3.13.7 on Windows with the installed versions listed in F-13.
