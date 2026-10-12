# Project assessment: Seoul rental market (Jeonse and Wolse), 2022-2025

**Overall score: 8.0 / 10**

The project is a rigorous, well-documented analysis whose strongest points are leakage control, an explicit and amended selection protocol, and honest reporting of uncertainty and failures. Its weaker points are reproducibility gaps around the generated README and the data provenance, a modest Wolse accuracy (best 2025 WAPE 40.80%) and a tuning budget that is still small. By the rubric below (5 = competent course project, 7 = good junior portfolio project, 9 = senior-level work) it sits between a good portfolio project and senior-level work, and the remaining gap is mostly engineering and provenance, not statistics.

Method note: each criterion is scored from 0 to 10 with the weights below; the total is the weighted sum (7.955 before rounding), recomputed by a script. The scores reflect the repository state at commit `5fd0599` (`git log`, 23 commits at that point) and the pytest run done for this assessment (198 passed in 123.94s (0:02:03)). The two documents added after that commit (this file and `docs/PROJECT_DOCUMENTATION.md`) are not part of what is scored. Scores are judgments supported by the evidence column; they are calibrated against the rubric, not against other projects.

## Scores by criterion

Table 1. Criterion, weight, score (0-10) and the evidence and main gap behind it.

| Criterion | Weight | Score | Evidence | Main gap |
|---|---:|---:|---|---|
| Problem framing and storytelling | 15% | 7.8 | README lines 5-9 state the answer first (Jeonse deposit and Wolse monthly rent, 2025 WAPE 14.14% and 40.80%) with a worked example (60 m2 apartment in Gangnam-gu, base deposit 77,221.8 in 10,000 KRW); a business-question section and a model card explain the audience and the limits. | The headline answer is one example profile; the Wolse error (WAPE 40.80%) is large for a "how much will I pay" message. The README cleaning table has a rendering defect (see below). The first screen is dense with four-decimal numbers. |
| Data understanding and cleaning | 10% | 8.0 | Column dictionary with units and caveats (Notebook.ipynb, introduction); every cleaning step counted (12,674 of 400,000 rows dropped, 3.17%; 387,326 rows kept); notebooks/00 counts the 24 remaining duplicate rows (0.0062%) and shows that none remains when the lot numbers are kept. | The script that built the sample (sampling of about 100,000 rows per year, romanization, `lease_type` inference) is not in the repository, so the CSV cannot be rebuilt from the source. The data terms of use are not stated (deliberately). `lease_type` is inferred in three of four years. |
| Exploratory analysis | 10% | 7.3 | Notebook.ipynb section 9 lists 13 findings with printed numbers; skewed variables are summarised with medians and Spearman correlations (0.647 overall, 0.749 within Jeonse, 0.547 within Wolse); pooled and per-contract-type comparisons are separated; the quality-adjusted index (notebooks/07) shows the raw monthly median rise (142.58 and 110.00 rebased) is largely sample mix (index 103.96 and 95.28). | The EDA notebook is long (162 cells) and several conclusions about foreigner-oriented areas rest on legal-dong approximations with few rows (216 for Sinchon Jeonse). No formal test supports the seasonality statement. |
| Validity and leakage control | 15% | 8.7 | Strict temporal split (TRAIN 2022-2024, TEST 2025); everything fit on the fit window only; forbidden features enforced by tests (tests/test_features.py, tests/test_renewal.py); selection on pseudo-tests with an assertion that no 2025 row selects (notebooks/10 and 11); the 2025 inspection is disclosed in README, CLAUDE.md and the model card; the renewal-only exception to rule 5 is isolated and tested. | 2025 was inspected in earlier iterations, so the candidate list and the saved settings of the earlier models carry that influence (disclosed). Earlier models were tuned with CV folds that include 2023 and 2024 (stated in notebooks/10). |
| Modeling and model comparison | 15% | 8.2 | 44 models x 2 tracks (88 runs, 84 valid): baselines, linear, forests, boosting, K-Means, hierarchical clustering, Gaussian mixture, time-series index forecast, hybrid, neighbourhood-aware, median loss, recalibration; one-standard-error selection with block bootstraps; behavioural extrapolation criterion; component effects measured (notebooks/11). | Each tuning samples 12 candidates and the best `max_leaf_nodes` is the largest grid value in most searches; the earlier boosting-based unsupervised-assisted models reuse the settings of `hgb_time` and `ridge_time`; the best Wolse model still has a WAPE of 40.80%. |
| Evaluation and uncertainty | 10% | 8.0 | WAPE is justified and complemented by MAE, MdAPE, median and aggregate bias, MAPE by district x quarter; error by building type, area, quarter, district, predicted-price tercile and repeat versus new units; conformal intervals with measured coverage (2025: 0.8186 and 0.9595 in Jeonse, 0.8095 and 0.9542 in Wolse) and a candid report of failures (Jeonse detached houses 0.5582; Wolse pseudo-test 2024 0.7504). | Group-conditional coverage is not achieved and the Wolse intervals are very wide (median relative width 1.6647 of the prediction at the 80% level). There is no calibration plot of the point predictions and no residual-over-time plot for the chosen models. |
| Code quality and reproducibility | 10% | 7.5 | Package `src/rent_model` (21 modules), 20 test files, pytest result run for this assessment: 198 passed in 123.94s (0:02:03); pyflakes clean; seed 42 defined once; `constraints.txt` pins the environment; CI defined (pyflakes and tests); the helper duplication of the review was removed; refactor check shows identical metrics (reports/refactor_check.csv). | The generator of the README is not in the repository, so the README cannot be regenerated by a command in the repo, and the test that ties README numbers to outputs did not catch the broken cleaning-table cell. `make install` does not use `constraints.txt`. The CI has no recorded run in the repository (not verifiable here). `reports/predictions/` is gitignored, so a fresh clone must run the notebooks in order; notebook 11 takes hours. |
| Documentation and presentation | 10% | 7.8 | README, `docs/MODEL_CARD.md`, `CLAUDE.md` (protocol), a review file, notebooks with an interpretation after each result, figure titles that state the message with units (`reports/figures/09_leaderboard.png`, `12_coverage_by_group.png`). | README defect at line 40 (the duplicates cell of the cleaning table prints a Python object instead of the number of duplicate rows dropped); a model-card sentence on low-rent contracts that the printed numbers do not support; mixed two- and four-decimal precision; the repository map lists notebooks 12 and 13 before 11 and 09. Before this assessment there was no long-form documentation for non-experts. |
| Fit for South Korean tech recruiters | 5% | 7.8 | Clear English; executive summary at the top of the README; end-to-end ownership (data cleaning, modeling, selection protocol, uncertainty, tests, CI, a Streamlit explorer in `app/streamlit_app.py`). | No hosted demo or screenshot of the app; Korean terms appear only in the dictionary of the EDA and this documentation; the README is long for a first read. |
| **Weighted total** | 100% | **8.0** | | |

## Defects found while reading the repository

1. `README.md` line 40 (the "Exact duplicates" row of the cleaning table) contains the text `<re.Match object; span=(0, 77), match=...>` instead of the number of duplicate rows dropped. The executed `Notebook.ipynb` prints "Duplicate rows dropped: 10". The cause is a variable-name collision in the script that generates the README (that script is not in the repository).
2. `docs/MODEL_CARD.md`, "Known failure modes", first bullet ("Cheap and low-rent Wolse contracts") says that relative errors are large for low rents. The printed median absolute percentage error of the Wolse winner by predicted-price tercile is 25.3310% in the lowest tercile and 32.4710% in the highest (notebooks/11, 2025 report), so that wording is not supported. What the outputs support is that percentage errors are heavy-tailed (median MAPE by district and quarter 86.8146% against an MdAPE of 27.3385%).

This assessment did not edit the README beyond the single link line that the task allows, and did not edit the model card; both defects are listed here so that they can be fixed. `docs/PROJECT_DOCUMENTATION.md` words the second point as the outputs support it.

## Status of the findings of the review

Table 2. Status of the 22 findings of `reports/REVIEW_MODELING.md` in the current repository (16 fixed, 6 partly fixed, 0 open). Evidence was checked by search or by running code at this commit.

| ID | Finding | Status | Evidence |
|---|---|---|---|
| F-01 | Row bootstrap understates uncertainty | fixed | `metrics.block_bootstrap_wape_diff` and `conservative_wape_diff`; `compare.one_se_selection` uses the largest of three standard errors; notebooks/10 block a. |
| F-02 | Wrong narrative "2025 level shift" | partly | Rewritten in README, notebooks 04, 08, 09, 10 using bias by year; the phrase "level shift" remains in the markdown of notebooks 00, 01, 02, 03 and 07 (checked by search now). |
| F-03 | Squared loss vs WAPE | fixed | Median-loss variants (`_median`), Duan smearing comparison in notebooks/10; median loss is part of both chosen models. |
| F-04 | Extrapolation judged by label | fixed | `compare.classify_projection` (behavioural); behaviour column in the leaderboard; CLAUDE.md rule 8. |
| F-05 | Trend-aware models handicapped | fixed | Neighbourhood-aware trend models and median loss in notebooks/11; the unconstrained and the extrapolating choice are the same model in each track. |
| F-06 | Winner reported on the selection rows | fixed | Selection on pseudo-tests 2023 and 2024, 2025 reported once (assertion in notebooks/10 and 11); disclosure of earlier inspection. |
| F-07 | Stale numbers in markdown | fixed | Commit "fix: correct stale markdown numbers and add numeric traceability test"; the two stale strings are absent now (search). |
| F-08 | Repeat versus new units | fixed | notebooks/11 block 7 with an approximate unit key; README states the approximation. |
| F-09 | No calibrated interval for the chosen model | fixed | notebooks/12 and `src/rent_model/conformal.py`; coverage reported, including failures. |
| F-10 | Hyperparameters reused and not tuned on own target | partly | The finalists are tuned on their own target inside each fit window with a wider grid (notebooks/11); the earlier models and the unsupervised-assisted models still reuse saved settings, and the best `max_leaf_nodes` is at the grid edge. |
| F-11 | WAPE hides cheap contracts | fixed | MdAPE and MAE by predicted-price tercile and building type (notebooks/11 block 8 and 2025 report). |
| F-12 | README does not answer the question first | fixed | Headline block at the top of the README (a new rendering defect in the cleaning table is reported separately). |
| F-13 | Environment not pinned | partly | `constraints.txt` and upper caps exist and the CI uses them; `make install` ignores the constraints and no CI run can be verified from the repository. |
| F-14 | Duplicated helpers | fixed | `last_run`, `bias_table`, `partial_dependence`, `comparable_contracts` moved to `src/rent_model`; `TRAIN_LAST` read from `config` (search shows no helper definitions left in the notebooks). |
| F-15 | No learning curve or window sensitivity | partly | Learning curve and window sensitivity in notebooks/11 block 6; no calibration plot or residual-over-time plot for the chosen models. |
| F-16 | Renewals and the 5% cap not discussed | fixed | Renewal-only study (notebooks/13), policy shares, README section; isolated exception in CLAUDE.md. |
| F-17 | Unused imports | fixed | pyflakes reports nothing on `src tests app`; pyflakes runs in CI. |
| F-18 | Weak tests and no traceability test | partly | `tests/test_traceability.py` ties README tables and key lines to the leaderboard and checks the temporal split in the notebooks; it did not detect the broken cleaning-table cell of the README. |
| F-19 | Hygiene, data size and licence | partly | `.gitignore` extended; the CSV (about 61 MB) is still tracked without Git LFS; the licence is intentionally not stated and the reader is told to check the source page. |
| F-20 | Permutation table with non-features | fixed | Notebook 04 permutes only the model's own features (cell 20). |
| F-21 | Duplicate rows remain after cleaning | fixed | Counted and explained in notebooks/00 and the README; the rows are kept by decision. |
| F-22 | Figure titles state the method | fixed | Figures of notebooks 09 and 12 carry message titles built from printed values. |

## What would raise the score

Table 3. The five changes with the highest estimated gain, with the criteria they affect and the estimated overall score if only that change were made. The criterion scores move by the amounts in the third column; the overall estimate is recomputed with the weights.

| # | Change | Criteria | Estimated criterion changes (points) | Estimated overall score if done alone |
|---:|---|---|---|---:|
| 1 | Add lagged macro and policy covariates (base rate, rental-law dates) and test them under the pseudo-test protocol | Modeling (5), Framing (1), Validity (4) | Modeling +0.5, Problem framing +0.3, Validity +0.2 | 8.1 |
| 2 | Version the README generator, fix the cleaning-table cell, and make the traceability test cover every README table row | Code (7), Documentation (8) | Code quality +0.8, Documentation +0.6 | 8.1 |
| 3 | Document the data provenance (a download and sampling script or a precise recipe), confirm the terms of use on the source page and state them, and make a fresh clone runnable (constraints in `make install`, a documented cached-predictions path, a verified CI run) | Data (2), Code (7) | Data understanding +0.8, Code quality +0.4 | 8.1 |
| 4 | Work on the Wolse track: error analysis of cheap contracts, a joint model of deposit and rent, and an alternative metric report | Modeling (5), Evaluation (6) | Modeling +0.6, Evaluation +0.3 | 8.1 |
| 5 | Segment-aware (Mondrian) conformal intervals by building type and price tier, plus a calibration plot and residuals over time | Evaluation (6) | Evaluation +0.7 | 8.0 |

Done together, the five changes are estimated to give 8.6 / 10. Each criterion has a small weight, so each single change moves the total by about one tenth of a point; the gains are estimates, not measurements.

## Strengths to keep

- Temporal discipline: a strict time split, fit-window-only fitting and rolling-origin pseudo-tests, with assertions that no 2025 row selects a model (`notebooks/10`, `notebooks/11`).
- An explicit, amended selection protocol in `CLAUDE.md` (block bootstrap, one-standard-error rule, behavioural extrapolation) that was applied, and a candid disclosure that 2025 was inspected earlier.
- Honest failure reporting: Jeonse detached houses are covered 0.5582 at the 80% level, the Wolse pseudo-test of 2024 covers 0.7504, and the Jeonse scenario mixes an 8.17% yearly trend with a flat index band.
- Separation of products: the renewal-only study uses the previous price as an isolated, tested exception and is excluded from the main leaderboard.
- Numeric traceability: markdown numbers come from executed outputs and a test ties README tables to `leaderboard.csv`.
- A quality-adjusted price index that shows the raw monthly median rise is mostly sample mix.
- A model card and a review file that document intended use, out-of-scope use and known failure modes.
- A shared package with a test suite, a pinned environment and a refactor check showing identical metrics after the helpers were moved.

## Honest ceiling

Three limits hold regardless of effort. The data cover 48 months, so trends and seasonality are barely identifiable and every 2026-2028 number is an extrapolation scenario. The CSV is a sample with about 100,000 records per year, not proportional to the market, so volumes cannot be inferred and the index only partly corrects composition. There are no macro variables (interest rates, policy, supply), which drive deposits. These limits cap the attainable accuracy, especially for the monthly rent, but they do not lower the score because the project states them, measures their effect where it can (error by segment, bias by year, coverage under shift) and words every projection accordingly. A project that hid them would score lower, not higher.
