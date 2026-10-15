# Decision log

| # | Date | Stage | Decision | Reason |
|---|---|---|---|---|
| 0.1 | 2026-10-15 | 0 | Raw file is read-only; record its SHA-256 and re-check it at the end of every intake run. | Proves the raw data were never modified (reproducibility). |
| 0.2 | 2026-10-15 | 0 | Detect the metadata header end (`-END HEADER-`) in code rather than hard-coding `skiprows`. | Robust if POWER changes the header length. |
| 0.3 | 2026-10-15 | 0 | Build the date index from YEAR+DOY with `format="%Y%j", errors="raise"`. | Strict parsing rejects an invalid DOY instead of rolling it into the next year. |
| 0.4 | 2026-10-15 | 0 | Keep dates in UTC (no conversion to local time). | This is how POWER defines a day; it must be stated in the paper and respected for rule 5. |
| 0.5 | 2026-10-15 | 0 | Only transformation in the parsed copy is `-999 → NaN`; save as Parquet (`data/processed/power_parsed.parquet`). | Rule 12; Parquet keeps dtypes and the DatetimeIndex exactly. |
| 0.6 | 2026-10-15 | 0 | No imputation in Stage 0. | Missingness is structural (IMERG before 1998, radiation before 1984), so it is a study-design choice, not a cleaning step. |
| 0.7 | 2026-10-15 | 0 | WD2M invalid values (2012-12-31 onward) are kept unchanged in the parsed file and flagged. Resolved in 1.3. | Rule 12: show the problem first; the test period would be 100 % affected. |
| 0.8 | 2026-10-15 | 0 | Flag PRECTOTCORR inhomogeneity; Resolved in 1.1–1.2. | Step changes in the target would confound the chronological evaluation. |
| 0.9 | 2026-10-15 | 0 | Environment for Stage 0: Python 3.13, pandas 3.0.5, numpy 2.5.3, matplotlib 3.11.2, pyarrow 26.0.0. | Reproducibility; pin in `requirements.txt`. |
| 1.1 | 2026-10-15 | 1 | **Primary target = PRECTOTCORR, 1981–2025.** Split: train 1981–2019 (validation folds within 2014–2019), **test 2020–2025 locked**. | Researcher's decision: follow the project protocol and use the full record; the regime shift is itself reported as a finding. |
| 1.2 | 2026-10-15 | 1 | **Secondary target = IMERG_PRECTOT, 1998–2025**, used for the Stage-1 comparison and Stage-6 robustness / window ablations. | IMERG is homogeneous where PRECTOTCORR is not; it provides an independent check of conclusions. |
| 1.3 | 2026-10-15 | 1 | Keep WD2M unchanged during the Stage-1 audit (flagged); **drop WD2M from the feature sets in Stage 2**. | Every value from 2012-12-31 onward is invalid, so the variable is unusable in validation and test; imputing 13 years would fabricate data. |
| 1.4 | 2026-10-15 | 1 | EDA that can inform modelling (distributions, climatology, spells, ACF/PACF, correlations) uses **1981–2019 only**; data-integrity checks (missingness, ranges, annual totals, homogeneity, last 12 months) use the full record. | Rule 4: avoid data snooping on the test period, while still detecting broken data everywhere. |
| 1.5 | 2026-10-15 | 1 | Seasons: pre-monsoon Mar–May, monsoon Jun–Sep, post-monsoon Oct–Nov, winter Dec–Feb (season of day t). | Standard Bangladesh convention; DJF/MAM/JJA/SON would split the monsoon. |
| 1.6 | 2026-10-15 | 1 | Spearman (not Pearson) for predictor–next-day-rain correlations, overall and by season. | Rainfall is extremely skewed (skew ≈ 4) and relations are monotone but non-linear. |
| 1.7 | 2026-10-15 | 1 | Mann–Kendall via `scipy.stats.kendalltau(year, total)`, Sen's slope via `scipy.stats.theilslopes`; Pettitt test implemented in-notebook (not available in scipy 1.18 / statsmodels 0.15). | Rule 13: no invented APIs; transparent short implementations. |
| 1.8 | 2026-10-15 | 1 | WD2M excluded from climatology and correlation analyses. | Circular variable (means/rank correlations meaningless without sin/cos encoding) and invalid after 2012. |
| 1.9 | 2026-10-15 | 1 | Findings recorded, **not corrected**: PRECTOTCORR change points ~2000 and 2016 (Pettitt p < 0.001), ratio to IMERG 0.82 → 1.44, rain-day share 50.5 % → 57.0 %; parallel steps in humidity inputs. | Rule 12; correcting a target would fabricate data. To be handled with robustness analyses (pending approval of the Stage-2 plan). |
| 1.10 | 2026-10-15 | 1 | Stage-2 cleaning plan proposed in `01_eda.ipynb` Section 9 — **pending approval**. | Rule: propose, then wait for the researcher. |
