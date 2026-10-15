# Decision log

| # | Date | Stage | Decision | Reason |
|---|---|---|---|---|
| 0.1 | 2026-10-15 | 0 | Raw file is read-only; record its SHA-256 and re-check it at the end of every intake run. | Proves the raw data were never modified (reproducibility). |
| 0.2 | 2026-10-15 | 0 | Detect the metadata header end (`-END HEADER-`) in code rather than hard-coding `skiprows`. | Robust if POWER changes the header length. |
| 0.3 | 2026-10-15 | 0 | Build the date index from YEAR+DOY with `format="%Y%j", errors="raise"`. | Strict parsing rejects an invalid DOY instead of rolling it into the next year. |
| 0.4 | 2026-10-15 | 0 | Keep dates in UTC (no conversion to local time). | This is how POWER defines a day; it must be stated in the paper and respected for rule 5. |
| 0.5 | 2026-10-15 | 0 | Only transformation in the parsed copy is `-999 → NaN`; save as Parquet (`data/processed/power_parsed.parquet`). | Rule 12; Parquet keeps dtypes and the DatetimeIndex exactly. |
| 0.6 | 2026-10-15 | 0 | No imputation in Stage 0. | Missingness is structural (IMERG before 1998, radiation before 1984), so it is a study-design choice, not a cleaning step. |
| 0.7 | 2026-10-15 | 0 | WD2M invalid values (2012-12-31 onward) are kept unchanged in the parsed file and flagged. The decision on how to handle them is **pending** (drop / re-download U,V / not impute). | Rule 12: show the problem first; the test period would be 100 % affected. |
| 0.8 | 2026-10-15 | 0 | Flag PRECTOTCORR inhomogeneity; choice of target product and study period is **pending** (recommendation: IMERG_PRECTOT, 1998–2025). | Step changes in the target would confound the chronological evaluation. |
| 0.9 | 2026-10-15 | 0 | Environment for Stage 0: Python 3.13, pandas 3.0.5, numpy 2.5.3, matplotlib 3.11.2, pyarrow 26.0.0. | Reproducibility; pin in `requirements.txt`. |
