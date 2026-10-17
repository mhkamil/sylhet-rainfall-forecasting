# NASA POWER Data Source
- **Provider:** NASA Prediction Of Worldwide Energy Resources (POWER), https://power.larc.nasa.gov/
- **Download Date:** 2026-10-14
- **Community:** Agroclimatology (AG)
- **Product line (header):** "NASA/POWER Source Native Resolution Daily Data"
- **Temporal Level:** Daily, 1981-01-01 to 2025-12-31. Days are defined in **UTC** (local Sylhet time is UTC+6, so one POWER day = 06:00–06:00 local).
- **Coordinates:** Latitude 24.9157, Longitude 91.832
- **Elevation:** 103.47 m (MERRA-2 average for the 0.5° × 0.625° lat/lon grid cell; not a station elevation)
- **Missing-value code:** `-999` ("missing source data that cannot be computed or is outside of the sources availability range")
- **Raw File:** `data/raw/sylhet_power_daily_AG_1981-2025_raw.csv` (21-line metadata header, then a CSV table with columns `YEAR, DOY, <parameters>`)
- **Raw file SHA-256:** `396f799b5cf6a551aae8de603e9f23cdbb7cc9fdbdbf38face8885169cdba7d8` (the raw file is read-only; never edit it)

## Parameters (as given in the file header)

| Column | Source (header) | Description | Units | Availability in this file |
|---|---|---|---|---|
| QV2M | MERRA-2 | Specific humidity at 2 m | g/kg | 1981–2025, complete |
| RH2M | MERRA-2 | Relative humidity at 2 m | % | 1981–2025, complete |
| IMERG_PRECTOT | header says "MERRA-2" ⚠️ | Total precipitation | mm/day | **1998-01-01 → 2025-12-31** (−999 before) |
| T2M | MERRA-2 | Air temperature at 2 m (daily mean) | °C | complete |
| T2M_MAX | MERRA-2 | Daily maximum temperature at 2 m | °C | complete |
| T2M_MIN | MERRA-2 | Daily minimum temperature at 2 m | °C | complete |
| T2MDEW | MERRA-2 | Dew/frost point at 2 m | °C | complete |
| PS | MERRA-2 | Surface pressure | kPa | complete |
| WS2M | MERRA-2 | Wind speed at 2 m | m/s | complete |
| WD2M | MERRA-2 | Wind direction at 2 m | degrees | **invalid from 2012-12-31 onward** ⚠️ |
| WS10M | MERRA-2 | Wind speed at 10 m | m/s | complete |
| ALLSKY_SFC_SW_DWN | CERES SYN1deg | All-sky surface shortwave downward irradiance | MJ/m²/day | **1984-01-01 → 2025-12-31** (−999 before) |
| PRECTOTCORR | MERRA-2 | Precipitation, bias-corrected | mm/day | complete, but **inhomogeneous** ⚠️ |

## Known issues found in Stage 0 (`notebooks/00_intake.ipynb`)
1. **IMERG_PRECTOT header label.** The header describes it as "MERRA-2 Total Precipitation", but the column name and data behaviour (starts 1998, daily r = 0.62 with PRECTOTCORR, different annual totals) point to GPM IMERG satellite precipitation. Treated as a header labelling error; **verify against the POWER parameter dictionary before citing.**
2. **WD2M invalid after 2012-12-30.** Every value from 2012-12-31 to 2025-12-31 (4,749 days) is between about −9,000 and −5,400. These are not `-999` codes and are kept unchanged in `power_parsed.parquet`.
3. **PRECTOTCORR inhomogeneity.** Mean annual totals of about 1,925 mm (1981–1997), 2,970 mm (1998–2015) and 5,380 mm (2016–2025); IMERG is stable (3,680 vs 3,757 mm for 1998–2015 vs 2016–2025).
4. **Radiation source.** The header names CERES SYN1deg, but CERES SYN1deg begins in 2000. POWER usually fills earlier years from GEWEX SRB, so a source change around 2000 is possible; to check in EDA.

## Processed copy
`data/processed/power_parsed.parquet`: same rows and columns as the raw table, a daily `DatetimeIndex` named `date` (from YEAR+DOY), and `-999 → NaN`. Nothing else is changed.

## How to cite these data (Stage 9)
NASA Langley Research Center (LaRC) Prediction Of Worldwide Energy Resources (POWER) Project, Agroclimatology (AG) daily point data, 24.9157° N, 91.832° E, 1981-01-01 to 2025-12-31, downloaded 2026-10-14 from https://power.larc.nasa.gov/ (the researcher's notes give an access date of 2026-10-16; this file records the download of the raw file on 2026-10-14, which the manuscript uses). Cite together with Sparks (2018), Gelaro et al. (2017), Reichle et al. (2017) and Huffman et al. (2020); see `reports/references.bib`.

The file header literally labels `IMERG_PRECTOT` as "MERRA-2 Total Precipitation (mm/day)" (header line 10), and `PRECTOTCORR` as "MERRA-2 Precipitation Corrected (mm/day)" (line 20).
