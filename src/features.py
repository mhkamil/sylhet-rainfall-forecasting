"""Stage 2 feature and target construction for next-day rainfall forecasting.

Rule: every feature at row t uses only rows <= t (shift with k >= 0, trailing rolling windows).
Targets are the only columns that look forward (row t+1), and they are built separately.
"""
import numpy as np
import pandas as pd

RAIN = "PRECTOTCORR"
CORE = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M"]          # Feature Set A
EXTENDED = ["PS", "QV2M", "T2MDEW", "WS10M"]                                # added in Feature Set B
LAGS = [1, 2, 3, 5, 7]
WINDOWS = [3, 7, 14, 30]
WET_MM = 1.0                                                                # primary rain-day threshold
TENDENCY = {"A": ["T2M", "RH2M"], "B": ["T2M", "RH2M", "PS", "T2MDEW", "QV2M"]}
THRESHOLDS = {"y_cls": 1.0, "y_cls_01": 0.1, "y_cls_25": 2.5}


def run_length(flag):
    """Length of the current run of True values up to and including each row (0 where False)."""
    return flag.astype(int).groupby((~flag).cumsum()).cumsum()


def build_features(clean, fset):
    """Return (X, dictionary) for feature set 'A' or 'B'. Rows are NOT dropped here (warm-up rows contain NaN)."""
    vars_ = CORE + (EXTENDED if fset == "B" else [])
    base = clean[vars_].drop(columns=RAIN).assign(rain_log1p=np.log1p(clean[RAIN]))
    cols, rows = {}, []

    def add(name, values, definition, family):
        cols[name] = values
        rows.append((name, definition, family))

    for v in base:
        add(v, base[v], f"{v} on day t" + (" = log(1 + PRECTOTCORR mm)" if v == "rain_log1p" else ""), "current")
    add("wet_today", (clean[RAIN] >= WET_MM).astype(int), "1 if PRECTOTCORR on day t >= 1 mm", "current")
    for v in base:
        for k in LAGS:
            add(f"{v}_lag{k}", base[v].shift(k), f"{v} on day t-{k}", "lag")
    for v in base.columns.drop("rain_log1p"):
        for w in WINDOWS:
            add(f"{v}_mean{w}", base[v].rolling(w).mean(), f"mean of {v} over days t-{w - 1}..t", "rolling mean")
    for w in WINDOWS:
        add(f"rain_sum{w}_log1p", np.log1p(clean[RAIN].rolling(w).sum()),
            f"log(1 + sum of PRECTOTCORR over days t-{w - 1}..t)", "rolling rain sum")
    wet = clean[RAIN] >= WET_MM
    add("wet_run_len", run_length(wet), "consecutive days >= 1 mm ending on day t (0 if day t is dry)", "spell")
    add("days_since_rain", run_length(~wet),
        "days since the last day >= 1 mm (0 if day t is wet; equals the current dry-run length)", "spell")
    for v in TENDENCY[fset]:
        add(f"{v}_chg1", base[v].diff(1), f"{v}(t) - {v}(t-1)", "tendency")
    add("DTR", clean["T2M_MAX"] - clean["T2M_MIN"], "diurnal temperature range T2M_MAX - T2M_MIN on day t", "derived")
    if fset == "B":
        add("DPD", clean["T2M"] - clean["T2MDEW"], "dew-point depression T2M - T2MDEW on day t", "derived")
    doy = clean.index.dayofyear.to_numpy()
    for k in (1, 2):
        add(f"doy_sin{k}", np.sin(2 * np.pi * k * doy / 365.25), f"sin(2*pi*{k}*day_of_year/365.25)", "season")
        add(f"doy_cos{k}", np.cos(2 * np.pi * k * doy / 365.25), f"cos(2*pi*{k}*day_of_year/365.25)", "season")
    add("month", clean.index.month.to_numpy(), "calendar month of day t (1-12)", "season")

    X = pd.DataFrame(cols, index=clean.index)
    return X, pd.DataFrame(rows, columns=["feature", "definition", "family"])


def build_targets(clean, col=RAIN):
    """Targets for row t from day t+1 (calendar next day; index must be a regular daily index)."""
    nxt = clean[col].shift(-1)
    Y = pd.DataFrame({"y_reg": nxt, "y_reg_log": np.log1p(nxt)}, index=clean.index)
    for name, thr in THRESHOLDS.items():
        Y[name] = (nxt >= thr).astype(float).where(nxt.notna())
    return Y


def assemble(clean):
    """Build A, B and targets; keep only rows complete in A, B and the target, so both sets share one index."""
    XA, dict_a = build_features(clean, "A")
    XB, dict_b = build_features(clean, "B")
    Y = build_targets(clean)
    keep = XA.notna().all(axis=1) & XB.notna().all(axis=1) & Y["y_reg"].notna()
    Y = Y[keep].astype({k: int for k in THRESHOLDS})
    return XA[keep], XB[keep], Y, dict_a, dict_b
