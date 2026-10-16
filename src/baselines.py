"""Stage 3 reference forecasts. Every baseline is fitted on training rows only and then predicts any rows.

`df` is a feature table (features_A/B), `rain_today` is PRECTOTCORR (mm) on the feature date t.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

SEASON_COLS = ["doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"]
CLIM_HALF_WINDOW = 7


def doy_climatology(dates, values, half_window=CLIM_HALF_WINDOW):
    """Mean of `values` per day of year (1-366), smoothed with a circular +/- half_window-day moving window.
    Pools the raw sums and counts, so days with few samples (29 Feb) are not over-weighted."""
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy() - 1
    sums = np.bincount(doy, weights=np.asarray(values, float), minlength=366)
    counts = np.bincount(doy, minlength=366)
    shifts = range(-half_window, half_window + 1)
    return sum(np.roll(sums, k) for k in shifts) / sum(np.roll(counts, k) for k in shifts)


def _doy(idx):
    return pd.DatetimeIndex(idx).dayofyear.to_numpy() - 1


def classification_baselines(df, rain_today, train_idx, eval_idx, target, thr):
    """Rain-probability forecasts {name: array on eval_idx} for target `target` (threshold `thr` mm)."""
    y_tr = df.loc[train_idx, target].to_numpy()
    wet_tr, wet_ev = rain_today[train_idx].to_numpy() >= thr, rain_today[eval_idx].to_numpy() >= thr
    seasonal = LogisticRegression(max_iter=1000).fit(df.loc[train_idx, SEASON_COLS], y_tr)
    sp_tr = df.loc[train_idx, SEASON_COLS].assign(wet_today=wet_tr.astype(int))
    sp_ev = df.loc[eval_idx, SEASON_COLS].assign(wet_today=wet_ev.astype(int))
    seas_pers = LogisticRegression(max_iter=1000).fit(sp_tr, y_tr)
    return {
        "majority": np.full(len(eval_idx), y_tr.mean()),
        "persistence": np.where(wet_ev, y_tr[wet_tr].mean(), y_tr[~wet_tr].mean()),
        "climatology": doy_climatology(train_idx + pd.Timedelta(days=1), y_tr)[_doy(eval_idx + pd.Timedelta(days=1))],
        "seasonal": seasonal.predict_proba(df.loc[eval_idx, SEASON_COLS])[:, 1],
        "seasonal+persistence": seas_pers.predict_proba(sp_ev)[:, 1],      # extra reference, not one of the four
    }


def regression_baselines(df, rain_today, train_idx, eval_idx, target="y_reg"):
    """Rainfall-amount forecasts in mm {name: array on eval_idx}; negative predictions are clipped to 0."""
    y_tr = df.loc[train_idx, target].to_numpy()
    seasonal = Ridge(alpha=1.0).fit(df.loc[train_idx, SEASON_COLS], y_tr)
    seas_pers = Ridge(alpha=1.0).fit(df.loc[train_idx, SEASON_COLS].assign(rain_today=rain_today[train_idx]), y_tr)
    return {
        "majority": np.full(len(eval_idx), y_tr.mean()),
        "persistence": rain_today[eval_idx].to_numpy(),
        "climatology": doy_climatology(train_idx + pd.Timedelta(days=1), y_tr)[_doy(eval_idx + pd.Timedelta(days=1))],
        "seasonal": np.clip(seasonal.predict(df.loc[eval_idx, SEASON_COLS]), 0, None),
        "seasonal+persistence": np.clip(seas_pers.predict(
            df.loc[eval_idx, SEASON_COLS].assign(rain_today=rain_today[eval_idx])), 0, None),
    }
