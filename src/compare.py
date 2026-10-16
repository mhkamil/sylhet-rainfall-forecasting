"""Stage 5.6: one table and paired block-bootstrap differences for baselines, classical and deep models on the DL split
(validation targets 2015-2019). Point estimates are means over seeds; bootstrap differences use seed-0 forecasts so that
every comparison is between two single, fully specified forecasts on the same resampled months."""
from pathlib import Path

import numpy as np
import pandas as pd

import deep_cls as DC
import deep_ts as D
import experiment as E
import metrics as M
import splits as S

DL_RESULTS, DL_OOF = E.SPLITS["dl"][1], E.SPLITS["dl"][2]


def unified_results():
    """Long table: family, model, seed, task, metric, value for every DL-split result."""
    parts = []
    c = pd.read_csv(DL_RESULTS)
    c["family"] = np.where(c.model.str.startswith("baseline"), "baseline", "classical")
    parts.append(c.assign(model=c.model + np.where(c.family == "classical", " (" + c.feature_set + ")", ""))
                 [["family", "model", "task", "seed", "metric", "value"]])
    r = pd.read_csv(D.RESULTS)
    r = r[r.lookback == 14]
    parts.append(r.assign(family="deep", task="reg", model=r.model + " " + r.target)[["family", "model", "task", "seed", "metric", "value"]])
    k = pd.read_csv(DC.RESULTS)
    parts.append(k.assign(family="deep", task="cls", model=k.model + " " + k.target + " " + k.option)
                 [["family", "model", "task", "seed", "metric", "value"]])
    return pd.concat(parts, ignore_index=True)


def summary(u, task, metrics, sort_by, ascending):
    s = u[(u.task == task) & u.metric.isin(metrics)]
    g = s.groupby(["family", "model", "metric"])["value"]
    cell = (g.mean().map("{:.3f}".format) + " ± " + g.std().fillna(0).map("{:.3f}".format)).unstack("metric")[metrics]
    cell["n_seeds"] = s.groupby(["family", "model"])["seed"].nunique()
    order = g.mean().unstack("metric")[sort_by].sort_values(ascending=ascending).index
    return cell.loc[order]


def forecasts(task):
    """Seed-0 validation forecasts {model label: array} for baselines, classical (Set B) and deep models."""
    X, Y = E.load("A")
    tr, va = S.dl_split(X.index)
    refs = E.references(task, X, Y, E.rain_today(X.index), tr, va)
    out = {f"baseline:{k}": np.asarray(v) for k, v in refs.items()}
    for f in sorted(Path(DL_OOF).glob(f"*__{task}__B__seed0.parquet")):
        out[f.name.split("__")[0] + " (B)"] = pd.read_parquet(f).loc[va, "pred"].to_numpy()
    if task == "reg":
        for f in sorted(D.DEEP_DIR.glob("*_L14__seed0.parquet")):
            kind, transform = f.name.split("_L14")[0].split("_", 1)
            if transform == "bin":
                continue
            o = pd.read_parquet(f).loc[va]
            labels = {"pred": "q50", "pred_mean": "qmean"} if "q0.5" in o else (
                {"pred": transform} if transform == "mm" else {"pred": "log_smear", "pred_naive": "log_naive"})
            out.update({f"{kind} {lab}": o[col].to_numpy() for col, lab in labels.items()})
    else:
        for f in sorted(DC.PRED_DIR.glob("*__seed0.parquet")):
            kind, rest = f.name.split("__")[0].split("_", 1)
            transform, option = rest.split("_", 1)
            out[f"{kind} {transform} {option}"] = pd.read_parquet(f).loc[va, "pred"].to_numpy()
    y = Y.loc[va, "y_cls" if task == "cls" else "y_reg"].to_numpy()
    return va, y, refs["climatology"], out


def paired(task, refs_to, n_boot=1000):
    """Block-bootstrap (calendar months of the target date) differences model - reference.
    cls: BSS difference (higher = better); reg: MAE difference (negative = better) and MSESS difference."""
    va, y, clim, F = forecasts(task)
    dates = S.target_dates(va)
    rows = []
    for ref in refs_to:
        for name, p in F.items():
            if name == ref:
                continue
            r = F[ref]
            if task == "cls":
                fns = {"dBSS": lambda yy, a, b, c: (np.mean((b - yy) ** 2) - np.mean((a - yy) ** 2)) / np.mean((c - yy) ** 2)}
            else:
                fns = {"dMAE": lambda yy, a, b, c: np.mean(np.abs(a - yy)) - np.mean(np.abs(b - yy)),
                       "dMSESS": lambda yy, a, b, c: (np.mean((b - yy) ** 2) - np.mean((a - yy) ** 2)) / np.mean((c - yy) ** 2)}
            for metric, fn in fns.items():
                est, lo, hi = M.block_bootstrap(fn, dates, y, p, r, clim, n_boot=n_boot)
                rows.append({"model": name, "reference": ref, "metric": metric, "estimate": est, "ci_low": lo,
                             "ci_high": hi, "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)
