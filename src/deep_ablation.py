"""Stage 6.4 ablations for the tuned deep model(s) on the DL split (train targets <= 2014, validation 2015-2019).

Deep models use raw daily variables (no engineered features), so the applicable ablations are:
  set_A_vars  past covariates = the Set-A variables only (RH2M, T2M, T2M_MAX, T2M_MIN, WS2M);
  win_2000    training (inner and refit) starts in 2000 instead of 1981;
  imerg_1998  IMERG_PRECTOT is the rain variable (target, and rain as past covariate for Bernoulli), 1998+;
  thr_0.1 / thr_2.5  Bernoulli target at 0.1 / 2.5 mm (classification only);
  lookback 14 / 30 / 60 -> consolidated from the Stage-5 runs (lookback_deltas), no new fits.
References are fitted on the DL-split training rows of the variant's own target (same as the classical ablations).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import ablation as A
import baselines as B
import deep_ts as D
import deep_tuning as DT
import experiment as E
import metrics as M
import splits as S

OUT = D.DEEP_DIR / "ablation"
RESULTS = E.ROOT / "reports/tables/deep_ablation_results.csv"
SET_A_VARS = ["RH2M", "T2M", "T2M_MAX", "T2M_MIN", "WS2M"]
VARIANTS = {"full": {}, "set_A_vars": {"past": SET_A_VARS}, "win_2000": {"start": "2000-01-01"},
            "imerg_1998": {"source": "IMERG_PRECTOT"}, "thr_0.1": {"wet_mm": 0.1, "tasks": ["cls"]},
            "thr_2.5": {"wet_mm": 2.5, "tasks": ["cls"]}}
CLASSICAL_TWIN = {"imerg_1998": "imerg_1998", "thr_0.1": "thr_0.1", "thr_2.5": "thr_2.5"}   # for the references
SEEDS = [0, 1, 2]


def variant_input(name):
    v = VARIANTS[name]
    df, fut = D.load_daily()
    if v.get("source"):
        raw = pd.read_parquet(E.PROC / "clean_daily.parquet").loc[:D.DEV_END, v["source"]]
        df = df.assign(**{D.TARGET: raw}).loc["1998-01-01":]
        fut = fut.loc[df.index]
    if v.get("start"):
        df, fut = df.loc[v["start"]:], fut.loc[v["start"]:]
    return {"df": df, "fut": fut, "past": v.get("past"), "wet_mm": v.get("wet_mm", D.WET_MM)}


def references(name, task):
    """Validation index, observed target and the reference forecasts (DL split) for the variant's target."""
    X, Y, rain, season = A.variant_data(CLASSICAL_TWIN.get(name, "full"))
    tr, va = S.dl_split(X.index)
    target = {"thr_0.1": "y_cls_01", "thr_2.5": "y_cls_25"}.get(name, "y_cls") if task == "cls" else "y_reg"
    thr = VARIANTS[name].get("wet_mm", 1.0)
    df = season.join(Y)
    refs = B.classification_baselines(df, rain, tr, va, target, thr) if task == "cls" else \
        B.regression_baselines(df, rain, tr, va)
    return va, Y.loc[va, target], refs


def stem(kind, task, name, seed):
    """'full' = the tuned 5-seed re-run of Stage 6.3 (models/deep/tuned); variants live in models/deep/ablation."""
    if name == "full":
        return DT.TUNED_DIR / f"{DT.tag(kind, task)}__seed{seed}"
    return OUT / f"{kind}_{task}_{name}__seed{seed}"


def run(kind, task, name, seed):
    """One fit of the tuned configuration on a variant; saves forecasts and returns tidy scores."""
    import torch
    torch.set_num_threads(1)
    transform = DT.ROUTE[task][kind]
    params = DT.load_best(kind, task)["best_params"]
    tag, st = f"{kind}_{task}_{name}", stem(kind, task, name, seed)
    if Path(f"{st}.parquet").exists():
        out = pd.read_parquet(f"{st}.parquet")
    else:
        out, _ = D.run(kind, transform, DT.L, seed, params=params, variant=variant_input(name), tag=tag, out_dir=OUT)
    va, y, refs = references(name, task)
    out = out.loc[va]
    if task == "cls":
        sl = pd.read_parquet(f"{st}__slice.parquet")
        thr = M.choose_threshold((sl["y_true"] >= VARIANTS[name].get("wet_mm", 1.0)).astype(int), sl["p"], "CSI")
        m = E.score("cls", y.to_numpy(), out["p"].to_numpy(), thr, refs, va)
    else:
        m = E.score("reg", y.to_numpy(), out[DT.POINT[transform]].to_numpy(), np.nan, refs, va)
    return pd.DataFrame([{"model": kind, "task": task, "variant": name, "seed": seed, "metric": k, "value": v}
                         for k, v in m.items()])


def forecasts(kind, task, name):
    """Seed-mean validation forecast, observed target and climatology for the variant."""
    transform = DT.ROUTE[task][kind]
    col = "p" if task == "cls" else DT.POINT[transform]
    va, y, refs = references(name, task)
    preds = [pd.read_parquet(f"{stem(kind, task, name, s)}.parquet").loc[va, col].to_numpy() for s in SEEDS]
    return va, y.to_numpy(), np.mean(preds, axis=0), np.asarray(refs["climatology"])


def delta(kind, task, name, n_boot=1000):
    """Variant minus full (same deep model), paired monthly block bootstrap; only for variants with the same target."""
    va, y, p, clim = forecasts(kind, task, name)
    va0, y0, p0, clim0 = forecasts(kind, task, "full")
    assert va.equals(va0) and np.allclose(y, y0)
    return _boot(task, va, y, p, p0, clim0, {"model": kind, "task": task, "variant": name, "versus": "full"}, n_boot)


def _boot(task, va, y, p, q, clim, info, n_boot=1000):
    fns = {"dBSS": lambda yy, a, b, c: (np.mean((b - yy) ** 2) - np.mean((a - yy) ** 2)) / np.mean((c - yy) ** 2)} \
        if task == "cls" else \
        {"dMAE": lambda yy, a, b, c: np.mean(np.abs(a - yy)) - np.mean(np.abs(b - yy)),
         "dMSESS": lambda yy, a, b, c: (np.mean((b - yy) ** 2) - np.mean((a - yy) ** 2)) / np.mean((c - yy) ** 2)}
    rows = []
    for metric, fn in fns.items():
        est, lo, hi = M.block_bootstrap(fn, S.target_dates(va), y, p, q, clim, n_boot=n_boot)
        rows.append({**info, "metric": metric, "n_days": len(va), "estimate": est, "ci_low": lo, "ci_high": hi,
                     "significant": bool(lo > 0 or hi < 0)})
    return rows


def lookback_deltas(n_boot=1000):
    """Stage-5 default configurations: L = 30 and 60 minus L = 14 (seed-mean over the seeds all three lookbacks share)."""
    va, y, refs = D.dl_references()
    clim = np.asarray(refs["climatology"])
    rows = []
    for kind, transform, col in [("tsmixer", "mm", "pred"), ("tsmixer", "log", "pred"), ("tide", "mm", "pred"),
                                 ("tide", "log", "pred"), ("tft", "mm", "pred_mean")]:
        seeds = sorted(set.intersection(*[{int(f.stem.split("seed")[1]) for f in D.DEEP_DIR.glob(
            f"{kind}_{transform}_L{L}__seed?.parquet")} for L in D.LOOKBACKS]))
        pred = {L: np.mean([pd.read_parquet(D.DEEP_DIR / f"{kind}_{transform}_L{L}__seed{s}.parquet").loc[va, col]
                            for s in seeds], axis=0) for L in D.LOOKBACKS}
        for L in (30, 60):
            rows += _boot("reg", va, y.to_numpy(), pred[L], pred[14], clim,
                          {"model": kind, "task": "reg", "variant": f"{kind} {transform} L{L}", "versus": "L14",
                           "seeds": ",".join(map(str, seeds))}, n_boot)
    return pd.DataFrame(rows)


def _job(key):
    import os
    import warnings
    os.environ["OMP_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore")
    return run(*key)


def run_all(kinds, workers=4):
    """kinds: {"cls": kind, "reg": kind} -- the deep model ablated for each task."""
    from concurrent.futures import ProcessPoolExecutor
    keys = [(kinds[task], task, name, seed) for task in ("cls", "reg") for name, v in VARIANTS.items()
            if task in v.get("tasks", ["cls", "reg"]) for seed in SEEDS]
    with ProcessPoolExecutor(workers) as ex:
        res = pd.concat(list(ex.map(_job, keys)))
    res.to_csv(RESULTS, index=False)
    return res
