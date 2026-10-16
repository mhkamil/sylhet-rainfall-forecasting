"""Stage 6.4 ablations of the tuned classical models on the 5 expanding CV folds (development data only, targets <= 2019).

Every variant changes ONE thing relative to "full" (Feature Set B, PRECTOTCORR, all training years, >= 1 mm, class
weights on) and is run through the same fold protocol (experiment.fit_fold / reg_variants.hurdle_amount).
References (climatology, seasonal + persistence) are always fitted on the fold's complete training rows for the variant's
target, so a training-window restriction changes the model only, not the yardstick.
Models: classification LightGBM (tuned, 5 seeds) and logreg_l1 (tuned; deterministic in practice, seed std 0 -> seed 0);
regression CatBoost Tweedie (tuned, 5 seeds) and the MSESS-tuned hurdle (P(wet) = the same variant's LightGBM OOF
probability x tuned CatBoost wet-day amount model, 5 seeds).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baselines as B
import experiment as E
import features as F
import metrics as M
import reg_variants as R
import splits as S
import tuning as T

OUT = E.ROOT / "models/ablation"
PARTS = OUT / "parts"
RESULTS = E.ROOT / "reports/tables/ablation_results.csv"
MIN_TRAIN_YEARS = 4                       # inner-train >= 1 year + 3-year slice
SOLAR = "ALLSKY_SFC_SW_DWN"
DICT = pd.read_csv(E.ROOT / "reports/tables/02_feature_dictionary.csv")
FAMILIES = {"lags": ["lag"], "rolling": ["rolling mean", "rolling rain sum"], "spell": ["spell"], "season": ["season"],
            "tendency": ["tendency"], "derived": ["derived"]}

VARIANTS = {
    "full": {},
    "set_A": {"fset": "A"},
    "B_1984": {"start": 1984},                              # twin of B+solar (same training years)
    "B+solar_1984": {"start": 1984, "solar": True},
    **{f"drop_{k}": {"drop": v} for k, v in FAMILIES.items()},
    "thr_0.1": {"target": "y_cls_01", "thr": 0.1, "tasks": ["cls"]},
    "thr_2.5": {"target": "y_cls_25", "thr": 2.5, "tasks": ["cls"]},
    "no_class_weights": {"weights": False, "tasks": ["cls"]},
    "win_1998": {"start": 1998},                            # twin of IMERG (same training years)
    "win_2000": {"start": 2000},
    "win_2010": {"start": 2010},
    "imerg_1998": {"source": "IMERG_PRECTOT", "start": 1998},
}
MODELS = {"cls": ["lightgbm", "logreg_l1"], "reg": ["catboost_tweedie", "hurdle_catboost_log_msess"]}
SEEDS = {"lightgbm": [0, 1, 2, 3, 4], "logreg_l1": [0], "catboost_tweedie": [0, 1, 2, 3, 4],
         "hurdle_catboost_log_msess": [0, 1, 2, 3, 4]}


# ---------- data of one variant ----------
def _imerg_tables():
    """Feature Set B and targets rebuilt with IMERG_PRECTOT as the rain variable (features AND targets), 1998+."""
    clean = pd.read_parquet(E.PROC / "clean_daily.parquet")
    clean = clean.assign(PRECTOTCORR=clean["IMERG_PRECTOT"])
    X, _ = F.build_features(clean, "B")
    Y = F.build_targets(clean)
    keep = X.notna().all(axis=1) & Y["y_reg"].notna()
    return X[keep], Y[keep].astype({k: int for k in F.THRESHOLDS}), clean["PRECTOTCORR"]


def _solar_features(index):
    s = pd.read_parquet(E.PROC / "clean_daily.parquet")[SOLAR]
    cols = {SOLAR: s, **{f"{SOLAR}_lag{k}": s.shift(k) for k in F.LAGS},
            **{f"{SOLAR}_mean{w}": s.rolling(w).mean() for w in F.WINDOWS}}
    return pd.DataFrame(cols).reindex(index)


def variant_data(name):
    """X, Y, rain-today series and the season columns for references; development rows only."""
    v = VARIANTS[name]
    if v.get("source"):
        X, Y, rain = _imerg_tables()
    else:
        X, Y = E.load(v.get("fset", "B"))
        rain = E.rain_today(X.index)
    if v.get("solar"):
        X = X.join(_solar_features(X.index))
        X = X[X.notna().all(axis=1)]
    if v.get("drop"):
        gone = DICT.loc[DICT.family.isin(v["drop"]) & DICT.set_B, "feature"]
        X = X.drop(columns=[c for c in gone if c in X])
    dev = S.dev_index(X.index)
    X, Y = X.loc[dev], Y.loc[dev]
    doy = X.index.dayofyear.to_numpy()
    season = pd.DataFrame({f"doy_{f}{k}": fn(2 * np.pi * k * doy / 365.25) for k in (1, 2)
                           for f, fn in (("sin", np.sin), ("cos", np.cos))}, index=X.index)[B.SEASON_COLS]
    return X, Y, rain.reindex(X.index), season


def folds(name, index):
    """(fold number, train, validation) with the variant's training-window restriction; folds that would have
    fewer than MIN_TRAIN_YEARS training years are skipped. Also returns the unrestricted training rows (references)."""
    start = VARIANTS[name].get("start")
    out = []
    for k, (tr, va) in enumerate(S.cv_folds(index), 1):
        tr_model = tr[S.target_dates(tr).year >= start] if start else tr
        if S.target_dates(tr_model).year.nunique() >= MIN_TRAIN_YEARS:
            out.append((k, tr_model, va, tr))
    return out


def factory(model, task, name):
    """Tuned estimator factory; with no_class_weights the class weighting of the tuned model is switched off."""
    tuned = T.load_best(model, task)["best_params"]
    f = T.MODELS[task][model][1](tuned, task)
    if VARIANTS[name].get("weights", True):
        return f

    def unweighted(seed, y):
        est = f(seed, y)
        key = "class_weight" if hasattr(est, "class_weight") else "logisticregression__class_weight"
        return est.set_params(**{key: None})
    return unweighted


# ---------- one run ----------
def run(name, model, seed):
    task = "cls" if model in MODELS["cls"] else "reg"
    v = VARIANTS[name]
    target, thr = v.get("target", "y_cls" if task == "cls" else "y_reg"), v.get("thr", 1.0)
    X, Y, rain, season = variant_data(name)
    y, ref_df = Y[target], season.join(Y)
    if model.startswith("hurdle"):
        params = T.load_best(model, task)["best_params"]
        p_wet = pd.read_parquet(OUT / f"{name}__lightgbm__seed{seed}.parquet")["pred"]
    rows, oof, t0 = [], [], time.time()
    for k, tr, va, tr_full in folds(name, X.index):
        if task == "cls":
            refs = B.classification_baselines(ref_df, rain, tr_full, va, target, thr)
            pred, th, n_trees, _ = E.fit_fold(factory(model, task, name), task, seed, X, y, tr, va)
        else:
            refs = B.regression_baselines(ref_df, rain, tr_full, va)
            if model.startswith("hurdle"):
                wet = tr[y[tr].to_numpy() >= R.WET_MM]
                pred = p_wet.loc[va].to_numpy() * R.hurdle_amount(T.b_cat(params, task), seed, X, y, wet, va, "log")
                th, n_trees = np.nan, np.nan
            else:
                pred, th, n_trees, _ = E.fit_fold(factory(model, task, name), task, seed, X, y, tr, va)
        m = E.score(task, y[va].to_numpy(), pred, th, refs, va)
        rows.append({"fold": k, **m, "n_trees": n_trees})
        oof.append(pd.DataFrame({"fold": k, "y_true": y[va].to_numpy(), "pred": pred, "threshold": th,
                                 "clim": refs["climatology"], "seas_pers": refs["seasonal+persistence"]}, index=va))
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    pd.concat(oof).to_parquet(OUT / f"{name}__{model}__seed{seed}.parquet")
    res = pd.DataFrame(rows).melt(id_vars="fold", var_name="metric", value_name="value").assign(
        variant=name, model=model, task=task, seed=seed, n_features=X.shape[1], seconds=time.time() - t0)
    res.to_csv(PARTS / f"{name}__{model}__seed{seed}.csv", index=False)
    return res


def jobs():
    """All (variant, model, seed) runs; LightGBM classification runs first because the hurdle needs them."""
    first, second = [], []
    for name, v in VARIANTS.items():
        for task in v.get("tasks", ["cls", "reg"]):
            for model in MODELS[task]:
                for seed in SEEDS[model]:
                    (second if model.startswith("hurdle") else first).append((name, model, seed))
    return first, second


def _job(key):
    import os
    import warnings
    os.environ["OMP_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore")
    if (OUT / f"{key[0]}__{key[1]}__seed{key[2]}.parquet").exists():
        return key, 0.0
    t0 = time.time()
    run(*key)
    return key, time.time() - t0


def run_all(workers=4):
    from concurrent.futures import ProcessPoolExecutor
    first, second = jobs()
    with ProcessPoolExecutor(workers) as ex:
        for batch in (first, second):
            for key, sec in ex.map(_job, batch):
                print(key, f"{sec:.0f}s", flush=True)
    print("rows:", len(merge()), flush=True)


def merge():
    res = pd.concat([pd.read_csv(f) for f in sorted(PARTS.glob("*.csv"))], ignore_index=True)
    res.to_csv(RESULTS, index=False)
    return res


# ---------- paired block-bootstrap differences ----------
def seed_mean(name, model):
    files = sorted(OUT.glob(f"{name}__{model}__seed*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    out = frames[0].copy()
    out["pred"] = np.mean([f["pred"].to_numpy() for f in frames], axis=0)
    return out


def delta(name, model, base="full", n_boot=1000, other_model=None, fold_subset=None):
    """Variant minus base on the common validation days (monthly block bootstrap of the target date, 95 % CI).
    cls: dBSS = (Brier_base - Brier_var) / Brier_clim (> 0 = variant better); reg: dMAE (< 0 = better) and dMSESS.
    With other_model, compares two models within the same variant instead (model minus other_model).
    fold_subset: restrict to these validation folds (e.g. (4, 5) = 2014-2019)."""
    a = seed_mean(name, model)
    b = seed_mean(name, other_model) if other_model else seed_mean(base, model)
    idx = a.index.intersection(b.index)
    if fold_subset:
        idx = idx[a.loc[idx, "fold"].isin(fold_subset).to_numpy()]
    a, b = a.loc[idx], b.loc[idx]
    y, clim, dates = a["y_true"].to_numpy(), b["clim"].to_numpy(), S.target_dates(idx)
    assert np.allclose(y, b["y_true"].to_numpy()), "different targets: use other_model comparisons"
    task = "cls" if model in MODELS["cls"] else "reg"
    fns = {"dBSS": lambda yy, p, q, c: (np.mean((q - yy) ** 2) - np.mean((p - yy) ** 2)) / np.mean((c - yy) ** 2)} \
        if task == "cls" else \
        {"dMAE": lambda yy, p, q, c: np.mean(np.abs(p - yy)) - np.mean(np.abs(q - yy)),
         "dMSESS": lambda yy, p, q, c: (np.mean((q - yy) ** 2) - np.mean((p - yy) ** 2)) / np.mean((c - yy) ** 2)}
    rows = []
    for metric, fn in fns.items():
        est, lo, hi = M.block_bootstrap(fn, dates, y, a["pred"].to_numpy(), b["pred"].to_numpy(), clim, n_boot=n_boot)
        rows.append({"variant": name, "model": model, "versus": other_model or base, "metric": metric,
                     "folds": ",".join(map(str, sorted(a.fold.unique()))), "n_days": len(idx), "estimate": est,
                     "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0 or hi < 0)})
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    run_all(ap.parse_args().workers)
