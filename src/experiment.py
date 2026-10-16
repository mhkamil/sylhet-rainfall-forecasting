"""Stage 4 shared cross-validation runner.

run_cv(name, factory, task, fset, seed) runs the 5 expanding-window folds of src/splits.py:
  1. inner split of the training fold: inner-train + the last 3 target years ("slice"), with the same 1-day purge;
  2. fit a copy on inner-train, predict the slice -> decision threshold (classification, max CSI) and, for gradient
     boosters, the early-stopping number of trees;
  3. refit on the whole training fold (boosters with that number of trees) and predict the validation block.
The validation block is used only for scoring, never for any choice. Preprocessing lives inside each model's Pipeline,
so it is fitted on training rows only. Out-of-fold predictions go to models/oof/, tidy scores to cv_results.csv.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import baselines as B
import metrics as M
import splits as S

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
RESULTS = ROOT / "reports/tables/cv_results.csv"
OOF_DIR = ROOT / "models/oof"
TARGET = {"cls": "y_cls", "reg": "y_reg"}
SLICE_YEARS = 3
EARLY_STOP_ROUNDS = 100
MONSOON = (6, 7, 8, 9)
KEYS = ["model", "task", "feature_set", "seed"]


def load(fset):
    """Features (all non-target columns) and targets of feature set 'A' or 'B'."""
    df = pd.read_parquet(PROC / f"features_{fset}.parquet")
    return df[[c for c in df if not c.startswith("y_")]], df[[c for c in df if c.startswith("y_")]]


def rain_today(index):
    return pd.read_parquet(PROC / "clean_daily.parquet")["PRECTOTCORR"].reindex(index)


def inner_split(train_idx, years=SLICE_YEARS):
    """Inner-train and the last `years` target years of a training fold (1-day purge between them)."""
    last = S.target_dates(train_idx).year.max()
    return S.make_split(train_idx, last - years, last - years + 1, last)


# ---------- gradient-boosting helpers (each library has its own early-stopping API) ----------
def booster_family(est):
    name = type(est).__name__
    return next((f for f in ("XGB", "LGBM", "CatBoost", "TorchMLP") if name.startswith(f)), None)


def fit_early_stopping(est, Xa, ya, Xb, yb, rounds=EARLY_STOP_ROUNDS):
    """Fit with early stopping on (Xb, yb); return the best number of trees (epochs for the MLP)."""
    fam = booster_family(est)
    if fam == "TorchMLP":
        est.fit(Xa, ya, eval_set=(Xb, yb))
        return est.best_epoch_
    if fam == "XGB":
        est.set_params(early_stopping_rounds=rounds)
        est.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        return est.best_iteration + 1
    if fam == "LGBM":
        import lightgbm as lgb
        est.fit(Xa, ya, eval_X=(Xb,), eval_y=(yb,), callbacks=[lgb.early_stopping(rounds, verbose=False)])   # lightgbm 4.7 API
        return est.best_iteration_
    est.fit(Xa, ya, eval_set=(Xb, yb), early_stopping_rounds=rounds, verbose=False)   # CatBoost
    return est.get_best_iteration() + 1


def set_trees(est, n):
    if booster_family(est) == "XGB":
        est.set_params(n_estimators=n, early_stopping_rounds=None)
    elif booster_family(est) == "LGBM":
        est.set_params(n_estimators=n)
    elif booster_family(est) == "TorchMLP":
        est.set_params(max_epochs=n, patience=None)          # refit for exactly the early-stopped number of epochs
    else:
        est.set_params(iterations=n)


def predict(est, X, task):
    return est.predict_proba(X)[:, 1] if task == "cls" else np.clip(est.predict(X), 0, None)


# ---------- scoring ----------
def score(task, y, p, threshold, refs, idx):
    """All metrics for one validation block, incl. skill vs seasonal+persistence and monsoon-only scores."""
    y = np.asarray(y)
    monsoon = S.target_dates(idx).month.isin(MONSOON)
    if task == "cls":
        m = M.classification_metrics(y, p, threshold=threshold, p_ref=refs["climatology"])
        m["BSS_vs_seas_pers"] = 1 - m["Brier"] / np.mean((refs["seasonal+persistence"] - y) ** 2)
        m["ROC_AUC_monsoon"] = roc_auc_score(y[monsoon], p[monsoon])
    else:
        m = M.regression_metrics(y, p, y_ref=refs["climatology"])
        m["MSESS_vs_seas_pers"] = 1 - np.mean((p - y) ** 2) / np.mean((refs["seasonal+persistence"] - y) ** 2)
        m["MAE_monsoon"] = np.mean(np.abs(p - y)[monsoon])
    return m


def references(task, X, Y, rain, tr, va):
    """Climatology and seasonal+persistence forecasts of this fold (fitted on its training rows)."""
    df = X[B.SEASON_COLS].join(Y)
    if task == "cls":
        return B.classification_baselines(df, rain, tr, va, "y_cls", 1.0)
    return B.regression_baselines(df, rain, tr, va)


def tidy(rows, name, task, fset, seed):
    out = pd.DataFrame(rows).melt(id_vars="fold", var_name="metric", value_name="value")
    return out.assign(model=name, task=task, feature_set=fset, seed=seed)[KEYS[:3] + ["fold", "seed", "metric", "value"]]


def append_results(new, path=RESULTS):
    """Write rows, replacing earlier rows with the same (model, task, feature_set, seed) -> re-runs never duplicate."""
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path)
        key = lambda d: d[KEYS].astype(str).agg("|".join, axis=1)
        new = pd.concat([old[~key(old).isin(set(key(new)))], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)


# ---------- runners ----------
def run_cv(name, factory, task, fset, seed=0, save=True, results_path=RESULTS, oof_dir=OOF_DIR):
    """factory(seed, y_train) -> unfitted estimator. Returns (tidy scores, out-of-fold predictions, fold log)."""
    X, Y = load(fset)
    y, rain = Y[TARGET[task]], rain_today(X.index)
    rows, oof, log = [], [], []
    for k, (tr, va) in enumerate(S.cv_folds(X.index), 1):
        t0 = time.time()
        est = factory(seed, y[tr])
        boost, thr, n_trees = booster_family(est) is not None, np.nan, np.nan
        if boost or task == "cls":
            itr, sl = inner_split(tr)
            inner = factory(seed, y[itr])
            if boost:
                n_trees = fit_early_stopping(inner, X.loc[itr], y[itr], X.loc[sl], y[sl])
            else:
                inner.fit(X.loc[itr], y[itr])
            if task == "cls":
                thr = M.choose_threshold(y[sl], predict(inner, X.loc[sl], task), "CSI")
            log.append({"fold": k, "inner_train_last_target": S.target_dates(itr).max().date(),
                        "slice_targets": f"{S.target_dates(sl).min().date()}..{S.target_dates(sl).max().date()}"})
        if boost:
            set_trees(est, n_trees)
        est.fit(X.loc[tr], y[tr])
        p = predict(est, X.loc[va], task)
        m = score(task, y[va], p, thr, references(task, X, Y, rain, tr, va), va)
        rows.append({"fold": k, **m, "n_trees": n_trees, "fit_seconds": time.time() - t0})
        oof.append(pd.DataFrame({"fold": k, "y_true": y[va].to_numpy(), "pred": p, "threshold": thr}, index=va))
    res, oof = tidy(rows, name, task, fset, seed), pd.concat(oof)
    if save:
        append_results(res, results_path)
        Path(oof_dir).mkdir(parents=True, exist_ok=True)
        oof.to_parquet(Path(oof_dir) / f"{name}__{task}__{fset}__seed{seed}.parquet")
    return res, oof, pd.DataFrame(log)


def run_baselines(task, save=True, results_path=RESULTS):
    """Stage-3 baselines scored with the same metrics and stored in the same table (feature_set '-', seed 0)."""
    X, Y = load("A")
    y, rain = Y[TARGET[task]], rain_today(X.index)
    rows = {}
    for k, (tr, va) in enumerate(S.cv_folds(X.index), 1):
        refs = references(task, X, Y, rain, tr, va)
        fit_preds = references(task, X, Y, rain, tr, tr) if task == "cls" else None
        for name, p in refs.items():
            thr = np.nan if task == "reg" else (0.5 if name == "majority" else M.choose_threshold(y[tr], fit_preds[name]))
            rows.setdefault(name, []).append({"fold": k, **score(task, y[va], p, thr, refs, va)})
    res = pd.concat([tidy(r, f"baseline:{n}", task, "-", 0) for n, r in rows.items()])
    if save:
        append_results(res, results_path)
    return res


def summarise(res, metrics):
    """Mean over folds and seeds, fold std (of seed-averaged folds), folds 1-3 vs 4-5 means, and seed std."""
    r = res[res.metric.isin(metrics)]
    ids = ["model", "task", "feature_set", "metric"]
    per_fold = r.groupby(ids + ["fold"])["value"].mean().reset_index()
    per_seed = r.groupby(ids + ["seed"])["value"].mean().reset_index()
    g = per_fold.groupby(ids)["value"]
    out = pd.DataFrame({"mean": g.mean(), "fold_std": g.std(),
                        "folds_1_3": per_fold[per_fold.fold <= 3].groupby(ids)["value"].mean(),
                        "folds_4_5": per_fold[per_fold.fold >= 4].groupby(ids)["value"].mean(),
                        "seed_std": per_seed.groupby(ids)["value"].std(),
                        "n_seeds": per_seed.groupby(ids)["seed"].nunique()})
    return out.reset_index()
