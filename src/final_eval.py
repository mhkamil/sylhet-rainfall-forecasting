"""Stage 7 final evaluation (frozen). Nothing in this file may be changed after the freeze commit.

Two entry points:
  freeze()                    development data only. Computes everything that must be decided BEFORE the test years are
                              seen (CSI thresholds, early-stopping trees/epochs, smearing factors) with the inner split of the
                              final training set (inner-train targets <= 2016-12-30, slice 2017-01-01..2019-12-30) and writes
                              final_config.yaml. Run once, at Stage 7.1.
  run(confirm_final=True)     Stage 7.2 only. Refits every frozen model ONCE on all development rows (targets
                              1981-01-31..2019-12-30; 1-day purge before the test period) with the frozen values, predicts the
                              locked test targets 2020-01-01..2025-12-31, and saves predictions per model and seed in
                              models/test/ plus a manifest. It refuses to run without confirm_final=True, if final_config.yaml
                              or src/ has uncommitted changes, or if the data files differ from the frozen hashes.
Statistics for Stage 7.2 (pre-registered): dm_test (Diebold-Mariano, Bartlett/Newey-West HAC variance, Harvey-Leybourne-
Newbold correction), holm (step-down adjustment); block bootstrap = metrics.block_bootstrap.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

import baselines as B
import experiment as E
import metrics as M
import models as Mo
import reg_variants as R
import splits as S
import tuning as T

ROOT = E.ROOT
CONFIG = ROOT / "final_config.yaml"
TEST_DIR = ROOT / "models/test"
SEEDS = [0, 1, 2, 3, 4]
FSET = "B"
L = 14
TEST_FIRST, TEST_LAST = pd.Timestamp("2020-01-01"), pd.Timestamp("2025-12-31")
DEEP_FINAL_DATES = {"inner_last": pd.Timestamp("2016-12-30"), "slice_first": pd.Timestamp("2017-01-01"),
                    "train_last": pd.Timestamp("2019-12-30"), "val_first": TEST_FIRST, "val_last": TEST_LAST}
DATA_FILES = ["data/processed/features_B.parquet", "data/processed/clean_daily.parquet"]

# ---------- the frozen model set (DECISIONS 6.6-6.13) ----------
# classical: name -> (base model in tuning.MODELS or Stage-4 default factory, "tuned"/"default")
CLASSICAL = {
    "cls": {"logreg_none": "default", "logreg_l2": "tuned", "logreg_l1": "tuned", "logreg_enet": "tuned",
            "sgd_logloss": "tuned", "random_forest": "tuned", "xgboost": "tuned", "lightgbm": "tuned",
            "catboost": "tuned", "mlp": "tuned"},
    "reg": {"linreg": "default", "ridge": "tuned", "lasso": "tuned", "elasticnet": "tuned", "sgd_reg": "tuned",
            "random_forest": "tuned", "xgboost": "tuned", "lightgbm": "tuned", "catboost": "tuned",
            "mlp_huber": "tuned", "catboost_tweedie": "tuned", "hurdle_catboost_log_msess": "tuned"},
}
HURDLE_CLASSIFIER = "catboost"            # P(wet) of the hurdle = the frozen tuned CatBoost classifier (same seed)
# deep: (kind, task) -> (route, "tuned"/"default")  (Stage 6.3: defaults kept except TiDE classification)
DEEP = {("tsmixer", "cls"): ("bin", "default"), ("tide", "cls"): ("bin", "tuned"), ("tft", "cls"): ("bin", "default"),
        ("tsmixer", "reg"): ("log", "default"), ("tide", "reg"): ("log", "default"), ("tft", "reg"): ("mm", "default")}
BASELINES = ["majority", "persistence", "climatology", "seasonal", "seasonal+persistence"]
REFERENCES = {"cls": {"Brier": "lightgbm"}, "reg": {"abs_error": "catboost_tweedie", "sq_error": "hurdle_catboost_log_msess"}}


# ---------- data ----------
def development():
    """Feature Set B, development rows only, and the final training rows (all development rows but the last one)."""
    X, Y = E.load(FSET)
    dev = S.dev_index(X.index)
    X, Y = X.loc[dev], Y.loc[dev]
    return X, Y, dev[:-1]                       # 1-day purge before the test period (Decision 3.1)


def stochastic(task, name):
    if CLASSICAL[task][name] == "default":
        return False
    return T.MODELS[task][name][2]


def seeds_for(task, name):
    return SEEDS if stochastic(task, name) else [0]


def classical_factory(task, name, params=None):
    if CLASSICAL[task][name] == "default":
        return (Mo.LINEAR_CLS if task == "cls" else Mo.LINEAR_REG)[name][0]
    params = params if params is not None else T.load_best(name, task)["best_params"]
    return T.MODELS[task][name][1](params, task)


def deep_params(kind, task):
    import deep_tuning as DT
    return DT.load_best(kind, task)["best_params"] if DEEP[kind, task][1] == "tuned" else None


# ---------- freeze (development data only) ----------
def _hurdle_inner(params, seed, X, y, tr):
    """Inner fit of the wet-day amount model (log target): early-stopping trees and the out-of-sample smearing factor."""
    wet = tr[y[tr].to_numpy() >= R.WET_MM]
    itr, sl = E.inner_split(wet)
    z = np.log1p(y)
    inner = T.b_cat(params, "reg")(seed, z[itr])
    n_trees = E.fit_early_stopping(inner, X.loc[itr], z[itr], X.loc[sl], z[sl])
    return n_trees, float(np.mean(np.exp(z[sl] - inner.predict(X.loc[sl]))))


def freeze_classical(key):
    task, name, seed = key
    X, Y, tr = development()
    y = Y[E.TARGET[task]]
    if name.startswith("hurdle"):
        n_trees, smear = _hurdle_inner(T.load_best(name, task)["best_params"], seed, X, Y["y_reg"], tr)
        return key, {"n_trees": int(n_trees), "smear_factor": smear}
    thr, n_trees, _ = E.inner_fit(classical_factory(task, name), task, seed, X, y, tr)
    out = {}
    if task == "cls":
        out["threshold"] = float(thr)
    if not np.isnan(n_trees):
        out["n_trees"] = int(n_trees)
    return key, out


def freeze_deep(key):
    import torch
    import deep_ts as D
    torch.set_num_threads(1)
    kind, task, seed = key
    route = DEEP[kind, task][0]
    sl, info = D.run(kind, route, L, seed, save=False, params=deep_params(kind, task), inner_only=True,
                     dates=DEEP_FINAL_DATES, reseed=True, tag=f"freeze_{kind}_{task}")
    out = {"best_epoch": int(info["best_epoch"])}
    if route == "log":
        out["smear_factor"] = float(info["smear_factor"])
    if task == "cls":
        out["threshold"] = float(M.choose_threshold((sl["y_true"] >= D.WET_MM).astype(int), sl["p"], "CSI"))
    return key, out


def freeze_baselines():
    """Thresholds of the baseline probability forecasts: max CSI on their own training-row forecasts (Decision 3.5)."""
    X, Y, tr = development()
    df = X[B.SEASON_COLS].join(Y)
    rain = E.rain_today(X.index)
    fit = B.classification_baselines(df, rain, tr, tr, "y_cls", 1.0)
    return {n: (0.5 if n == "majority" else float(M.choose_threshold(Y.loc[tr, "y_cls"], p))) for n, p in fit.items()}


def _sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def freeze(workers=4, path=CONFIG):
    """Compute the frozen development quantities (parallel, one thread each) and write final_config.yaml."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    os.environ["OMP_NUM_THREADS"] = "1"
    ckeys = [(task, name, s) for task in CLASSICAL for name in CLASSICAL[task] for s in seeds_for(task, name)]
    dkeys = [(kind, task, s) for (kind, task) in DEEP for s in SEEDS]
    cache = ROOT / "models/freeze_values.json"            # raw frozen values, so a writer error never costs the fits
    if cache.exists():
        raw = json.loads(cache.read_text())
        classical = {tuple(json.loads(k)): v for k, v in raw["classical"].items()}
        deep = {tuple(json.loads(k)): v for k, v in raw["deep"].items()}
    else:
        with ProcessPoolExecutor(workers) as ex:
            deep = dict(ex.map(freeze_deep, dkeys))
            classical = dict(ex.map(freeze_classical, ckeys))
        cache.write_text(json.dumps({"classical": {json.dumps(list(k)): v for k, v in classical.items()},
                                     "deep": {json.dumps(list(k)): v for k, v in deep.items()}}, indent=1))
    write_config(classical, deep, freeze_baselines(), path)
    return load_config(path)


def write_config(classical, deep, base_thr, path=CONFIG):
    X, Y, tr = development()
    itr, sl = E.inner_split(tr)
    td = lambda idx: [str(S.target_dates(idx).min().date()), str(S.target_dates(idx).max().date())]
    models = {"baselines": {"cls": {n: {"threshold": base_thr[n]} for n in BASELINES}, "reg": {n: {} for n in BASELINES}},
              "classical": {}, "deep": {}}
    for task, names in CLASSICAL.items():
        models["classical"][task] = {}
        for name, cfg in names.items():
            entry = {"config": cfg, "seeds": seeds_for(task, name),
                     "frozen": {int(s): classical[(task, name, s)] for s in seeds_for(task, name)}}
            if cfg == "tuned":
                b = T.load_best(name, task)
                entry |= {"params": b["best_params"], "tuning_objective": b["objective"],
                          "source": f"models/optuna/best_params_{name}__{task}.json"}
            else:
                entry["params"] = "Stage-4 default factory (models.py), no hyper-parameters to tune"
            if name.startswith("hurdle"):
                entry["p_wet_from"] = f"classical.cls.{HURDLE_CLASSIFIER} (same seed)"
            models["classical"][task][name] = entry
    for (kind, task), (route, cfg) in DEEP.items():
        models["deep"].setdefault(task, {})[kind] = {
            "config": cfg, "route": route, "lookback": L, "params": deep_params(kind, task) or "Stage-5 default (deep_ts.DEFAULTS)",
            "seeds": SEEDS, "frozen": {s: deep[(kind, task, s)] for s in SEEDS},
            "point_forecast": {"bin": "Bernoulli probability p", "log": "exp(z) * smear_factor - 1 (Duan)",
                               "mm": "mean of the sorted quantile function (pred_mean); median and quantiles also stored"}[route]}
    cfg = {
        "freeze": {"date": str(pd.Timestamp.now().date()), "pre_test_commit": _git("rev-parse", "HEAD"),
                   "note": "pre_test_commit = HEAD when this file was generated (before the freeze commit). The freeze "
                           "commit adds this file and src/final_eval.py; final_eval.run records HEAD at run time and refuses "
                           "to run if final_config.yaml or src/ have uncommitted changes.",
                   "data_sha256": {f: _sha(f) for f in DATA_FILES},
                   "versions": {"python": "3.13", "pandas": pd.__version__, "numpy": np.__version__,
                                **{m: str(__import__(m).__version__) for m in ("sklearn", "xgboost", "lightgbm", "catboost",
                                                                          "torch", "darts", "optuna")}},
                   "threads": {"classical": T.THREADS, "random_forest": T.RF_THREADS, "torch": 1}},
        "data": {
            "feature_set": FSET, "n_features": int(X.shape[1]), "features": list(X.columns),
            "feature_set_A": "not evaluated on the test set (Stage-6 ablation: A = B within noise); B is primary",
            "targets": {"classification": "y_cls = PRECTOTCORR(t+1) >= 1.0 mm (primary; 0.1/2.5 mm were development-only "
                                          "sensitivity analyses)", "regression": "y_reg = PRECTOTCORR(t+1) in mm"},
            "development": {"feature_dates": [str(X.index.min().date()), str(X.index.max().date())],
                            "target_dates": td(X.index)},
            "final_training_rows": {"target_dates": td(tr), "rule": "all development rows except the last one (1-day purge, "
                                                                    "Decision 3.1)", "n_rows": int(len(tr))},
            "inner_split_for_frozen_values": {"inner_train_target_dates": td(itr), "slice_target_dates": td(sl),
                                              "rule": "last 3 target years of the final training rows, 1-day purge "
                                                      "(experiment.inner_split); used for thresholds, early stopping, "
                                                      "smearing and deep best epochs"},
            "test": {"target_dates": [str(TEST_FIRST.date()), str(TEST_LAST.date())],
                     "feature_dates": [str((TEST_FIRST - pd.Timedelta(days=1)).date()),
                                       str((TEST_LAST - pd.Timedelta(days=1)).date())],
                     "expected_n_rows": int((TEST_LAST - TEST_FIRST).days + 1), "status": "LOCKED until Prompt 7.2"},
            "deep_dates": {k: str(v.date()) for k, v in DEEP_FINAL_DATES.items()}},
        "seeds": {"stochastic_models": SEEDS, "deterministic_models": [0],
                  "deep_reseed_global_rng": True, "sampler_seed_optuna": T.SAMPLER_SEED},
        "threshold_rule": "probability threshold maximising CSI on the inner-slice forecasts (2017-2019) of the inner model "
                          "(grid 0.01..0.99, step 0.01); baselines: max CSI on their training-row forecasts; majority 0.5; "
                          "applied unchanged to the test forecasts (values under models.*.frozen)",
        "models": models,
        "selection_and_statistics": {
            "no_selection_on_test": "every frozen model is evaluated and reported; no model, threshold, route or setting is "
                                    "chosen or changed after the test scores are seen",
            "primary_metrics": {"cls": ["BSS (vs climatology of the training rows)"],
                                "reg": ["MAE", "MSESS (vs climatology)"],
                                "secondary": ["ROC-AUC", "PR-AUC", "CSI at the frozen threshold", "ECE", "RMSE", "bias",
                                              "wet-day MAE", "heavy-rain recall/precision >= 20/50 mm",
                                              "skill vs seasonal+persistence", "per-year and monsoon scores"]},
            "headline_models_fixed_in_advance": {
                "rule": "best 5-fold development CV score among the frozen classical models on the primary metric",
                "cls": "lightgbm (dev BSS 0.2650)", "reg_MAE": "catboost_tweedie (dev MAE 6.897)",
                "reg_MSESS": "hurdle_catboost_log_msess (dev MSESS 0.275; mean-type forecast)"},
            "forecast_used_in_tests": "seed-mean forecast of each model (probability / mm); per-seed scores reported as "
                                      "mean +- std",
            "comparisons": {"family_A": "every frozen model vs seasonal+persistence",
                            "family_B": "every frozen model vs the headline classical reference: cls Brier vs lightgbm; "
                                        "reg absolute error vs catboost_tweedie; reg squared error vs "
                                        "hurdle_catboost_log_msess"},
            "losses": {"cls": "Brier (squared probability error)", "reg": ["absolute error (MAE)", "squared error (MSE)"]},
            "diebold_mariano": "two-sided DM test on the daily loss differential; HAC variance with Bartlett kernel, "
                               "bandwidth floor(4 (n/100)^(2/9)) (Newey-West 1994); Harvey-Leybourne-Newbold small-sample "
                               "correction (h = 1), Student-t with n-1 df (final_eval.dm_test)",
            "multiplicity": "Holm step-down within each family x task x loss at alpha = 0.05 (final_eval.holm)",
            "bootstrap": "paired moving-calendar-month block bootstrap of the target date (whole year-month blocks), "
                         "1,000 resamples, seed 42, 95 % percentile CIs for dBSS, dMAE, dMSESS (metrics.block_bootstrap)",
            "rule_11_triggers": "BSS > 0.45, monsoon ROC-AUC > 0.90 (1 mm target), MSESS > 0.60 -> investigate before reporting",
            "secondary_robustness": "the frozen PRECTOTCORR forecasts are also scored against IMERG_PRECTOT observed rain "
                                    "on the test days (no refit; tests the reanalysis self-consistency caveat)"},
    }
    path.write_text("# Stage 7.1 freeze -- generated by src/final_eval.freeze() from development data only. DO NOT EDIT.\n"
                    + yaml.safe_dump(cfg, sort_keys=False, width=120, allow_unicode=True))


def load_config(path=CONFIG):
    return yaml.safe_load(Path(path).read_text())


# ---------- integrity guard ----------
def check_integrity(cfg):
    """Refuse a final run unless the frozen files are committed unchanged and the data are the frozen data."""
    dirty = _git("status", "--porcelain", "--", "final_config.yaml", "src")
    if dirty:
        raise RuntimeError(f"uncommitted changes in frozen files:\n{dirty}")
    for f, h in cfg["freeze"]["data_sha256"].items():
        if _sha(f) != h:
            raise RuntimeError(f"{f} differs from the frozen data (sha256)")
    return _git("rev-parse", "HEAD")


# ---------- prediction primitives (also used by the unit tests on development-only pseudo-test windows) ----------
def predict_classical(task, name, seed, X, y, tr, te, frozen, p_wet=None):
    """Refit one frozen classical model on `tr` with its frozen values and predict `te` (labels of `te` unused)."""
    if name.startswith("hurdle"):
        params = T.load_best(name, task)["best_params"]
        wet = tr[y[tr].to_numpy() >= R.WET_MM]
        z = np.log1p(y)
        est = T.b_cat(params, task)(seed, z[wet])
        E.set_trees(est, int(frozen["n_trees"]))
        est.fit(X.loc[wet], z[wet])
        amount = np.clip(np.exp(est.predict(X.loc[te])) * frozen["smear_factor"] - 1, R.WET_MM, None)
        return pd.DataFrame({"pred": np.asarray(p_wet) * amount, "p_wet": p_wet, "amount_if_wet": amount}, index=te)
    pred = E.refit_predict(classical_factory(task, name), task, seed, X, y, tr, te, frozen.get("n_trees", np.nan))
    out = pd.DataFrame({"pred": pred}, index=te)
    if task == "cls":
        out["threshold"] = frozen["threshold"]
    return out


def predict_deep(kind, task, seed, frozen, df, fut, dates):
    import torch
    import deep_ts as D
    torch.set_num_threads(1)
    route = DEEP[kind, task][0]
    out, info = D.run(kind, route, L, seed, save=False, params=deep_params(kind, task), variant={"df": df, "fut": fut},
                      dates=dates, refit_only={**frozen, "reseed": True}, log_val=False, tag=f"final_{kind}_{task}")
    if task == "cls":
        out["threshold"] = frozen["threshold"]
    return out.drop(columns="y_true"), info


# ---------- Stage 7.2 entry point ----------
def _save(df, name, meta):
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TEST_DIR / f"{name}.parquet")
    meta[name] = {"rows": int(len(df)), "first_feature_date": str(df.index.min().date()),
                  "last_feature_date": str(df.index.max().date())}


def run(confirm_final=False, include_deep=True):
    """Refit every frozen model on the development rows and predict the test years once. Stage 7.2 only."""
    if confirm_final is not True:
        raise PermissionError("Final evaluation is locked. Call run(confirm_final=True) only after the researcher's "
                              "explicit Stage-7.2 confirmation.")
    cfg = load_config()
    commit = check_integrity(cfg)
    t0 = time.time()
    X_all, Y_all = E.load(FSET)
    te = S.test_index(X_all.index, confirm_final=True)
    X, Y, tr = development()
    X = pd.concat([X, X_all.loc[te]])
    rain = E.rain_today(X.index)
    meta = {}
    # baselines (fitted on the training rows)
    df = X[B.SEASON_COLS].join(Y)
    for task in ("cls", "reg"):
        fc = B.classification_baselines(df, rain, tr, te, "y_cls", 1.0) if task == "cls" else \
            B.regression_baselines(df, rain, tr, te)
        for n, p in fc.items():
            out = pd.DataFrame({"pred": p}, index=te)
            if task == "cls":
                out["threshold"] = cfg["models"]["baselines"]["cls"][n]["threshold"]
            _save(out, f"baseline:{n}__{task}__seed0", meta)
    # classical models (the hurdle needs the CatBoost classifier of the same seed, so classification runs first)
    for task in ("cls", "reg"):
        y = Y[E.TARGET[task]]
        for name, entry in cfg["models"]["classical"][task].items():
            for seed in entry["seeds"]:
                p_wet = pd.read_parquet(TEST_DIR / f"{HURDLE_CLASSIFIER}__cls__seed{seed}.parquet")["pred"].to_numpy() \
                    if name.startswith("hurdle") else None
                out = predict_classical(task, name, seed, X, Y["y_reg"] if name.startswith("hurdle") else y, tr, te,
                                        entry["frozen"][seed], p_wet)
                _save(out, f"{name}__{task}__seed{seed}", meta)
    # deep models: series up to 2025-12-31, scalers fitted on rows <= 2019-12-30, refit for the frozen epochs
    if include_deep:
        import deep_ts as D
        clean = pd.read_parquet(E.PROC / "clean_daily.parquet").loc[:TEST_LAST, [D.TARGET] + D.PAST]
        doy = clean.index.dayofyear.to_numpy()
        fut = pd.DataFrame({f"doy_{f}{k}": fn(2 * np.pi * k * doy / 365.25) for k in (1, 2)
                            for f, fn in (("sin", np.sin), ("cos", np.cos))}, index=clean.index)[D.FUTURE]
        for task, kinds in cfg["models"]["deep"].items():
            for kind, entry in kinds.items():
                for seed in entry["seeds"]:
                    out, _ = predict_deep(kind, task, seed, entry["frozen"][seed], clean, fut, DEEP_FINAL_DATES)
                    assert out.index.equals(te), "deep forecasts are not aligned with the test rows"
                    _save(out, f"{kind}__{task}__seed{seed}", meta)
    manifest = {"run_at": str(pd.Timestamp.now()), "commit": commit, "config_sha256": _sha("final_config.yaml"),
                "seconds": time.time() - t0, "files": meta}
    (TEST_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ---------- pre-registered statistics ----------
def dm_test(loss_a, loss_b, h=1):
    """Diebold-Mariano test of equal expected loss (d = loss_a - loss_b). HAC variance with a Bartlett kernel and
    Newey-West (1994) bandwidth floor(4 (n/100)^(2/9)); Harvey-Leybourne-Newbold correction; two-sided Student-t, n-1 df.
    Returns (mean differential, DM statistic, p-value). Negative mean = model a has lower loss."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    dc = d - d.mean()
    q = int(np.floor(4 * (n / 100) ** (2 / 9)))
    var = dc @ dc / n
    for k in range(1, q + 1):
        var += 2 * (1 - k / (q + 1)) * (dc[k:] @ dc[:-k]) / n
    if var <= 0:
        return float(d.mean()), 0.0, 1.0
    dm = d.mean() / np.sqrt(var / n)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(d.mean()), float(dm), float(2 * stats.t.sf(abs(dm), df=n - 1))


def holm(pvalues):
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Stage 7: freeze (development only) or the final test-set run")
    ap.add_argument("--freeze", action="store_true", help="compute frozen values and write final_config.yaml")
    ap.add_argument("--confirm-final", action="store_true", help="Stage 7.2 only: refit and predict the test years")
    a = ap.parse_args()
    if a.freeze:
        freeze()
    elif a.confirm_final:
        print(run(confirm_final=True))
    else:
        ap.print_help()
