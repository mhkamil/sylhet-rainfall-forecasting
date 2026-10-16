"""Stage 6.1 hyper-parameter tuning with Optuna (TPE sampler + median pruning across the 5 expanding CV folds).

tune(name, task, n_trials) -> (best_params, study)
  * data: Feature Set B, development rows only (targets <= 2019-12-31); the locked test years are dropped before anything
    else happens, so a trial cannot see them (tests/test_tuning.py proves it);
  * each trial evaluates one parameter set with seed 0 on the 5 folds of src/splits.py through experiment.fit_fold, i.e.
    the identical Stage-4 protocol (inner slice -> threshold and early stopping; refit on the training fold; preprocessing
    inside each Pipeline). Validation labels are used only to score a finished fold;
  * after each fold the running mean of the objective is reported, so MedianPruner can stop unpromising trials early;
  * objective (Decision 4.19): classification maximise BSS, regression minimise MAE; other metrics in trial user_attrs;
  * each study lives in its own SQLite file models/optuna/<model>__<task>__<set>.db (several studies can run in
    parallel processes without locking one file); best parameters in models/optuna/best_params_*.json.
rerun_best(...) refits the best parameters with seeds 0-4 through the normal runners (rows '<name>_tuned', OOF files in
models/oof/). Scores go to one part file per study (models/optuna/cv_parts/) and merge_results() copies them into
reports/tables/cv_results.csv from a single process, so parallel jobs never race on that table.
Run all studies of Stage 6.2 in parallel:  python src/tuning.py --workers 4
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

import experiment as E
import models as Mo
import reg_variants as R
import splits as S

OPTUNA_DIR = E.ROOT / "models/optuna"
PARTS_DIR = OPTUNA_DIR / "cv_parts"
FSET = "B"
TUNE_SEED = 0                 # model seed inside every trial
SAMPLER_SEED = 42
THREADS = 1                   # one thread per model; parallelism comes from running several studies at once
RF_THREADS = 4                # random forests run alone after the pool (sklearn RF results do not depend on n_jobs)
MAX_TREES = 3000              # booster cap; early stopping on the slice decides the actual number
OBJECTIVE = {"cls": ("BSS", "maximize"), "reg": ("MAE", "minimize")}
# Part-2 follow-up (researcher, 2026-10-16): the hurdle's role is the mean-preserving forecast -> a second study maximises MSESS
OBJECTIVE_OVERRIDE = {("reg", "hurdle_catboost_log_msess"): ("MSESS", "maximize")}


def objective_of(task, name):
    return OBJECTIVE_OVERRIDE.get((task, name), OBJECTIVE[task])
LOGGED = {"cls": ["BSS", "PR_AUC", "ROC_AUC", "Brier", "CSI", "log_loss", "ECE", "BSS_vs_seas_pers"],
          "reg": ["MAE", "MSESS", "RMSE", "bias", "MAE_wet", "MSESS_vs_seas_pers"]}
# P(wet) for the hurdle: trials use the Stage-4 default CatBoost classifier (fixed, so the amount-model study does not wait
# for the classifier study); the 5-seed re-run combines the TUNED classifier with the tuned amount model.
HURDLE_P_TRIALS, HURDLE_P_FINAL = "catboost", "catboost_tuned"


# ---------- search spaces (playbook ranges, Stage 6.2) ----------
def _boost_common(t):
    return {"learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True)}


def sp_C(t):
    return {"C": t.suggest_float("C", 1e-4, 1e2, log=True)}


def sp_C_l1(t):
    return {**sp_C(t), "l1_ratio": t.suggest_float("l1_ratio", 0.1, 0.9)}


def sp_alpha(t):
    return {"alpha": t.suggest_float("alpha", 1e-4, 1e2, log=True)}


def sp_alpha_wide(t):          # Ridge only: the 1e-4..1e2 optimum sat at the upper edge (Part 1) -> range widened to 1e5
    return {"alpha": t.suggest_float("alpha", 1e-4, 1e5, log=True)}


def sp_alpha_l1(t):
    return {**sp_alpha(t), "l1_ratio": t.suggest_float("l1_ratio", 0.1, 0.9)}


def sp_sgd(t):
    p = {"alpha": t.suggest_float("alpha", 1e-6, 1e-1, log=True),
         "penalty": t.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
         "learning_rate": t.suggest_categorical("learning_rate", ["optimal", "invscaling", "adaptive", "constant"])}
    if p["penalty"] == "elasticnet":
        p["l1_ratio"] = t.suggest_float("l1_ratio", 0.1, 0.9)
    if p["learning_rate"] != "optimal":                     # eta0 is ignored by the 'optimal' schedule
        p["eta0"] = t.suggest_float("eta0", 1e-4, 1e-1, log=True)
    return p


def sp_rf(t):
    p = {"n_estimators": t.suggest_int("n_estimators", 200, 800, step=50),
         "max_depth": t.suggest_int("max_depth", 4, 20),
         "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 50, log=True),
         "max_features": t.suggest_categorical("max_features", ["sqrt", "fraction"])}
    if p["max_features"] == "fraction":
        p["max_features"] = t.suggest_float("max_features_fraction", 0.3, 1.0)
    return p


def sp_xgb(t):
    return {**_boost_common(t), "max_depth": t.suggest_int("max_depth", 3, 8),
            "min_child_weight": t.suggest_float("min_child_weight", 1, 20, log=True),
            "subsample": t.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 0.01, 10, log=True)}


def sp_lgbm(t):
    return {**_boost_common(t), "num_leaves": t.suggest_int("num_leaves", 15, 127, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 10, 100, log=True),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}


def sp_cat(t):
    return {"depth": t.suggest_int("depth", 4, 8), "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 10, log=True)}


def sp_cat_tweedie(t):
    return {**sp_cat(t), "variance_power": t.suggest_float("variance_power", 1.1, 1.9)}


def sp_mlp(t):
    return {"n_layers": t.suggest_int("n_layers", 1, 3), "units": t.suggest_int("units", 32, 128, log=True),
            "dropout": t.suggest_float("dropout", 0.0, 0.4),
            "weight_decay": t.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "lr": t.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": t.suggest_categorical("batch_size", [64, 128, 256, 512])}


# ---------- builders: (params, task) -> factory(seed, y_train) -> unfitted estimator ----------
def _lin(make):
    return lambda p, task: (lambda s, y: Mo.make_pipeline(Mo.linear_pre(), make(p, s)))


def b_logreg(l1_ratio=None, saga=True):
    from sklearn.linear_model import LogisticRegression
    return _lin(lambda p, s: LogisticRegression(C=p["C"], l1_ratio=p.get("l1_ratio", l1_ratio), class_weight="balanced",
                                                max_iter=5000, **({"solver": "saga", "random_state": s} if saga else {})))


def b_linreg(cls):
    """Ridge: closed form. Lasso/ElasticNet: coordinate descent on the precomputed 125 x 125 Gram matrix (same objective as
    Stage 4; without it a trial with alpha < 0.01 took > 40 min and still stopped unconverged at 20,000 passes)."""
    return _lin(lambda p, s: cls(**p, max_iter=200000, precompute=True) if cls.__name__ != "Ridge" else cls(**p))


def b_sgd(p, task):
    from sklearn.linear_model import SGDClassifier, SGDRegressor
    kw = dict(p)
    if task == "cls":
        return lambda s, y: Mo.make_pipeline(Mo.linear_pre(), SGDClassifier(loss="log_loss", class_weight="balanced",
                                                                            random_state=s, **kw))
    return lambda s, y: Mo.make_pipeline(Mo.linear_pre(), SGDRegressor(random_state=s, **kw))


def b_rf(p, task):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    kw = {k: v for k, v in p.items() if k != "max_features_fraction"}
    if task == "cls":
        return lambda s, y: RandomForestClassifier(**kw, class_weight="balanced", n_jobs=RF_THREADS, random_state=s)
    return lambda s, y: RandomForestRegressor(**kw, n_jobs=RF_THREADS, random_state=s)


def b_xgb(p, task):
    import xgboost as xgb
    kw = dict(n_estimators=MAX_TREES, tree_method="hist", n_jobs=THREADS, **p)
    if task == "cls":
        return lambda s, y: xgb.XGBClassifier(**kw, eval_metric="logloss", scale_pos_weight=Mo._pos_weight(y), random_state=s)
    return lambda s, y: xgb.XGBRegressor(**kw, eval_metric="rmse", random_state=s)


def b_lgbm(p, task):
    import lightgbm as lgb
    kw = dict(n_estimators=MAX_TREES, subsample_freq=1, n_jobs=THREADS, verbose=-1, **p)
    if task == "cls":
        return lambda s, y: lgb.LGBMClassifier(**kw, class_weight="balanced", random_state=s)
    return lambda s, y: lgb.LGBMRegressor(**kw, random_state=s)


def b_cat(p, task, loss=None):
    import catboost as cb
    kw = dict(iterations=MAX_TREES, thread_count=THREADS, verbose=0, allow_writing_files=False,
              **{k: v for k, v in p.items() if k != "variance_power"})
    if task == "cls":
        return lambda s, y: cb.CatBoostClassifier(**kw, auto_class_weights="Balanced", random_seed=s)
    loss = loss or ("RMSE" if "variance_power" not in p else f"Tweedie:variance_power={p['variance_power']}")
    return lambda s, y: cb.CatBoostRegressor(**kw, loss_function=loss, random_seed=s)


def b_mlp(p, task):
    from mlp import TorchMLP
    kw = {k: v for k, v in p.items() if k not in ("n_layers", "units")}
    hidden = (p["units"],) * p["n_layers"]
    return lambda s, y: TorchMLP(task=task, loss="bce" if task == "cls" else "huber", hidden=hidden, seed=s,
                                 threads=THREADS, **kw)


from sklearn.linear_model import ElasticNet, Lasso, Ridge  # noqa: E402

# name -> (space, builder, stochastic?, kind). Names equal the Stage-4 default models they are compared with.
# kind "fit_fold" = normal runner; "hurdle" = P(wet) of the tuned classifier x tuned wet-day amount model (log + smearing).
MODELS = {
    "cls": {
        "logreg_l2": (sp_C, b_logreg(l1_ratio=0.0, saga=False), False, "fit_fold"),
        "logreg_l1": (sp_C, b_logreg(l1_ratio=1.0), True, "fit_fold"),
        "logreg_enet": (sp_C_l1, b_logreg(), True, "fit_fold"),
        "sgd_logloss": (sp_sgd, b_sgd, True, "fit_fold"),
        "random_forest": (sp_rf, b_rf, True, "fit_fold"),
        "xgboost": (sp_xgb, b_xgb, True, "fit_fold"),
        "lightgbm": (sp_lgbm, b_lgbm, True, "fit_fold"),
        "catboost": (sp_cat, b_cat, True, "fit_fold"),
        "mlp": (sp_mlp, b_mlp, True, "fit_fold"),
    },
    "reg": {
        "ridge": (sp_alpha_wide, b_linreg(Ridge), False, "fit_fold"),
        "lasso": (sp_alpha, b_linreg(Lasso), False, "fit_fold"),
        "elasticnet": (sp_alpha_l1, b_linreg(ElasticNet), False, "fit_fold"),
        "sgd_reg": (sp_sgd, b_sgd, True, "fit_fold"),
        "random_forest": (sp_rf, b_rf, True, "fit_fold"),
        "xgboost": (sp_xgb, b_xgb, True, "fit_fold"),
        "lightgbm": (sp_lgbm, b_lgbm, True, "fit_fold"),
        "catboost": (sp_cat, b_cat, True, "fit_fold"),
        "mlp_huber": (sp_mlp, b_mlp, True, "fit_fold"),
        "catboost_tweedie": (sp_cat_tweedie, b_cat, True, "fit_fold"),
        "hurdle_catboost_log": (sp_cat, b_cat, True, "hurdle"),
        "hurdle_catboost_log_msess": (sp_cat, b_cat, True, "hurdle"),
    },
}

# Stage-6.2 trial budgets (fewer for slow models).
BUDGET = {("cls", "logreg_l2"): 30, ("cls", "logreg_l1"): 20, ("cls", "logreg_enet"): 25, ("cls", "sgd_logloss"): 100,
          ("cls", "random_forest"): 15, ("cls", "xgboost"): 50, ("cls", "lightgbm"): 50, ("cls", "catboost"): 40,
          ("cls", "mlp"): 30,
          ("reg", "ridge"): 50, ("reg", "lasso"): 50, ("reg", "elasticnet"): 60, ("reg", "sgd_reg"): 100,
          ("reg", "random_forest"): 15, ("reg", "xgboost"): 50, ("reg", "lightgbm"): 50, ("reg", "catboost"): 40,
          ("reg", "mlp_huber"): 30, ("reg", "catboost_tweedie"): 40, ("reg", "hurdle_catboost_log"): 30,
          ("reg", "hurdle_catboost_log_msess"): 30}


# ---------- data and one evaluation ----------
def prepare(task, fset=FSET):
    """Development rows only (the locked test years are removed first), the 5 folds and their references."""
    X, Y = E.load(fset)
    dev = S.dev_index(X.index)
    X, Y = X.loc[dev], Y.loc[dev]
    rain, folds = E.rain_today(X.index), S.cv_folds(X.index)
    refs = [E.references(task, X, Y, rain, tr, va) for tr, va in folds]
    return {"task": task, "fset": fset, "X": X, "Y": Y, "y": Y[E.TARGET[task]], "folds": folds, "refs": refs}


def fold_predict(name, params, data, k, seed=TUNE_SEED):
    """Forecasts for validation fold k. Returns (pred, threshold, n_trees). Validation labels are never read here."""
    task, X, y = data["task"], data["X"], data["y"]
    _, build, _, kind = MODELS[task][name]
    tr, va = data["folds"][k]
    if kind == "hurdle":
        wet = tr[y[tr].to_numpy() >= R.WET_MM]
        p = R.hurdle_probs(data["fset"], seed, classifier=HURDLE_P_TRIALS).loc[va].to_numpy()
        return p * R.hurdle_amount(build(params, task), seed, X, y, wet, va, "log"), np.nan, np.nan
    pred, thr, n_trees, _ = E.fit_fold(build(params, task), task, seed, X, y, tr, va)
    return pred, thr, n_trees


def evaluate(name, params, data, trial=None, seed=TUNE_SEED):
    """Score `params` fold by fold; with a trial, report the running mean of the objective so the pruner can act."""
    task = data["task"]
    metric = objective_of(task, name)[0]
    rows = []
    for k, (tr, va) in enumerate(data["folds"]):
        t0 = time.time()
        pred, thr, n_trees = fold_predict(name, params, data, k, seed)
        if not np.all(np.isfinite(pred)):                       # e.g. a diverged SGD run
            raise optuna.TrialPruned(f"non-finite forecasts in fold {k + 1}")
        m = E.score(task, data["y"][va], pred, thr, data["refs"][k], va)
        rows.append({"fold": k + 1, **{x: m[x] for x in LOGGED[task]}, "threshold": thr, "n_trees": n_trees,
                     "seconds": time.time() - t0})
        if trial is not None:
            trial.report(float(np.mean([r[metric] for r in rows])), k)
            if k < len(data["folds"]) - 1 and trial.should_prune():     # pruning after the last fold saves nothing
                trial.set_user_attr("folds_done", k + 1)
                raise optuna.TrialPruned()
    return pd.DataFrame(rows)


def make_objective(name, data):
    space = MODELS[data["task"]][name][0]

    def objective(trial):
        params = space(trial)
        folds = evaluate(name, params, data, trial)
        for col in LOGGED[data["task"]]:
            trial.set_user_attr(col, float(folds[col].mean()))
        trial.set_user_attr("folds", folds.round(5).to_dict("list"))
        trial.set_user_attr("params_resolved", params)
        return float(folds[objective_of(data["task"], name)[0]].mean())
    return objective


def study_name(name, task, fset=FSET):
    return f"{name}__{task}__{fset}"


def storage_url(name, task, fset=FSET):
    return f"sqlite:///{OPTUNA_DIR / (study_name(name, task, fset) + '.db')}"


def tune(name, task, n_trials, fset=FSET, storage=None, data=None):
    """Run (or resume) the study until it has n_trials finished trials; return (best_params, study).
    best_params are the resolved model parameters (conditional choices merged), ready for the builder."""
    Path(OPTUNA_DIR).mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=study_name(name, task, fset), storage=storage or storage_url(name, task, fset),
                                load_if_exists=True,
                                direction=objective_of(task, name)[1], sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    for t in study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.RUNNING]):
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)       # left over from an interrupted run
    done = len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]))
    if done < n_trials:
        data = data or prepare(task, fset)
        t0 = time.time()
        study.optimize(make_objective(name, data), n_trials=n_trials - done, gc_after_trial=True)
        study.set_user_attr(f"seconds_run_{len(study.trials)}", time.time() - t0)
    best = study.best_trial.user_attrs["params_resolved"]
    if storage is None:                               # custom storages (unit tests) leave models/optuna untouched
        save_best(name, task, study, fset)
    return best, study


def save_best(name, task, study, fset=FSET):
    states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
    b = study.best_trial
    out = {"model": name, "task": task, "feature_set": fset, "objective": objective_of(task, name)[0],
           "direction": objective_of(task, name)[1], "best_value": b.value, "best_trial": b.number,
           "best_params": b.user_attrs["params_resolved"], "logged_metrics": {k: b.user_attrs[k] for k in LOGGED[task]},
           "trial_states": states, "tuning_seconds": sum(v for k, v in study.user_attrs.items() if k.startswith("seconds"))}
    path = OPTUNA_DIR / f"best_params_{name}__{task}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    return out


def load_best(name, task):
    return json.loads((OPTUNA_DIR / f"best_params_{name}__{task}.json").read_text())


def load_study(name, task, fset=FSET, storage=None):
    return optuna.load_study(study_name=study_name(name, task, fset), storage=storage or storage_url(name, task, fset))


# ---------- re-run with 5 seeds through the normal runners ----------
def rerun_best(name, task, params=None, fset=FSET, skip_existing=True):
    """Refit the best parameters with seeds 0-4 (seed 0 only for deterministic models) on the 5 CV folds.
    Rows are stored as '<name>_tuned' in this study's part file (see merge_results); returns the tidy scores."""
    params = params or load_best(name, task)["best_params"]
    _, build, stochastic, kind = MODELS[task][name]
    seeds = Mo.SEEDS if stochastic else [0]
    tuned = f"{name}_tuned"
    part = PARTS_DIR / f"{study_name(name, task, fset)}.csv"
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for seed in seeds:
        oof = E.OOF_DIR / f"{tuned}__{task}__{fset}__seed{seed}.parquet"
        if skip_existing and oof.exists():
            continue
        if kind == "hurdle":
            out.append(R.run_hurdle("catboost", build(params, task), fset, seed, classifier=HURDLE_P_FINAL,
                                    kinds=("log",), suffix=tuned[len("hurdle_catboost_log"):],
                                    results_path=part))
        else:
            out.append(E.run_cv(tuned, build(params, task), task, fset, seed, results_path=part)[0])
    return pd.concat(out) if out else None


def merge_results(results_path=E.RESULTS):
    """Copy every part file into the CV results table (rows with the same model/task/set/seed are replaced)."""
    parts = sorted(PARTS_DIR.glob("*.csv"))
    for f in parts:
        E.append_results(pd.read_csv(f), results_path)
    return len(parts)


# ---------- batch driver for Stage 6.2 ----------
def _job(key, rerun=True):
    import warnings
    warnings.filterwarnings("ignore")               # convergence warnings of weak trials would flood the log
    task, name = key
    t0 = time.time()
    tune(name, task, BUDGET[key])
    if rerun:
        rerun_best(name, task)
    return key, time.time() - t0


def run_all(workers=4):
    """All Stage-6.2 studies: the single-threaded ones in parallel processes (slowest first), then the two random forests
    one after the other with RF_THREADS threads each, then the hurdle re-run (it needs the tuned CatBoost classifier)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(THREADS)
    rf = [("reg", "random_forest"), ("cls", "random_forest")]
    hurdle = ("reg", "hurdle_catboost_log")
    slow = [("cls", "catboost"), ("reg", "catboost_tweedie"), ("reg", "catboost"), hurdle, ("cls", "logreg_l1"),
            ("cls", "logreg_enet"), ("cls", "logreg_l2"), ("cls", "xgboost")]
    order = slow + [k for k in BUDGET if k not in slow + rf]
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, k, k != hurdle) for k in order]
        for f in as_completed(futs):
            key, sec = f.result()
            print(f"{key}: {sec / 60:.1f} min", flush=True)
    for key in rf:
        print(f"{_job(key)[0]}: done", flush=True)
    rerun_best(hurdle[1], hurdle[0])
    print("merged part files:", merge_results(), flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    run_all(ap.parse_args().workers)
