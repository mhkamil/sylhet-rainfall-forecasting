"""Stage 6.3: Optuna tuning of TSMixer, TiDE and TFT on the DL split (lookback L = 14 fixed, Decision 5.8).

Selection criterion = out-of-sample forecasts of the INNER model (trained on targets <= 2011-12-30) for the slice
2012-01-01..2014-12-30, i.e. the same slice that already drives early stopping. The validation years 2015-2019 are never
used by a trial, so the 5-seed re-runs of the selected settings give validation scores WITHOUT hyper-parameter selection
optimism (the slice score itself is optimistic: it picked the epoch and the setting).
Routes (Stage 5): classification = Bernoulli likelihood (option 3, raw probability); regression = log target + Duan
smearing for TSMixer and TiDE, quantile regression (point = quantile mean) for TFT.
Objectives: classification max BSS on the slice (climatology of the inner-training years as reference); regression min MAE.
Pruning: MedianPruner on the per-epoch slice validation loss (sign-flipped for the BSS study).
Trial 0 of every study is the Stage-5 default configuration (enqueued), so the gain over the default is visible.
"""
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl

import baselines as B
import deep_ts as D
import experiment as E
import metrics as M
import splits as S

OPTUNA_DIR = E.ROOT / "models/optuna"
TUNED_DIR = D.DEEP_DIR / "tuned"
RESULTS = E.ROOT / "reports/tables/deep_tuned_results.csv"
L = 14
SAMPLER_SEED, TUNE_SEED = 42, 0
ROUTE = {"cls": {"tsmixer": "bin", "tide": "bin", "tft": "bin"}, "reg": {"tsmixer": "log", "tide": "log", "tft": "mm"}}
POINT = {"log": "pred", "mm": "pred_mean"}            # smeared mm forecast / mean of the TFT quantile function
BUDGET = {"tsmixer": 20, "tide": 20, "tft": 12}
OBJECTIVE = {"cls": ("BSS", "maximize"), "reg": ("MAE", "minimize")}


def space(trial, kind):
    hidden = [8, 16, 32, 64] if kind == "tft" else [16, 32, 64, 128]
    return {"hidden_size": trial.suggest_categorical("hidden_size", hidden),
            "n_layers": trial.suggest_int("n_layers", 1, 2 if kind == "tft" else 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512])}


def default_params(kind):
    return {**D.COMMON_DEFAULTS, **D.DEFAULTS[kind]}


def slice_climatology(slice_index):
    """P(rain >= 1 mm) climatology fitted on the inner-training target dates (<= 2011-12-30), for the slice targets."""
    rain = pd.read_parquet(E.PROC / "clean_daily.parquet")[D.TARGET]
    tr = rain.loc[:D.INNER_LAST]
    clim = B.doy_climatology(tr.index, (tr >= D.WET_MM).astype(float))
    return clim[(S.target_dates(slice_index)).dayofyear - 1]


def slice_score(task, transform, sl):
    """Objective value and logged metrics of one inner model's slice forecasts."""
    if task == "cls":
        y = (sl["y_true"] >= D.WET_MM).astype(int).to_numpy()
        m = M.classification_metrics(y, sl["p"].to_numpy(), threshold=0.5, p_ref=slice_climatology(sl.index))
        return m["BSS"], {k: m[k] for k in ("BSS", "Brier", "ROC_AUC", "PR_AUC", "log_loss", "ECE")}
    m = M.regression_metrics(sl["y_true"].to_numpy(), sl[POINT[transform]].to_numpy())
    return m["MAE"], {k: m[k] for k in ("MAE", "RMSE", "bias", "MAE_wet")}


class Pruning(pl.Callback):
    """Report the slice validation loss after every epoch; stop training when the median pruner says so."""
    def __init__(self, trial, sign):
        self.trial, self.sign, self.pruned = trial, sign, False

    def on_validation_epoch_end(self, trainer, module):
        if trainer.sanity_checking or "val_loss" not in trainer.callback_metrics:
            return
        epoch = trainer.current_epoch
        self.trial.report(self.sign * float(trainer.callback_metrics["val_loss"]), epoch)
        if self.trial.should_prune():
            self.pruned, trainer.should_stop = True, True


def study_name(kind, task):
    return f"deep_{kind}__{task}"


def storage_url(kind, task):
    return f"sqlite:///{OPTUNA_DIR / (study_name(kind, task) + '.db')}"


def tune(kind, task, n_trials=None, storage=None):
    import torch
    torch.set_num_threads(1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = n_trials or BUDGET[kind]
    transform, (metric, direction) = ROUTE[task][kind], OBJECTIVE[task]
    sign = -1.0 if direction == "maximize" else 1.0          # intermediate values must improve in the study direction
    study = optuna.create_study(study_name=study_name(kind, task), storage=storage or storage_url(kind, task),
                                load_if_exists=True, direction=direction,
                                sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED, n_startup_trials=5),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5))
    for t in study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.RUNNING]):
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)
    if not study.trials:
        study.enqueue_trial(default_params(kind))           # trial 0 = Stage-5 configuration

    def objective(trial):
        p = space(trial, kind)
        cb = Pruning(trial, sign)
        sl, info = D.run(kind, transform, L, TUNE_SEED, save=False, params=p, inner_only=True, extra_callbacks=[cb],
                         tag=f"tune_{kind}_{task}_{trial.number}")
        if cb.pruned:
            raise optuna.TrialPruned()
        value, logged = slice_score(task, transform, sl)
        for k, v in {**logged, "best_epoch": info["best_epoch"], "n_params": info["n_params"],
                     "seconds": info["fit_seconds"], "smear_factor": info["smear_factor"]}.items():
            trial.set_user_attr(k, float(v))
        return float(value)

    done = len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]))
    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done, gc_after_trial=True)
    if storage is None:
        save_best(kind, task, study)
    return study.best_params, study


def save_best(kind, task, study):
    b = study.best_trial
    out = {"model": kind, "task": task, "route": ROUTE[task][kind], "lookback": L, "selection": "2012-2014 slice (inner model)",
           "objective": OBJECTIVE[task][0], "best_value_slice": b.value, "best_trial": b.number, "best_params": b.params,
           "default_value_slice": study.trials[0].value, "best_user_attrs": b.user_attrs,
           "trial_states": pd.Series([t.state.name for t in study.trials]).value_counts().to_dict(),
           "trial_minutes": sum(t.duration.total_seconds() for t in study.trials if t.duration) / 60}
    (OPTUNA_DIR / f"best_params_deep_{kind}__{task}.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def load_best(kind, task):
    return json.loads((OPTUNA_DIR / f"best_params_deep_{kind}__{task}.json").read_text())


def load_study(kind, task):
    return optuna.load_study(study_name=study_name(kind, task), storage=storage_url(kind, task))


# ---------- 5-seed re-runs of the selected setting (full protocol: inner fit -> refit -> 2015-2019) ----------
def tag(kind, task):
    return f"{kind}_{ROUTE[task][kind]}_L{L}_tuned"


def score_run(kind, task, out, info, seed, refs_bundle, label):
    """Tidy validation scores of one run (same metrics/references as the classical models on the DL split)."""
    va, y, refs = refs_bundle
    assert out.index.equals(va)
    transform = ROUTE[task][kind]
    if task == "cls":
        sl = pd.read_parquet(TUNED_DIR / f"{tag(kind, task)}__seed{seed}__slice.parquet") if label == "tuned" else \
            pd.read_parquet(D.DEEP_DIR / f"{kind}_bin_L{L}__seed{seed}__slice.parquet")
        thr = M.choose_threshold((sl["y_true"] >= D.WET_MM).astype(int), sl["p"], "CSI")
        m = E.score("cls", y.to_numpy(), out["p"].to_numpy(), thr, refs, va)
    else:
        m = E.score("reg", y.to_numpy(), out[POINT[transform]].to_numpy(), np.nan, refs, va)
    m.update(best_epoch=info.get("best_epoch"), fit_seconds=info.get("fit_seconds"), n_params=info.get("n_params"))
    return pd.DataFrame([{"model": kind, "task": task, "config": label, "seed": seed, "metric": k, "value": v}
                         for k, v in m.items()])


def references(task):
    import deep_cls as DC
    return DC.cls_references() if task == "cls" else D.dl_references()


def rerun_best(kind, task, seeds=(0, 1, 2, 3, 4)):
    """Refit the selected setting with each seed (skips seeds already on disk) and store tidy scores."""
    import torch
    torch.set_num_threads(1)
    params, refs, rows = load_best(kind, task)["best_params"], references(task), []
    for seed in seeds:
        stem = TUNED_DIR / f"{tag(kind, task)}__seed{seed}"
        if Path(f"{stem}.parquet").exists():
            out, info = pd.read_parquet(f"{stem}.parquet"), json.loads(Path(f"{stem}_history.json").read_text())
        else:
            out, info = D.run(kind, ROUTE[task][kind], L, seed, params=params, tag=tag(kind, task), out_dir=TUNED_DIR)
        rows.append(score_run(kind, task, out, info, seed, refs, "tuned"))
    return store(pd.concat(rows))


def default_scores(kind, task, seeds=(0, 1, 2, 3, 4)):
    """Scores of the stored Stage-5 default runs, recomputed with score_run (identical metric code)."""
    refs, rows = references(task), []
    transform = ROUTE[task][kind]
    for seed in seeds:
        stem = D.DEEP_DIR / f"{kind}_{transform}_L{L}__seed{seed}"
        info = json.loads(Path(f"{stem}_history.json").read_text())
        rows.append(score_run(kind, task, pd.read_parquet(f"{stem}.parquet"), info, seed, refs, "default"))
    return store(pd.concat(rows))


def store(res):
    keys = ["model", "task", "config", "seed"]
    table = res
    if RESULTS.exists():
        old = pd.read_csv(RESULTS)
        k = lambda d: d[keys].astype(str).agg("|".join, axis=1)
        table = pd.concat([old[~k(old).isin(set(k(res)))], res], ignore_index=True)
    table.to_csv(RESULTS, index=False)
    return res


def _job(key):
    import os
    import warnings
    os.environ["OMP_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore")
    kind, task = key
    t0 = time.time()
    tune(kind, task)
    rerun_best(kind, task)
    return key, time.time() - t0


def run_all(workers=3):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    order = [("tft", "cls"), ("tft", "reg"), ("tide", "cls"), ("tide", "reg"), ("tsmixer", "cls"), ("tsmixer", "reg")]
    with ProcessPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(_job, k) for k in order]):
            key, sec = f.result()
            print(f"{key}: {sec / 60:.1f} min", flush=True)
    for kind, task in order:
        default_scores(kind, task)
    print("done", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    run_all(ap.parse_args().workers)
