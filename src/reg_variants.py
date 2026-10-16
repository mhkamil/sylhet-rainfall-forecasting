"""Stage 4.5 regression variants, all evaluated in mm with the same folds, references and metrics as src/experiment.py.

log    : fit on z = log(1 + mm); back-transform (a) naive expm1(z_hat), (b) Duan smearing exp(z_hat) * S - 1,
         S = mean(exp(residual)) on the out-of-sample inner slice.
hurdle : P(rain >= 1 mm) from the CatBoost classifier's saved OOF predictions (same seed and feature set)
         x E[mm | wet], a regressor fitted on wet training days only (direct mm, or log + smearing).
(Tweedie boosters need no special code: they run through experiment.run_cv with factories from models.tweedie().)
"""
from pathlib import Path

import numpy as np
import pandas as pd

import experiment as E

WET_MM = 1.0
HURDLE_CLASSIFIER = "catboost"


def _fit(factory, seed, X, y, tr):
    """Inner fit (early stopping for boosters / smearing for all) and refit on the whole `tr`."""
    itr, sl = E.inner_split(tr)
    est, inner = factory(seed, y[tr]), factory(seed, y[itr])
    if E.booster_family(est):
        E.set_trees(est, E.fit_early_stopping(inner, X.loc[itr], y[itr], X.loc[sl], y[sl]))
    else:
        inner.fit(X.loc[itr], y[itr])
    est.fit(X.loc[tr], y[tr])
    return est, inner, sl


def _smearing(inner, X, z, sl):
    return float(np.mean(np.exp(z[sl] - inner.predict(X.loc[sl]))))


def _store(name, fset, seed, rows, oof, save):
    res = E.tidy(rows, name, "reg", fset, seed)
    if save:
        E.append_results(res)
        E.OOF_DIR.mkdir(parents=True, exist_ok=True)
        pd.concat(oof).to_parquet(Path(E.OOF_DIR) / f"{name}__reg__{fset}__seed{seed}.parquet")
    return res


def run_log(name, factory, fset, seed=0, save=True):
    """Two result sets from one fit: '<name>+log_naive' and '<name>+log_smear'."""
    X, Y = E.load(fset)
    y, z, rain = Y["y_reg"], np.log1p(Y["y_reg"]), E.rain_today(X.index)
    out = {"log_naive": ([], []), "log_smear": ([], [])}
    for k, (tr, va) in enumerate(E.S.cv_folds(X.index), 1):
        est, inner, sl = _fit(factory, seed, X, z, tr)
        s = _smearing(inner, X, z, sl)
        z_hat = est.predict(X.loc[va])
        refs = E.references("reg", X, Y, rain, tr, va)
        for kind, pred in (("log_naive", np.expm1(z_hat)), ("log_smear", np.exp(z_hat) * s - 1)):
            pred = np.clip(pred, 0, None)
            out[kind][0].append({"fold": k, **E.score("reg", y[va], pred, np.nan, refs, va), "smear_factor": s})
            out[kind][1].append(pd.DataFrame({"fold": k, "y_true": y[va].to_numpy(), "pred": pred}, index=va))
    return pd.concat([_store(f"{name}+{kind}", fset, seed, r, o, save) for kind, (r, o) in out.items()])


def run_hurdle(name, factory, fset, seed=0, save=True):
    """Two result sets: 'hurdle_<name>_direct' and 'hurdle_<name>_log' (log with smearing)."""
    X, Y = E.load(fset)
    y, rain = Y["y_reg"], E.rain_today(X.index)
    p_all = pd.read_parquet(Path(E.OOF_DIR) / f"{HURDLE_CLASSIFIER}__cls__{fset}__seed{seed}.parquet")["pred"]
    out = {"direct": ([], []), "log": ([], [])}
    for k, (tr, va) in enumerate(E.S.cv_folds(X.index), 1):
        wet = tr[y[tr].to_numpy() >= WET_MM]
        refs = E.references("reg", X, Y, rain, tr, va)
        p = p_all.loc[va].to_numpy()
        est, _, _ = _fit(factory, seed, X, y, wet)
        amount = {"direct": np.clip(est.predict(X.loc[va]), WET_MM, None)}
        z = np.log1p(y)
        est, inner, sl = _fit(factory, seed, X, z, wet)
        amount["log"] = np.clip(np.exp(est.predict(X.loc[va])) * _smearing(inner, X, z, sl) - 1, WET_MM, None)
        for kind, m in amount.items():
            pred = p * m
            out[kind][0].append({"fold": k, **E.score("reg", y[va], pred, np.nan, refs, va)})
            out[kind][1].append(pd.DataFrame({"fold": k, "y_true": y[va].to_numpy(), "pred": pred, "p_wet": p,
                                              "amount_if_wet": m}, index=va))
    return pd.concat([_store(f"hurdle_{name}_{kind}", fset, seed, r, o, save) for kind, (r, o) in out.items()])
