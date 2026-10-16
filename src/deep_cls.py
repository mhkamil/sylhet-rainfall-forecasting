"""Stage 5.5: rain / no-rain (>= 1 mm) on the 2015-2019 validation period from the deep models.

Option 1  (all models): the regression forecast (mm) is the score. (a) raw: a mm threshold maximising CSI on the
          2012-2014 slice (yes/no forecasts only, no probabilities); (b) Platt: logistic regression of the outcome on
          log1p(score) fitted on the slice; (c) isotonic: monotone non-parametric mapping fitted on the slice.
Option 2  (TFT): P(rain >= 1 mm) read off the predicted quantile distribution (linear interpolation of the CDF between
          the quantile knots; clamped to [0.05, 0.95] outside them), raw and isotonic-calibrated on the slice.
Option 3  (all models): darts BernoulliLikelihood trained directly on the binary series (native in darts 0.48),
          raw and isotonic-calibrated on the slice.
All slice forecasts come from the inner models (trained <= 2011); the validation forecasts from the refitted models.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import deep_ts as D
import experiment as E
import metrics as M
import splits as S

RESULTS = D.ROOT / "reports/tables/deep_cls_results.csv"
PRED_DIR = D.DEEP_DIR / "cls"
MM_GRID = np.round(np.arange(0.0, 20.01, 0.1), 1)


def cls_references():
    X, Y = E.load("A")
    tr, va = S.dl_split(X.index)
    return va, Y.loc[va, "y_cls"], E.references("cls", X, Y, E.rain_today(X.index), tr, va)


def prob_from_quantiles(q):
    """P(Y >= 1 mm) from a frame of quantile columns q0.05..q0.95 (rows already sorted / non-crossing)."""
    taus = np.array(D.QUANTILES)
    vals = q[[f"q{t}" for t in D.QUANTILES]].to_numpy()
    cdf_at_1 = np.array([np.interp(D.WET_MM, row, taus, left=taus[0], right=taus[-1]) for row in vals])
    return 1 - cdf_at_1


def calibrators(score_slice, y_slice):
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(score_slice, y_slice)
    platt = LogisticRegression().fit(np.log1p(np.clip(score_slice, 0, None)).reshape(-1, 1), y_slice)
    return {"isotonic": iso.predict,
            "platt": lambda s: platt.predict_proba(np.log1p(np.clip(s, 0, None)).reshape(-1, 1))[:, 1]}


def raw_threshold_scores(y, score, thr, monsoon):
    """Ranking + yes/no scores for an uncalibrated mm score (no probabilities -> no Brier/BSS/ECE)."""
    y = np.asarray(y)
    return {"ROC_AUC": roc_auc_score(y, score), "PR_AUC": average_precision_score(y, score),
            "ROC_AUC_monsoon": roc_auc_score(y[monsoon], score[monsoon]), **M.hard_metrics(y, score >= thr),
            "threshold": thr}


def evaluate(kind, transform, seed, refs_bundle, L=14):
    """All options available for one deep run (files from deep_ts.run). Returns tidy rows; saves probabilities."""
    va, y, refs = refs_bundle
    stem = D.DEEP_DIR / f"{kind}_{transform}_L{L}__seed{seed}"
    val, sl = pd.read_parquet(f"{stem}.parquet"), pd.read_parquet(f"{stem}__slice.parquet")
    assert val.index.equals(va)
    y_sl = (sl["y_true"] >= D.WET_MM).astype(int).to_numpy()
    monsoon = S.target_dates(va).month.isin(E.MONSOON)
    scores, probs = {}, {}
    if transform == "bin":                                              # option 3
        probs["opt3_raw"] = (val["p"].to_numpy(), sl["p"].to_numpy())
        probs["opt3_isotonic"] = tuple(calibrators(sl["p"].to_numpy(), y_sl)["isotonic"](v)
                                       for v in (val["p"].to_numpy(), sl["p"].to_numpy()))
    else:
        col = "pred_mean" if "pred_mean" in val else "pred"            # option 1 score (mm)
        s_val, s_sl = val[col].to_numpy(), sl[col].to_numpy()
        thr = float(MM_GRID[int(np.nanargmax([M.hard_metrics(y_sl, s_sl >= t)["CSI"] for t in MM_GRID]))])
        scores["opt1_raw"] = raw_threshold_scores(y, s_val, thr, monsoon)
        for name, f in calibrators(s_sl, y_sl).items():
            probs[f"opt1_{name}"] = (f(s_val), f(s_sl))
        if "q0.5" in val:                                                # option 2 (TFT quantiles)
            p_val, p_sl = prob_from_quantiles(val), prob_from_quantiles(sl)
            probs["opt2_raw"] = (p_val, p_sl)
            probs["opt2_isotonic"] = tuple(calibrators(p_sl, y_sl)["isotonic"](v) for v in (p_val, p_sl))
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for option, (p_val, p_sl) in probs.items():
        thr = M.choose_threshold(y_sl, p_sl, "CSI")
        scores[option] = E.score("cls", y, p_val, thr, refs, va)
        pd.DataFrame({"y_true": y.to_numpy(), "pred": p_val}, index=va).to_parquet(
            PRED_DIR / f"{kind}_{transform}_{option}__seed{seed}.parquet")
    rows = [{"model": kind, "target": transform, "option": o, "seed": seed, "metric": k, "value": v}
            for o, m in scores.items() for k, v in m.items()]
    return pd.DataFrame(rows)


def store(res):
    keys = ["model", "target", "option", "seed"]
    table = res
    if RESULTS.exists():
        old = pd.read_csv(RESULTS)
        k = lambda d: d[keys].astype(str).agg("|".join, axis=1)
        table = pd.concat([old[~k(old).isin(set(k(res)))], res], ignore_index=True)
    table.to_csv(RESULTS, index=False)
