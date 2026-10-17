"""Stage 7.2-7.3 analysis of the one-shot test predictions in models/test/ (written by final_eval.run).

Nothing here refits or changes a model: it only reads the frozen predictions, the test labels and the frozen protocol
(final_config.yaml) and computes scores, bootstrap CIs, Diebold-Mariano tests with Holm correction, and figures.
Conventions (pre-registered, final_config.yaml): seed-mean forecast per model for tests and CIs; per-seed scores as
mean +- std; the seed-mean classification forecast uses the mean of the frozen per-seed thresholds for hard scores.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import experiment as E
import final_eval as FE
import metrics as M
import splits as S

TEST_DIR = FE.TEST_DIR
SEASONS = {"pre-monsoon": (3, 4, 5), "monsoon": (6, 7, 8, 9), "post-monsoon": (10, 11), "winter": (12, 1, 2)}
N_BOOT, BOOT_SEED = 1000, 42
CLS_METRICS = ["ROC_AUC", "PR_AUC", "Brier", "BSS", "BSS_vs_seas_pers", "ECE", "F1", "recall_POD", "FAR", "CSI", "HSS"]
REG_METRICS = ["MAE", "RMSE", "bias", "spearman_rho", "R2", "MSESS", "MSESS_vs_seas_pers", "MAE_wet", "RMSE_wet",
               "recall_ge20", "precision_ge20", "recall_ge50", "precision_ge50"]


# ---------- loading ----------
def model_list(task, cfg=None):
    """(model id, family) in table order: baselines, classical, deep."""
    cfg = cfg or FE.load_config()
    out = [(f"baseline:{n}", "baseline") for n in FE.BASELINES]
    out += [(n, "classical") for n in cfg["models"]["classical"][task]]
    out += [(f"deep:{k}", "deep") for k in cfg["models"]["deep"][task]]
    return out


def _file(model, task, seed):
    stem = model.replace("deep:", "") if model.startswith("deep:") else model
    return TEST_DIR / f"{stem}__{task}__seed{seed}.parquet"


def point(f, task):
    """Pre-registered point forecast: Bernoulli p (deep cls), quantile mean (TFT), else `pred` (smeared for log routes)."""
    if task == "cls":
        return f["p"] if "p" in f else f["pred"]
    return f["pred_mean"] if "pred_mean" in f else f["pred"]


def load_predictions(task, cfg=None):
    """{model: DataFrame (index = feature dates of the test rows) with one column per seed}, plus thresholds."""
    cfg = cfg or FE.load_config()
    preds, thr, extra = {}, {}, {}
    for model, fam in model_list(task, cfg):
        seeds = sorted(int(f.stem.rsplit("seed", 1)[1]) for f in TEST_DIR.glob(_file(model, task, 0).name.replace(
            "seed0", "seed*")))
        frames = {s: pd.read_parquet(_file(model, task, s)) for s in seeds}
        preds[model] = pd.DataFrame({s: point(f, task) for s, f in frames.items()})
        if task == "cls":
            thr[model] = {s: float(f["threshold"].iloc[0]) for s, f in frames.items()}
        if model == "deep:tft" and task == "reg":
            extra["tft_quantiles"] = {s: f.filter(like="q0.") for s, f in frames.items()}
    return preds, thr, extra


def truth(source="PRECTOTCORR"):
    """Test rows (feature dates) and next-day targets; source='IMERG_PRECTOT' for the secondary check.
    Allowed only after the confirmed one-shot run (its manifest must exist)."""
    if not (TEST_DIR / "manifest.json").exists():
        raise PermissionError("test labels are read only after final_eval.run(confirm_final=True) has written its manifest")
    X, Y = E.load(FE.FSET)
    te = S.test_index(X.index, confirm_final=True)
    if source == "PRECTOTCORR":
        return te, Y.loc[te, "y_cls"].astype(int), Y.loc[te, "y_reg"]
    imerg = pd.read_parquet(E.PROC / "clean_daily.parquet")["IMERG_PRECTOT"]
    nxt = imerg.reindex(te + pd.Timedelta(days=1)).to_numpy()
    y = pd.Series(nxt, index=te)
    return te, (y >= 1.0).astype(int), y


# ---------- scoring ----------
def score(task, y, p, thr, refs, idx):
    return E.score(task, np.asarray(y), np.asarray(p), thr, refs, idx)


def references(preds):
    return {"climatology": preds["baseline:climatology"].mean(axis=1).to_numpy(),
            "seasonal+persistence": preds["baseline:seasonal+persistence"].mean(axis=1).to_numpy()}


def per_seed_table(task, preds, thr, y, idx, metrics):
    """mean +- std over seeds of every metric (deterministic models: one seed, std 0)."""
    refs = references(preds)
    rows = []
    for model, P in preds.items():
        for s in P.columns:
            m = score(task, y, P[s], thr.get(model, {}).get(s, np.nan) if task == "cls" else np.nan, refs, idx)
            rows.append({"model": model, "seed": s, **{k: m[k] for k in metrics}})
    r = pd.DataFrame(rows)
    return r.groupby("model")[metrics].agg(["mean", "std"]).fillna(0.0), r


def seed_mean(preds, thr=None):
    p = {m: P.mean(axis=1).to_numpy() for m, P in preds.items()}
    t = {m: float(np.mean(list(v.values()))) for m, v in (thr or {}).items()}
    return p, t


def _blocks(idx, n_boot=N_BOOT, seed=BOOT_SEED):
    """The same resampled calendar-month blocks (of the target date) for every model -> paired comparisons."""
    month = S.target_dates(idx).to_period("M").astype(str).to_numpy()
    ids = np.unique(month)
    rows = [np.flatnonzero(month == b) for b in ids]
    rng = np.random.default_rng(seed)
    return [np.concatenate([rows[i] for i in rng.integers(0, len(ids), len(ids))]) for _ in range(n_boot)]


def bootstrap_table(task, pm, tm, y, idx, metrics, n_boot=N_BOOT):
    """Point estimate (seed-mean forecast) and 95 % percentile CI of every metric, same blocks for all models."""
    y = np.asarray(y)
    refs = {"climatology": pm["baseline:climatology"], "seasonal+persistence": pm["baseline:seasonal+persistence"]}
    blocks = _blocks(idx, n_boot)
    out = {}
    for model, p in pm.items():
        est = score(task, y, p, tm.get(model, np.nan), refs, idx)
        boot = []
        for b in blocks:
            m = score(task, y[b], p[b], tm.get(model, np.nan), {k: v[b] for k, v in refs.items()}, idx[b])
            boot.append([m[k] for k in metrics])
        lo, hi = np.nanpercentile(np.array(boot, float), [2.5, 97.5], axis=0)
        out[model] = {**{f"{k}": est[k] for k in metrics}, **{f"{k}_lo": l for k, l in zip(metrics, lo)},
                      **{f"{k}_hi": h for k, h in zip(metrics, hi)}}
    return pd.DataFrame(out).T


def formatted(task, seed_tab, boot_tab, metrics, cfg=None):
    """One cell per metric: 'seed mean ± std [CI low, CI high]'."""
    fam = dict(model_list(task, cfg))
    rows = []
    for model in seed_tab.index:
        row = {"family": fam[model], "model": model, "n_seeds": None}
        for k in metrics:
            mu, sd = seed_tab.loc[model, (k, "mean")], seed_tab.loc[model, (k, "std")]
            lo, hi = boot_tab.loc[model, f"{k}_lo"], boot_tab.loc[model, f"{k}_hi"]
            row[k] = f"{mu:.3f} ± {sd:.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    order = [m for m, _ in model_list(task, cfg)]
    return pd.DataFrame(rows).set_index("model").loc[order]


# ---------- paired differences, DM tests, Holm ----------
def losses(task, y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"Brier": (p - y) ** 2} if task == "cls" else {"abs_error": np.abs(p - y), "sq_error": (p - y) ** 2}


EFFECT = {"Brier": "dBSS", "abs_error": "dMAE", "sq_error": "dMSESS"}


def comparisons(task, pm, y, idx, n_boot=N_BOOT):
    """Family A: every model vs seasonal+persistence; family B: vs the pre-declared headline reference.
    Effect sizes (> 0 better for dBSS/dMSESS, < 0 better for dMAE) with paired block-bootstrap CIs; DM p-values,
    Holm-adjusted within family x loss."""
    y = np.asarray(y, float)
    clim = pm["baseline:climatology"]
    blocks = _blocks(idx, n_boot)
    refs = {"A": {loss: "baseline:seasonal+persistence" for loss in losses(task, y, clim)}, "B": FE.REFERENCES[task]}
    rows = []
    for fam, by_loss in refs.items():
        for loss, ref in by_loss.items():
            for model, p in pm.items():
                if model == ref or model == "baseline:seasonal+persistence" and fam == "A":
                    continue
                la, lb, lc = losses(task, y, p)[loss], losses(task, y, pm[ref])[loss], losses(task, y, clim)[loss]
                eff = (lambda a, b, c: (b.mean() - a.mean()) / c.mean()) if loss != "abs_error" else \
                    (lambda a, b, c: a.mean() - b.mean())
                est = eff(la, lb, lc)
                boot = [eff(la[b], lb[b], lc[b]) for b in blocks]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                d, stat, pval = FE.dm_test(la, lb)
                rows.append({"family": fam, "task": task, "loss": loss, "model": model, "reference": ref,
                             "effect": EFFECT[loss], "estimate": est, "ci_low": lo, "ci_high": hi,
                             "mean_loss_diff": d, "DM": stat, "p": pval})
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby(["family", "loss"])["p"].transform(lambda s: FE.holm(s.to_numpy()))
    out["significant"] = out["p_holm"] < 0.05
    out["better"] = np.where(out.effect == "dMAE", out.estimate < 0, out.estimate > 0)
    return out


# ---------- season breakdown ----------
def season_of(idx):
    m = S.target_dates(idx).month
    return pd.Series([next(s for s, ms in SEASONS.items() if mm in ms) for mm in m], index=idx)


def by_season(task, pm, tm, y, idx, models):
    y = np.asarray(y)
    seas = season_of(idx).to_numpy()
    refs = {"climatology": pm["baseline:climatology"], "seasonal+persistence": pm["baseline:seasonal+persistence"]}
    rows = []
    for s in SEASONS:
        k = seas == s
        for model in models:
            if task == "cls":                       # plain metrics: E.score would add a monsoon-only AUC (empty here)
                m = M.classification_metrics(y[k], pm[model][k], tm.get(model, 0.5), p_ref=refs["climatology"][k])
            else:
                m = M.regression_metrics(y[k], pm[model][k], y_ref=refs["climatology"][k])
            rows.append({"season": s, "model": model, "n_days": int(k.sum()), **{
                key: m[key] for key in (["BSS", "ROC_AUC", "CSI"] if task == "cls" else ["MAE", "MSESS", "bias"])}})
    return pd.DataFrame(rows)


# ---------- development vs test ranking ----------
def dev_scores(task, metric):
    """Development 5-fold CV mean of the frozen classical models and baselines (Stage 6 names)."""
    cv = pd.read_csv(E.RESULTS)
    cfg = FE.load_config()
    out = {}
    for model, fam in model_list(task, cfg):
        if fam == "deep":
            continue
        dev_name = model if fam == "baseline" or cfg["models"]["classical"][task][model]["config"] == "default" \
            else f"{model}_tuned"
        r = cv[(cv.model == dev_name) & (cv.task == task) & (cv.metric == metric) &
               (cv.feature_set.isin(["B", "-"]))]
        if len(r):
            out[model] = r.groupby("fold").value.mean().mean()
    return pd.Series(out)


def to_latex(df, path, caption, label):
    Path(path).write_text(df.to_latex(escape=True, caption=caption, label=label, na_rep="--"))
