"""Stage 4.6 reporting helpers: master tables with paired fold-wise win counts, seed stability, test-set audit."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

import experiment as E
import splits as S

LOWER_IS_BETTER = {"MAE", "RMSE", "MAE_wet", "RMSE_wet", "Brier", "log_loss", "ECE", "MAE_monsoon"}


def fold_table(cv, task, metric):
    """Seed-averaged value per (model, feature_set) x fold."""
    s = cv[(cv.task == task) & (cv.metric == metric)]
    return s.groupby(["model", "feature_set", "fold"])["value"].mean().unstack("fold")


def wins(cv, task, metric, ref_model, ref_fset=None):
    """Number of folds (of 5) in which each configuration beats the reference on `metric` (paired by fold).
    ref_fset=None compares with the reference trained on the same feature set; baselines use '-'."""
    ft = fold_table(cv, task, metric)
    out = {}
    for (model, fs), row in ft.iterrows():
        key = (ref_model, ref_fset or fs)
        if key not in ft.index:
            continue
        d = row - ft.loc[key]
        out[(model, fs)] = int((d < 0).sum() if metric in LOWER_IS_BETTER else (d > 0).sum())
    return pd.Series(out, dtype="Int64")


def master_table(cv, task, metrics, rank_by, refs):
    """mean ± fold std for `metrics`, folds 1-3 / 4-5 means and seed std of rank_by[0], and win counts vs refs
    ({label: (model, fset or None)}) on rank_by[0]; sorted by rank_by (first key primary)."""
    s = E.summarise(cv[cv.task == task], metrics)
    s["cell"] = s["mean"].map("{:.3f}".format) + " ± " + s["fold_std"].map("{:.3f}".format)
    tab = s.pivot_table(index=["model", "feature_set"], columns="metric", values="cell", aggfunc="first")[metrics]
    head = s[s.metric == rank_by[0]].set_index(["model", "feature_set"])
    tab[f"{rank_by[0]} folds 1-3"] = head["folds_1_3"].round(3)
    tab[f"{rank_by[0]} folds 4-5"] = head["folds_4_5"].round(3)
    tab[f"{rank_by[0]} seed std"] = head["seed_std"].fillna(0).round(4)
    tab["n_seeds"] = head["n_seeds"]
    for label, (m, fs) in refs.items():
        tab[f"wins vs {label}"] = wins(cv, task, rank_by[0], m, fs).reindex(tab.index).astype(str) + "/5"
    keys = {k: s[s.metric == k].set_index(["model", "feature_set"])["mean"] for k in rank_by}
    order = pd.DataFrame(keys).sort_values(list(rank_by), ascending=[k in LOWER_IS_BETTER for k in rank_by]).index
    return tab.loc[order]


def seed_stability(cv, metrics):
    """Seed std of the 5-fold mean for every multi-seed configuration, largest first."""
    s = E.summarise(cv[cv.metric.isin(metrics)], metrics)
    s = s[s.n_seeds > 1]
    return s.sort_values("seed_std", ascending=False)[["model", "task", "feature_set", "metric", "mean", "seed_std", "n_seeds"]]


def test_set_audit(root=E.ROOT):
    """Evidence that no stored prediction or score involves a 2020+ target, and that the test getter is only
    unlocked in the unit test that checks the lock."""
    oof = sorted(Path(E.OOF_DIR).glob("*.parquet"))
    last_target = max(S.target_dates(pd.read_parquet(f, columns=["fold"]).index).max() for f in oof)
    cv = pd.read_csv(E.RESULTS)
    call = re.compile(r"test_index\(\s*[A-Za-z_][\w.]*\s*,\s*confirm_final\s*=\s*True\s*\)")   # a real call with an index
    unlock = [str(p.relative_to(root)) for p in list((root / "src").glob("*.py")) + list((root / "tests").glob("*.py"))
              + list((root / "notebooks").glob("*.ipynb")) if call.search(p.read_text())]
    return {"OOF files checked": len(oof), "latest target date in any OOF file": str(last_target.date()),
            "folds present in cv_results": sorted(cv.fold.unique().tolist()),
            "files calling test_index(..., confirm_final=True)": unlock}
