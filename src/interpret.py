"""Stage 8 helpers: feature grouping, per-fold development models, grouped permutation importance, ALE.

Interpretation uses DEVELOPMENT data only: each of the 5 expanding CV folds refits the frozen (tuned) model on its training rows
(same protocol as Stage 4/6: inner slice -> trees, refit) and is explained on its own validation block (out-of-sample, 2005-2019).
The test years are never used here.
"""
import numpy as np
import pandas as pd

import experiment as E
import final_eval as FE
import splits as S
import tuning as T

DICT = pd.read_csv(E.ROOT / "reports/tables/02_feature_dictionary.csv")
FAMILY = dict(zip(DICT.feature, DICT.family.replace({"rolling rain sum": "rolling", "rolling mean": "rolling"})))
VARIABLE_GROUPS = [  # order matters: T2MDEW before T2M
    ("rain memory", ("rain_", "wet_", "days_since_rain")),
    ("moisture", ("RH2M", "QV2M", "T2MDEW", "DPD")),
    ("pressure", ("PS",)),
    ("wind", ("WS2M", "WS10M")),
    ("temperature", ("T2M", "DTR")),
    ("season", ("doy_", "month")),
]


def variable_group(col):
    return next(g for g, prefixes in VARIABLE_GROUPS if col.startswith(prefixes))


def groups(columns, kind):
    """{group: [columns]} for kind 'family' (engineering family) or 'variable' (physical quantity)."""
    key = (lambda c: FAMILY[c]) if kind == "family" else variable_group
    out = {}
    for c in columns:
        out.setdefault(key(c), []).append(c)
    return out


def development():
    X, Y = E.load(FE.FSET)
    dev = S.dev_index(X.index)
    return X.loc[dev], Y.loc[dev]


def fold_model(task, name, seed, X, y, tr):
    """Refit one frozen model on the fold's training rows (inner slice -> trees, then refit) and return it."""
    factory = FE.classical_factory(task, name)
    _, n_trees, _ = E.inner_fit(factory, task, seed, X, y, tr)
    est = factory(seed, y[tr])
    if E.booster_family(est) is not None:
        E.set_trees(est, int(n_trees))
    return est.fit(X.loc[tr], y[tr])


def predict(est, X, task):
    return E.predict(est, X, task)


def loss(task, y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return np.mean((p - y) ** 2) if task == "cls" else np.mean(np.abs(p - y))


def permutation_importance(est, X, y, task, grouping, n_repeats=5, seed=0):
    """Increase of the loss (Brier / MAE) when the columns of a group are permuted JOINTLY (rows shuffled together,
    keeping the within-group correlation); one row per group and repeat."""
    rng = np.random.default_rng(seed)
    base = loss(task, y, predict(est, X, task))
    rows = []
    for g, cols in grouping.items():
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            rows.append({"group": g, "repeat": r, "delta_loss": loss(task, y, predict(est, Xp, task)) - base,
                         "base_loss": base})
    return pd.DataFrame(rows)


def ale(est, X, feature, task, bins=20):
    """First-order accumulated local effects (Apley & Zhu 2020) on the prediction scale (probability / mm).
    Robust to correlated features because it only moves each row within its own quantile bin."""
    x = X[feature].to_numpy()
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return pd.DataFrame({"x": edges, "ale": 0.0})
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    effects, counts = [], []
    for k in range(len(edges) - 1):
        rows = idx == k
        if not rows.any():
            effects.append(0.0); counts.append(0); continue
        lo, hi = X.loc[rows].copy(), X.loc[rows].copy()
        lo[feature], hi[feature] = edges[k], edges[k + 1]
        effects.append(float(np.mean(predict(est, hi, task) - predict(est, lo, task))))
        counts.append(int(rows.sum()))
    acc = np.concatenate([[0.0], np.cumsum(effects)])
    centred = acc - np.average((acc[:-1] + acc[1:]) / 2, weights=np.maximum(counts, 1))
    return pd.DataFrame({"x": edges, "ale": centred})
