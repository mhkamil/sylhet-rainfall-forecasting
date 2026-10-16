"""Tests for the PyTorch MLP wrapper and its integration with the runner's early-stopping path."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment as E  # noqa: E402
from mlp import TorchMLP  # noqa: E402


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)) * [1, 100, 0.01], columns=["a", "b", "c"]).assign(month=rng.integers(1, 13, n))
    return X, (X["a"] + X["b"] / 100 > 0).astype(int)


def test_early_stopping_and_exact_refit_epochs():
    X, y = _toy()
    m = TorchMLP(task="cls", seed=0, max_epochs=40, patience=5).fit(X[:400], y[:400], eval_set=(X[400:], y[400:]))
    assert 1 <= m.best_epoch_ <= len(m.history_["val"]) <= 40
    E.set_trees(m, m.best_epoch_)
    m.fit(X[:400], y[:400])
    assert len(m.history_["train"]) == m.best_epoch_ and m.history_["val"] == []


def test_scaler_is_fitted_on_training_rows_only_and_learns():
    X, y = _toy()
    m = TorchMLP(task="cls", seed=0, max_epochs=30, patience=None).fit(X[:400], y[:400])
    scaler = m.pre_.named_transformers_["remainder"]
    assert np.allclose(scaler.mean_, X[:400].drop(columns="month").mean().to_numpy())
    assert ((m.predict_proba(X[400:])[:, 1] >= 0.5) == y[400:]).mean() > 0.9


def test_same_seed_same_result():
    X, y = _toy()
    p = [TorchMLP(task="reg", loss="huber", seed=3, max_epochs=10).fit(X, X["a"]).predict(X) for _ in range(2)]
    assert np.allclose(p[0], p[1])
