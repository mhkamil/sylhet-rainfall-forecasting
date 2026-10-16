"""Hand-checked unit tests for src/metrics.py and src/baselines.py. Run: python -m pytest -q tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import baselines as B  # noqa: E402
import metrics as M  # noqa: E402

Y = np.array([1, 1, 0, 0])
P = np.array([0.9, 0.4, 0.6, 0.1])          # at 0.5: tp=1 (0.9), fn=1 (0.4), fp=1 (0.6), tn=1 (0.1)


def test_hard_metrics_by_hand():
    m = M.hard_metrics(Y, P >= 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    for k in ("accuracy", "balanced_accuracy", "precision", "recall_POD", "F1", "FAR"):
        assert m[k] == pytest.approx(0.5)
    assert m["CSI"] == pytest.approx(1 / 3)           # 1 / (1 + 1 + 1)
    assert m["HSS"] == pytest.approx(0.0)             # no better than chance


def test_perfect_forecast():
    m = M.hard_metrics(Y, Y)
    assert m["CSI"] == m["HSS"] == m["F1"] == 1 and m["FAR"] == 0


def test_probabilistic_scores_by_hand():
    m = M.classification_metrics(Y, P, p_ref=np.full(4, 0.5))
    assert m["ROC_AUC"] == pytest.approx(0.75)        # 3 of 4 (pos, neg) pairs ranked correctly
    assert m["Brier"] == pytest.approx((0.01 + 0.36 + 0.36 + 0.01) / 4)   # 0.185
    assert m["BSS"] == pytest.approx(1 - 0.185 / 0.25)                   # reference 0.5 -> Brier 0.25
    assert m["log_loss"] == pytest.approx(-np.mean(np.log([0.9, 0.4, 0.4, 0.9])))


def test_ece_by_hand():
    # bin 0.2-0.3: mean prob 0.25, observed 0.5 -> 0.25; bin 0.7-0.8: 0.75 vs 1.0 -> 0.25; weights 1/2 each
    assert M.ece([0, 1, 1, 1], [0.25, 0.25, 0.75, 0.75]) == pytest.approx(0.25)
    assert M.ece([0, 1], [0.0, 1.0]) == pytest.approx(0.0)


def test_choose_threshold_separates_classes():
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.3, 0.7, 0.8])
    t = M.choose_threshold(y, p, "CSI")
    assert 0.3 < t <= 0.7 and M.hard_metrics(y, p >= t)["CSI"] == 1


def test_regression_by_hand():
    m = M.regression_metrics([0, 2, 4], [1, 2, 3], y_ref=[2, 2, 2])
    assert m["MAE"] == pytest.approx(2 / 3) and m["RMSE"] == pytest.approx(np.sqrt(2 / 3))
    assert m["bias"] == pytest.approx(0)
    assert m["R2"] == pytest.approx(1 - (2 / 3) / (8 / 3))        # 0.75
    assert m["MSESS"] == pytest.approx(0.75)
    assert m["pearson_r"] == pytest.approx(1.0)
    assert m["MAE_wet"] == pytest.approx(0.5)                    # wet days: y=2 (err 0), y=4 (err 1) -> mean 0.5


def test_heavy_rain_scores_by_hand():
    m = M.regression_metrics([25, 5, 60, 0], [30, 25, 10, 0])
    assert m["recall_ge20"] == 0.5 and m["precision_ge20"] == 0.5   # hit 25; miss 60; false alarm on 5
    assert m["recall_ge50"] == 0 and np.isnan(m["precision_ge50"])  # never forecast >= 50 mm


def test_block_bootstrap_is_reproducible_and_brackets_estimate():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-01", periods=730)
    y = rng.normal(size=730)
    est, lo, hi = M.block_bootstrap(np.mean, dates, y, n_boot=200, seed=1)
    assert lo <= est <= hi
    assert (est, lo, hi) == M.block_bootstrap(np.mean, dates, y, n_boot=200, seed=1)


def test_doy_climatology_constant_and_circular():
    dates = pd.date_range("2001-01-01", "2002-12-31")
    clim = B.doy_climatology(dates, np.ones(len(dates)))
    assert np.allclose(clim, 1.0)                       # mean of ones is one everywhere, incl. day 366
    winter = np.where(dates.month.isin([12, 1]), 1.0, 0.0)
    clim = B.doy_climatology(dates, winter)
    assert clim[0] == pytest.approx(1.0) and clim[180] == pytest.approx(0.0)   # wraps 31 Dec -> 1 Jan
