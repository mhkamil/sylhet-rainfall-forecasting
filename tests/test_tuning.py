"""Tests for src/tuning.py: tuning cannot see the locked test years (targets 2020-2025) and validation labels never
reach preprocessing, thresholds or early stopping. Run: python -m pytest -q tests/test_tuning.py"""
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment as E  # noqa: E402
import metrics as M  # noqa: E402
import splits as S  # noqa: E402
import tuning as T  # noqa: E402

FAST = {"cls": ("sgd_logloss", {"alpha": 1e-3, "penalty": "l2", "learning_rate": "optimal"}),
        "reg": ("ridge", {"alpha": 1.0})}
LGBM = {"learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 20, "colsample_bytree": 0.8, "subsample": 0.8,
        "reg_alpha": 0.01, "reg_lambda": 0.01}
LAST_DEV_TARGET = pd.Timestamp("2019-12-31")


def poisoned_load(fset):
    """Same data, but every row whose target lies in 2020-2025 is replaced by garbage (features and labels)."""
    X, Y = _real_load(fset)
    test = S.target_dates(X.index) > LAST_DEV_TARGET
    X, Y = X.copy(), Y.copy()
    num = X.columns[X.dtypes != object]
    X.loc[test, num] = 1e6
    Y.loc[test] = 999
    return X, Y


_real_load = E.load


def _values(study):
    return [t.value for t in study.trials]


@pytest.mark.parametrize("task", ["cls", "reg"])
def test_prepared_data_end_before_the_test_years(task):
    d = T.prepare(task)
    assert S.target_dates(d["X"].index).max() <= LAST_DEV_TARGET
    for tr, va in d["folds"]:
        assert S.target_dates(va).max() <= LAST_DEV_TARGET
        assert S.target_dates(tr).max() < S.target_dates(va).min()


@pytest.mark.parametrize("task", ["cls", "reg"])
def test_poisoning_the_test_years_changes_nothing(task, tmp_path, monkeypatch):
    """If any trial could read a 2020-2025 row, garbage there would change its score."""
    name = FAST[task][0]
    _, clean = T.tune(name, task, 3, storage=f"sqlite:///{tmp_path / 'clean.db'}")
    monkeypatch.setattr(E, "load", poisoned_load)
    _, dirty = T.tune(name, task, 3, storage=f"sqlite:///{tmp_path / 'dirty.db'}")
    assert np.allclose(_values(clean), _values(dirty))
    assert [t.params for t in clean.trials] == [t.params for t in dirty.trials]       # same sampler path (seed 42)


def test_negative_control_a_leaky_fold_is_detected(monkeypatch):
    """Sanity check of the test above: if fold 5's training rows included 2020+ rows, the poison would show up."""
    name, params = FAST["reg"]
    clean = T.evaluate(name, params, T.prepare("reg"))["MAE"].iloc[-1]
    monkeypatch.setattr(E, "load", poisoned_load)
    monkeypatch.setattr(S, "dev_index", lambda idx: idx)                     # leak 1: keep the test years
    real_folds = S.cv_folds
    leaky = lambda idx: real_folds(idx)[:4] + [(real_folds(idx)[4][0].append(idx[S.target_dates(idx) > LAST_DEV_TARGET]),
                                                 real_folds(idx)[4][1])]   # leak 2: train fold 5 on them
    monkeypatch.setattr(S, "cv_folds", leaky)
    dirty = T.evaluate(name, params, T.prepare("reg"))["MAE"].iloc[-1]
    assert not np.isclose(clean, dirty)


@pytest.mark.parametrize("task,name,params",[("cls", *FAST["cls"]), ("reg", *FAST["reg"]), ("cls", "lightgbm", LGBM)])
def test_validation_labels_do_not_change_forecasts_thresholds_or_trees(task, name, params):
    """For each fold, scramble every label from the validation block onwards: forecasts, the CSI threshold and the
    early-stopping tree count must be identical, i.e. those labels reach only the final scoring."""
    d = T.prepare(task)
    rng = np.random.default_rng(0)
    for k in (0, 4):
        tr, va = d["folds"][k]
        ref = T.fold_predict(name, params, d, k)
        y = d["y"].copy()
        later = S.target_dates(y.index) >= S.target_dates(va).min()
        y[later] = rng.permutation(y[later].to_numpy()) if task == "cls" else y[later] * 3 + 50
        out = T.fold_predict(name, params, {**d, "y": y}, k)
        assert np.allclose(ref[0], out[0])
        assert ref[1] == out[1] or (np.isnan(ref[1]) and np.isnan(out[1]))
        assert ref[2] == out[2] or (np.isnan(ref[2]) and np.isnan(out[2]))


class Spy:
    """Classifier that records the rows it is fitted on (stands in for any model, incl. its Pipeline)."""
    fitted = []

    def fit(self, X, y):
        Spy.fitted.append(X.index)
        self.rate = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.rate), np.full(len(X), self.rate)])


def test_fits_and_threshold_only_see_training_rows_of_the_fold(monkeypatch):
    seen = []
    real = M.choose_threshold
    monkeypatch.setattr(M, "choose_threshold", lambda y, p, metric="CSI": seen.append(y.index) or real(y, p, metric))
    monkeypatch.setitem(T.MODELS["cls"], "spy", (lambda t: {}, lambda p, task: (lambda s, y: Spy()), False, "fit_fold"))
    Spy.fitted = []
    d = T.prepare("cls")
    T.evaluate("spy", {}, d)
    assert len(Spy.fitted) == 2 * len(d["folds"]) and len(seen) == len(d["folds"])
    for k, (tr, va) in enumerate(d["folds"]):
        for idx in Spy.fitted[2 * k: 2 * k + 2] + [seen[k]]:
            assert idx.isin(tr).all()                                   # never a validation (or test) row
        assert Spy.fitted[2 * k + 1].equals(tr)                         # refit on the whole training fold
        assert not seen[k].isin(Spy.fitted[2 * k]).any()                # threshold slice is out-of-sample for the inner fit


def test_study_uses_seeded_tpe_median_pruner_and_reports_every_fold(tmp_path):
    _, study = T.tune(*FAST["reg"][:1], "reg", 6, storage=f"sqlite:///{tmp_path / 's.db'}")
    assert isinstance(study.sampler, optuna.samplers.TPESampler)
    assert isinstance(study.pruner, optuna.pruners.MedianPruner)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert all(sorted(t.intermediate_values) == [0, 1, 2, 3, 4] for t in done)
    assert {"MAE", "MSESS", "RMSE", "bias", "MAE_wet"} <= set(study.best_trial.user_attrs)


def test_tuning_code_never_unlocks_the_test_getter():
    src = (ROOT / "src/tuning.py").read_text()
    assert "test_index" not in src and "confirm_final" not in src
