"""Tests for the shared CV runner (src/experiment.py). Run: python -m pytest -q tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment as E  # noqa: E402
import splits as S  # noqa: E402


class Spy:
    """Classifier that predicts the training base rate and records which rows it was fitted on."""
    fitted = []

    def fit(self, X, y):
        Spy.fitted.append(X.index)
        self.rate = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.rate), np.full(len(X), self.rate)])

    def predict(self, X):
        return np.full(len(X), self.rate)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("exp")
    Spy.fitted = []
    res, oof, log = E.run_cv("spy", lambda s, y: Spy(), "cls", "A", seed=0,
                             results_path=tmp / "cv.csv", oof_dir=tmp / "oof")
    return res, oof, log, tmp, list(Spy.fitted)


def test_every_fit_uses_only_rows_before_the_validation_block(run):
    _, _, _, _, fitted = run
    folds = S.cv_folds(E.load("A")[0].index)
    assert len(fitted) == 2 * len(folds)                       # inner fit + refit per fold
    for k, (tr, va) in enumerate(folds):
        for idx in fitted[2 * k: 2 * k + 2]:
            assert len(idx.intersection(va)) == 0
            assert S.target_dates(idx).max() < S.target_dates(va).min()
        assert fitted[2 * k + 1].equals(tr)                     # final model sees the whole training fold


def test_threshold_slice_lies_inside_training_fold(run):
    _, _, log, _, _ = run
    folds = S.cv_folds(E.load("A")[0].index)
    for (_, row), (tr, va) in zip(log.iterrows(), folds):
        last_slice = pd.Timestamp(row["slice_targets"].split("..")[1])
        assert last_slice <= S.target_dates(tr).max() < S.target_dates(va).min()


def test_oof_predictions_cover_exactly_the_validation_blocks(run):
    _, oof, _, _, _ = run
    folds = S.cv_folds(E.load("A")[0].index)
    assert oof.index.equals(folds[0][1].append([va for _, va in folds[1:]]))
    assert not oof.index.duplicated().any()


def test_constant_forecast_has_no_ranking_skill_and_matches_majority_baseline(run):
    res, _, _, _, _ = run
    auc = res.loc[res.metric == "ROC_AUC", "value"]
    assert np.allclose(auc, 0.5)
    base = E.run_baselines("cls", save=False)
    bss_major = base[(base.model == "baseline:majority") & (base.metric == "BSS")].sort_values("fold")["value"]
    bss_spy = res[res.metric == "BSS"].sort_values("fold")["value"]
    assert np.allclose(bss_major.to_numpy(), bss_spy.to_numpy())   # same forecast -> same score


def test_tidy_format_and_no_duplicates_on_rerun(run):
    res, _, _, tmp, _ = run
    assert list(res.columns) == ["model", "task", "feature_set", "fold", "seed", "metric", "value"]
    n = len(pd.read_csv(tmp / "cv.csv"))
    E.append_results(res, tmp / "cv.csv")
    assert len(pd.read_csv(tmp / "cv.csv")) == n
    assert (tmp / "oof" / "spy__cls__A__seed0.parquet").exists()


def test_inner_split_is_chronological_with_gap():
    tr, _ = S.cv_folds(E.load("A")[0].index)[0]
    itr, sl = E.inner_split(tr)
    assert S.target_dates(itr).max() < S.target_dates(sl).min()
    assert (S.target_dates(sl).min() - S.target_dates(itr).max()).days - 1 >= S.GAP_DAYS
    assert set(S.target_dates(sl).year) == {2002, 2003, 2004}
