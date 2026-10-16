"""Unit tests for src/splits.py. Run from the project root: python -m pytest -q tests/"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import splits as S  # noqa: E402

IDX = pd.read_parquet(ROOT / "data/processed/features_A.parquet", columns=["y_cls"]).index
ALL_SPLITS = S.cv_folds(IDX) + [S.dl_split(IDX)]


def test_five_folds_with_expected_validation_years():
    folds = S.cv_folds(IDX)
    assert len(folds) == 5
    for (tr, va), (_, v0, v1) in zip(folds, S.CV_FOLDS):
        ty = S.target_dates(va).year
        assert ty.min() == v0 and ty.max() == v1 and len(set(ty)) == v1 - v0 + 1


@pytest.mark.parametrize("k", range(len(ALL_SPLITS)))
def test_no_overlap_and_chronological(k):
    tr, va = ALL_SPLITS[k]
    assert len(tr.intersection(va)) == 0
    assert tr.max() < va.min()
    assert tr.is_monotonic_increasing and va.is_monotonic_increasing


@pytest.mark.parametrize("k", range(len(ALL_SPLITS)))
def test_gap_between_last_train_target_and_first_val_target(k):
    tr, va = ALL_SPLITS[k]
    gap = (S.target_dates(va).min() - S.target_dates(tr).max()).days - 1
    assert gap >= S.GAP_DAYS


@pytest.mark.parametrize("k", range(len(ALL_SPLITS)))
def test_no_split_touches_test_targets(k):
    tr, va = ALL_SPLITS[k]
    assert S.target_dates(tr.union(va)).year.max() <= 2019


def test_expanding_window():
    trains = [tr for tr, _ in S.cv_folds(IDX)]
    assert all(a.min() == b.min() and len(a) < len(b) for a, b in zip(trains, trains[1:]))


def test_dev_index_stops_before_2020_targets():
    dev = S.dev_index(IDX)
    assert S.target_dates(dev).max() == pd.Timestamp("2019-12-31")


def test_test_set_is_locked():
    for flag in (False, None, "yes", 1):
        with pytest.raises(PermissionError):
            S.test_index(IDX, confirm_final=flag)
    te = S.test_index(IDX, confirm_final=True)
    assert S.target_dates(te).year.min() == 2020 and len(te.intersection(S.dev_index(IDX))) == 0


def test_validation_beyond_2019_is_refused():
    with pytest.raises(ValueError):
        S.make_split(IDX, 2018, 2019, 2020)
