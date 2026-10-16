"""Tests for the Stage-6.4 ablation data (src/ablation.py, src/deep_ablation.py). Run: python -m pytest -q tests/test_ablation.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ablation as A  # noqa: E402
import deep_ablation as DA  # noqa: E402
import splits as S  # noqa: E402

LAST = pd.Timestamp("2019-12-31")


@pytest.mark.parametrize("name", list(A.VARIANTS))
def test_every_variant_stays_in_the_development_period(name):
    X, Y, rain, season = A.variant_data(name)
    assert S.target_dates(X.index).max() <= LAST
    assert X.notna().all().all() and rain.notna().all()
    for _, tr, va, tr_full in A.folds(name, X.index):
        assert S.target_dates(tr).max() < S.target_dates(va).min() and tr.isin(tr_full).all()


@pytest.mark.parametrize("name,start", [("win_2000", 2000), ("win_2010", 2010), ("B_1984", 1984)])
def test_training_window_is_applied_and_references_keep_the_full_fold(name, start):
    X = A.variant_data(name)[0]
    for _, tr, _, tr_full in A.folds(name, X.index):
        assert S.target_dates(tr).year.min() == start and S.target_dates(tr_full).year.min() == 1981


def test_dropping_a_family_removes_exactly_its_columns():
    full = set(A.variant_data("full")[0].columns)
    for fam, fams in A.FAMILIES.items():
        gone = full - set(A.variant_data(f"drop_{fam}")[0].columns)
        expected = set(A.DICT.loc[A.DICT.family.isin(fams) & A.DICT.set_B, "feature"])
        assert gone == expected and len(gone) > 0


def test_imerg_variant_uses_imerg_for_features_and_next_day_targets():
    X, Y, rain, _ = A.variant_data("imerg_1998")
    imerg = pd.read_parquet(ROOT / "data/processed/clean_daily.parquet")["IMERG_PRECTOT"]
    t = X.index[100]
    assert np.isclose(Y.loc[t, "y_reg"], imerg.loc[t + pd.Timedelta(days=1)])
    assert np.isclose(X.loc[t, "rain_log1p"], np.log1p(imerg.loc[t])) and np.isclose(rain.loc[t], imerg.loc[t])
    assert X.index.min() >= pd.Timestamp("1998-01-01")


def test_deep_variants_never_load_test_years():
    for name in DA.VARIANTS:
        v = DA.variant_input(name)
        assert v["df"].index.max() <= LAST and v["fut"].index.max() <= LAST
