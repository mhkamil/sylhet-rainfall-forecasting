"""Leakage tests for src/features.py. Run from the project root: python -m pytest -q tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import features as F  # noqa: E402

CLEAN = pd.read_parquet(ROOT / "data/processed/clean_daily.parquet")
DATES = sorted(pd.DatetimeIndex(np.random.default_rng(42).choice(CLEAN.index[100:-100], size=3, replace=False)))


@pytest.mark.parametrize("fset", ["A", "B"])
@pytest.mark.parametrize("cutoff", DATES)
def test_truncation(fset, cutoff):
    """Features up to day d must not change when all data after d are removed."""
    full = F.build_features(CLEAN, fset)[0].loc[:cutoff]
    part = F.build_features(CLEAN.loc[:cutoff], fset)[0]
    pd.testing.assert_frame_equal(full, part)


def test_leaky_feature_is_detected():
    """Negative control: a centred window uses day t+1 and must fail the truncation idea."""
    leaky = lambda c: np.log1p(c["PRECTOTCORR"]).rolling(3, center=True).mean()
    d = DATES[0]
    assert not leaky(CLEAN).loc[:d].equals(leaky(CLEAN.loc[:d]))


def test_target_is_next_calendar_day():
    _, _, Y, _, _ = F.assemble(CLEAN)
    nxt = CLEAN["PRECTOTCORR"].reindex(Y.index + pd.Timedelta(days=1)).to_numpy()
    assert np.array_equal(Y["y_reg"].to_numpy(), nxt)
    for name, thr in F.THRESHOLDS.items():
        assert (Y[name] == (Y["y_reg"] >= thr).astype(int)).all()


@pytest.mark.parametrize("fset", ["A", "B"])
def test_saved_features_match_code_and_contain_no_target(fset):
    saved = pd.read_parquet(ROOT / f"data/processed/features_{fset}.parquet")
    XA, XB, Y, _, _ = F.assemble(CLEAN)
    X = XA if fset == "A" else XB
    pd.testing.assert_frame_equal(saved[X.columns], X)
    for c in X:
        assert not np.array_equal(X[c].to_numpy(), Y["y_reg"].to_numpy()), c
