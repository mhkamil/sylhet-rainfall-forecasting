"""Leakage tests for the deep-learning data pipeline (src/deep_ts.py). Run: python -m pytest -q tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import deep_ts as D  # noqa: E402

L = 14


@pytest.fixture(scope="module")
def tiny():
    """A small TSMixer trained for 2 epochs on 2005-2012 targets (only to have a working model)."""
    d = D.prepare("log", D.TRAIN_LAST)
    m = D.make_model("tsmixer", L, 0, 2, [], "pytest_tiny")
    m.fit(d["target"].slice(pd.Timestamp("2005-01-01"), pd.Timestamp("2012-12-31")),
          past_covariates=d["past"], future_covariates=d["future"], verbose=False)
    hf = D.one_step(m, d, pd.Timestamp("2016-01-01"), pd.Timestamp("2016-12-31"))
    return m, d, hf


def test_test_years_are_never_loaded():
    df, fut = D.load_daily()
    assert df.index.max() == fut.index.max() == pd.Timestamp("2019-12-31")


def test_window_contains_only_days_up_to_origin():
    df, fut = D.load_daily()
    for L_ in D.LOOKBACKS:
        w = D.window(df, fut, pd.Timestamp("2016-07-15"), L_)
        assert len(w["target_and_past"]) == L_ and w["target_and_past"].index.max() == pd.Timestamp("2016-07-15")
        assert w["future"].index.max() == w["forecast_for"] == pd.Timestamp("2016-07-16")


def test_scalers_are_fitted_on_training_rows_only():
    df, _ = D.load_daily()
    d = D.prepare("mm", D.TRAIN_LAST)
    mean = d["y_scaler"]._fitted_params[0].mean_[0]
    assert np.isclose(mean, df.loc[:D.TRAIN_LAST, D.TARGET].mean(), rtol=1e-5)
    assert not np.isclose(mean, df[D.TARGET].mean(), rtol=1e-3)          # would differ if 2015-2019 were included


@pytest.mark.parametrize("origin", ["2016-02-10", "2016-07-15", "2016-11-30"])
def test_one_step_forecast_equals_forecast_from_truncated_inputs(tiny, origin):
    """The validation forecast for origin+1 must equal a forecast made with all data after the origin physically
    removed (future calendar only up to origin+1): no future information, and the true observed lookback is used."""
    m, d, hf = tiny
    o = pd.Timestamp(origin)
    p = m.predict(n=1, series=D.upto(d["target"], o), past_covariates=D.upto(d["past"], o),
                  future_covariates=D.upto(d["future"], o + D.DAY), verbose=False)
    p = d["y_scaler"].inverse_transform(p).to_series()
    assert p.index[0] == o + D.DAY
    assert np.isclose(p.iloc[0], hf.loc[o + D.DAY], atol=1e-5)


def test_forecast_reacts_to_the_observed_value_at_the_origin(tiny):
    """Changing the *observed* rain on the origin day changes the next-day forecast -> inputs are observations,
    not the model's own earlier predictions (no recursion)."""
    m, d, _ = tiny
    o = pd.Timestamp("2016-07-15")
    base = D.upto(d["target"], o)
    vals = base.values().copy()
    vals[-1] += 3.0                                                    # +3 std on the origin day
    bumped = base.with_values(vals)
    kw = dict(n=1, past_covariates=D.upto(d["past"], o), future_covariates=D.upto(d["future"], o + D.DAY), verbose=False)
    assert not np.isclose(m.predict(series=base, **kw).values()[0, 0], m.predict(series=bumped, **kw).values()[0, 0])
