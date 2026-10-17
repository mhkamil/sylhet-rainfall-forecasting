"""Tests for the Stage-7 freeze (src/final_eval.py). Everything here uses development data only (pseudo-test windows inside
1981-2019) or synthetic numbers; the real test years are never loaded. Run: python -m pytest -q tests/test_final_eval.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment as E  # noqa: E402
import final_eval as FE  # noqa: E402
import splits as S  # noqa: E402
import tuning as T  # noqa: E402

LAST_DEV = pd.Timestamp("2019-12-31")


@pytest.fixture(scope="module")
def cfg():
    return FE.load_config()


# ---------- the lock ----------
def test_run_refuses_without_explicit_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(FE, "TEST_DIR", tmp_path / "test")
    for bad in (False, None, "yes", 1):
        with pytest.raises(PermissionError):
            FE.run(confirm_final=bad)
    assert not (tmp_path / "test").exists()


def test_no_test_predictions_exist_before_stage_7_2():
    """Every file in models/test must be listed in a manifest written by a confirmed run; before 7.2 there is none."""
    files = [f for f in FE.TEST_DIR.glob("*") if f.name not in (".gitkeep", "README.md")] if FE.TEST_DIR.exists() else []
    manifest = FE.TEST_DIR / "manifest.json"
    if not manifest.exists():
        assert files == []
    else:
        listed = set(json.loads(manifest.read_text())["files"])
        assert {f.stem for f in files if f.suffix == ".parquet"} <= listed


def test_development_rows_end_before_the_test_years():
    X, Y, tr = FE.development()
    assert S.target_dates(X.index).max() == LAST_DEV
    assert S.target_dates(tr).max() == pd.Timestamp("2019-12-30")          # 1-day purge
    assert len(X) - len(tr) == 1


# ---------- the frozen configuration ----------
def test_config_matches_the_tuning_results_and_seeds(cfg):
    assert cfg["seeds"]["stochastic_models"] == [0, 1, 2, 3, 4]
    assert cfg["data"]["n_features"] == 115 and len(cfg["data"]["features"]) == 115
    assert cfg["data"]["test"]["target_dates"] == ["2020-01-01", "2025-12-31"]
    for task, names in cfg["models"]["classical"].items():
        assert set(names) == set(FE.CLASSICAL[task])
        for name, entry in names.items():
            assert entry["seeds"] == FE.seeds_for(task, name)
            if entry["config"] == "tuned":
                assert entry["params"] == T.load_best(name, task)["best_params"]
            for s in entry["seeds"]:
                fr = entry["frozen"][s]
                if task == "cls":
                    assert 0 < fr["threshold"] < 1
                if name in ("xgboost", "lightgbm", "catboost", "catboost_tweedie", "mlp", "mlp_huber") or \
                        name.startswith("hurdle"):
                    assert fr["n_trees"] >= 1
    for task, kinds in cfg["models"]["deep"].items():
        for kind, entry in kinds.items():
            assert set(entry["frozen"]) == set(FE.SEEDS) and all(v["best_epoch"] >= 1 for v in entry["frozen"].values())
    assert cfg["data"]["inner_split_for_frozen_values"]["slice_target_dates"] == ["2017-01-01", "2019-12-30"]


def test_frozen_values_are_reproducible_from_development_data(cfg):
    """Re-deriving two frozen entries gives exactly the stored numbers (the freeze is deterministic)."""
    for key in [("cls", "logreg_l1", 0), ("cls", "lightgbm", 0)]:
        _, out = FE.freeze_classical(key)
        assert out == cfg["models"]["classical"]["cls"][key[1]]["frozen"][key[2]]


def test_integrity_guard_detects_changed_data(cfg, monkeypatch):
    bad = json.loads(json.dumps(cfg))
    bad["freeze"]["data_sha256"]["data/processed/features_B.parquet"] = "0" * 64
    monkeypatch.setattr(FE, "_git", lambda *a: "")                     # pretend the tree is clean
    with pytest.raises(RuntimeError, match="differs from the frozen data"):
        FE.check_integrity(bad)


# ---------- prediction primitives on a development-only pseudo-test window ----------
@pytest.fixture(scope="module")
def pseudo():
    """Train on targets <= 2015 (purged), 'test' = targets 2016-2019. Purely development data."""
    X, Y, _ = FE.development()
    ty = S.target_dates(X.index).year
    tr, te = X.index[ty <= 2015][:-1], X.index[ty >= 2016]
    return X, Y, tr, te


@pytest.mark.parametrize("task,name", [("cls", "logreg_l1"), ("cls", "lightgbm"), ("reg", "catboost_tweedie")])
def test_refit_uses_only_training_rows_and_never_test_labels(pseudo, task, name):
    X, Y, tr, te = pseudo
    y = Y[E.TARGET[task]]
    frozen = {"threshold": 0.5, "n_trees": 50}
    a = FE.predict_classical(task, name, 0, X, y, tr, te, frozen)
    y2 = y.copy()
    y2.loc[te] = np.random.default_rng(0).permutation(y2.loc[te].to_numpy())      # scramble the pseudo-test labels
    b = FE.predict_classical(task, name, 0, X, y2, tr, te, frozen)
    assert a.index.equals(te) and np.allclose(a["pred"], b["pred"])


def test_hurdle_combines_frozen_probability_and_amount(pseudo):
    X, Y, tr, te = pseudo
    p = np.full(len(te), 0.5)
    out = FE.predict_classical("reg", "hurdle_catboost_log_msess", 0, X, Y["y_reg"], tr, te,
                               {"n_trees": 50, "smear_factor": 1.5}, p_wet=p)
    assert np.allclose(out["pred"], 0.5 * out["amount_if_wet"]) and (out["amount_if_wet"] >= 1.0).all()


def test_deep_refit_never_uses_targets_after_its_training_window():
    """Pseudo-final TSMixer: train <= 2015-12-30, forecast 2016-01-01..2016-01-10. Scrambling every target from 2016 on
    leaves the forecast for the first day (whose inputs end on 2015-12-31) unchanged -> the fitted weights never saw them."""
    import deep_ts as D
    df, fut = D.load_daily()
    df, fut = df.loc[:"2016-01-10"], fut.loc[:"2016-01-10"]
    dates = {"train_last": pd.Timestamp("2015-12-30"), "val_first": pd.Timestamp("2016-01-01"),
             "val_last": pd.Timestamp("2016-01-10")}
    mm = lambda d: D.run("tsmixer", "mm", 14, 0, save=False, variant={"df": d, "fut": fut}, dates=dates,
                         refit_only={"best_epoch": 2}, log_val=False)[0]
    base = mm(df)
    scrambled = df.copy()
    scrambled.loc["2016-01-01":, D.TARGET] = 500.0
    other = mm(scrambled)
    assert base.index[0] == pd.Timestamp("2015-12-31") and len(base) == 10
    assert np.isclose(base["pred"].iloc[0], other["pred"].iloc[0])


# ---------- pre-registered statistics ----------
def test_dm_test_on_synthetic_losses():
    rng = np.random.default_rng(1)
    e = rng.normal(size=2000)
    assert FE.dm_test(e ** 2, e ** 2)[2] == 1.0                                 # identical losses
    a, b = np.abs(e), np.abs(e) + 0.2 + 0.05 * rng.normal(size=2000)
    mean, stat, p = FE.dm_test(a, b)
    assert mean < 0 and stat < 0 and p < 1e-6                                 # a clearly better
    null = [FE.dm_test(rng.normal(size=500), rng.normal(size=500))[2] for _ in range(300)]
    assert 0.02 < np.mean(np.array(null) < 0.05) < 0.10                         # ~ 5 % false positives under H0


def test_holm_matches_hand_computation():
    p = [0.01, 0.04, 0.03, 0.005]
    # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone 0.06
    assert np.allclose(FE.holm(p), [0.03, 0.06, 0.06, 0.02])


def test_only_the_final_run_and_its_analysis_unlock_the_test_getter():
    """Stage 7.2 addition: src/final_report.py reads the test labels, but only after the confirmed run's manifest exists."""
    import reporting as Rep
    unlock = Rep.test_set_audit()["files calling test_index(..., confirm_final=True)"]
    assert set(unlock) <= {"tests/test_splits.py", "src/final_eval.py", "src/final_report.py"}


def test_report_refuses_test_labels_without_the_manifest(tmp_path, monkeypatch):
    import final_report as FR
    monkeypatch.setattr(FR, "TEST_DIR", tmp_path)
    with pytest.raises(PermissionError):
        FR.truth()
