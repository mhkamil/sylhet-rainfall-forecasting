"""Stage 5 deep time-series models with darts 0.48.0 (TSMixer, TiDE; TFT follows in part 2).

Data: daily series up to 2019-12-31 only (test years are never loaded).
Target: PRECTOTCORR in mm, or log(1 + mm). Past covariates: the other nine Set-B weather variables.
Known-future covariates: day-of-year sin/cos (harmonics 1 and 2) -- the calendar of the forecast day is known in advance.
Split (DL split of src/splits.py, by target date): training targets 1981-01-01 .. 2014-12-30 (1-day purge),
validation targets 2015-01-01 .. 2019-12-31.

Protocol (same logic as the classical runner):
 1. inner fit on targets <= 2011-12-30 with early stopping on the slice 2012-01-01 .. 2014-12-30 (last 3 training years);
    best weights restored -> best epoch; for log targets, Duan smearing factor from slice residuals;
 2. refit on all training targets (<= 2014-12-30) for exactly the best number of epochs, scalers refitted on those rows;
 3. one-step-ahead forecasts for every validation day from its true observed lookback window (historical_forecasts,
    retrain=False). The validation period is only scored (its loss is logged for the learning-curve plot, never used).
"""
import json
import logging
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TiDEModel, TSMixerModel
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC, DEEP_DIR = ROOT / "data/processed", ROOT / "models/deep"
CKPT_DIR = DEEP_DIR / "_checkpoints"
TARGET = "PRECTOTCORR"
PAST = ["QV2M", "RH2M", "T2M", "T2M_MAX", "T2M_MIN", "T2MDEW", "PS", "WS2M", "WS10M"]
FUTURE = ["doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"]
DEV_END = pd.Timestamp("2019-12-31")
INNER_LAST, SLICE_FIRST = pd.Timestamp("2011-12-30"), pd.Timestamp("2012-01-01")
TRAIN_LAST, VAL_FIRST = pd.Timestamp("2014-12-30"), pd.Timestamp("2015-01-01")
DAY = pd.Timedelta(days=1)
LOOKBACKS = [14, 30, 60]
MAX_EPOCHS, PATIENCE, BATCH = 60, 8, 256
RESULTS = ROOT / "reports/tables/deep_results.csv"
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*LeafSpec.*")
for _name in ("pytorch_lightning", "lightning", "lightning.pytorch", "lightning_fabric", "darts"):
    logging.getLogger(_name).setLevel(logging.ERROR)


# ---------------- data ----------------
def load_daily():
    """Clean daily data truncated at 2019-12-31 and the known-future calendar covariates."""
    c = pd.read_parquet(PROC / "clean_daily.parquet").loc[:DEV_END, [TARGET] + PAST]
    doy = c.index.dayofyear.to_numpy()
    fut = pd.DataFrame({f"doy_{f}{k}": fn(2 * np.pi * k * doy / 365.25) for k in (1, 2)
                        for f, fn in (("sin", np.sin), ("cos", np.cos))}, index=c.index)[FUTURE]
    return c, fut


def upto(ts, last):
    """Series up to and including `last` (inclusive slice)."""
    return ts.slice(ts.start_time(), last)


def to_series(df):
    return TimeSeries.from_dataframe(df.astype("float32"), freq="D")


def prepare(transform, last_train, df=None, fut=None):
    """Scaled darts series. Scalers (StandardScaler) are fitted on rows <= last_train only."""
    if df is None:
        df, fut = load_daily()
    y = np.log1p(df[[TARGET]]) if transform == "log" else df[[TARGET]]
    target, past, future = to_series(y), to_series(df[PAST]), to_series(fut)
    ys, ps = Scaler(StandardScaler()), Scaler(StandardScaler())
    ys.fit(upto(target, last_train))
    ps.fit(upto(past, last_train))
    return {"target": ys.transform(target), "past": ps.transform(past), "future": future, "y_scaler": ys,
            "raw": df[TARGET]}


def window(df, fut, origin, L):
    """The exact inputs of the forecast made at the end of day `origin` for day origin+1 (for illustration/tests)."""
    lo = origin - pd.Timedelta(days=L - 1)
    return {"target_and_past": df.loc[lo:origin], "future": fut.loc[lo:origin + DAY],
            "forecast_for": origin + DAY}


# ---------------- models ----------------
class History(pl.Callback):
    """Records the mean training loss (average over all batches of the epoch; darts logs `train_loss` per step only)
    and the epoch validation loss."""
    def __init__(self):
        self.train, self.val, self._batches = [], [], []

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self._batches.append(float(loss))

    def on_validation_epoch_end(self, trainer, module):
        if not trainer.sanity_checking and "val_loss" in trainer.callback_metrics:
            self.val.append(float(trainer.callback_metrics["val_loss"]))

    def on_train_epoch_end(self, trainer, module):
        if self._batches:
            self.train.append(float(np.mean(self._batches)))
            self._batches = []


def make_model(kind, L, seed, n_epochs, callbacks, name, checkpoints=False):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    common = dict(input_chunk_length=L, output_chunk_length=1, batch_size=BATCH, n_epochs=n_epochs, dropout=0.1,
                  optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-4}, random_state=seed, model_name=name,
                  work_dir=str(CKPT_DIR), save_checkpoints=checkpoints, force_reset=True,
                  pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False, "enable_model_summary": False,
                                     "logger": False, "callbacks": callbacks})
    if kind == "tsmixer":
        return TSMixerModel(hidden_size=32, ff_size=32, num_blocks=2, **common)
    if kind == "tide":
        return TiDEModel(num_encoder_layers=1, num_decoder_layers=1, decoder_output_dim=8, hidden_size=64,
                         temporal_width_past=4, temporal_width_future=0,   # 0: use the 4 calendar features raw temporal_decoder_hidden=16,
                         use_layer_norm=True, **common)
    raise ValueError(kind)


def one_step(model, d, first, last):
    """One-step-ahead forecasts for target dates first..last, each from the true observed lookback window."""
    fc = model.historical_forecasts(series=upto(d["target"], last), past_covariates=d["past"],
                                    future_covariates=d["future"], start=first, forecast_horizon=1, stride=1,
                                    retrain=False, last_points_only=True, verbose=False, show_warnings=False)
    return d["y_scaler"].inverse_transform(fc).to_series()


def run(kind, transform, L, seed, save=True):
    """Inner fit (early stopping + smearing) -> refit -> validation forecasts. Returns (predictions, info)."""
    name, t0 = f"{kind}_{transform}_L{L}_s{seed}", time.time()
    # 1. inner fit with early stopping on the slice, best weights restored
    d = prepare(transform, INNER_LAST)
    h1 = History()
    m1 = make_model(kind, L, seed, MAX_EPOCHS, [h1, EarlyStopping("val_loss", patience=PATIENCE, min_delta=1e-4)],
                    name + "_inner", checkpoints=True)
    val_start = SLICE_FIRST - pd.Timedelta(days=L)
    m1.fit(upto(d["target"], INNER_LAST), past_covariates=d["past"], future_covariates=d["future"],
           val_series=d["target"].slice(val_start, TRAIN_LAST), val_past_covariates=d["past"],
           val_future_covariates=d["future"], load_best=True, verbose=False)
    best_epoch = int(np.argmin(h1.val)) + 1
    smear = np.nan
    if transform == "log":
        z_hat = one_step(m1, d, SLICE_FIRST, TRAIN_LAST)
        z = np.log1p(d["raw"].loc[z_hat.index])
        smear = float(np.mean(np.exp(z - z_hat)))
    # 2. refit on all training targets for best_epoch epochs (validation loss only logged for the curve)
    d = prepare(transform, TRAIN_LAST)
    h2 = History()
    m2 = make_model(kind, L, seed, best_epoch, [h2], name)
    m2.fit(upto(d["target"], TRAIN_LAST), past_covariates=d["past"], future_covariates=d["future"],
           val_series=d["target"].slice(VAL_FIRST - pd.Timedelta(days=L), DEV_END), val_past_covariates=d["past"],
           val_future_covariates=d["future"], verbose=False)
    # 3. one-step forecasts for every validation target date
    out = one_step(m2, d, VAL_FIRST, DEV_END).rename("raw_pred").to_frame()
    out["y_true"] = d["raw"].loc[out.index]
    if transform == "log":
        out["pred_naive"] = np.clip(np.expm1(out["raw_pred"]), 0, None)
        out["pred"] = np.clip(np.exp(out["raw_pred"]) * smear - 1, 0, None)
    else:
        out["pred"] = np.clip(out["raw_pred"], 0, None)
    out.index.name = "target_date"
    out = out.set_index(out.index - DAY).rename_axis("date")         # feature date t, like models/oof
    info = {"best_epoch": best_epoch, "smear_factor": smear, "fit_seconds": time.time() - t0, "inner_train": h1.train,
            "inner_val": h1.val, "refit_train": h2.train, "refit_val": h2.val}
    if save:
        DEEP_DIR.mkdir(parents=True, exist_ok=True)
        stem = DEEP_DIR / f"{kind}_{transform}_L{L}__seed{seed}"
        out.to_parquet(f"{stem}.parquet")
        Path(f"{stem}_history.json").write_text(json.dumps(info))
    shutil.rmtree(CKPT_DIR / (name + "_inner"), ignore_errors=True)
    return out, info


# ---------------- scoring (same metrics and references as the classical models, DL split) ----------------
def dl_references():
    """Validation index (feature dates), observed mm and the DL-split climatology / seasonal+persistence forecasts."""
    import experiment as E
    import splits as S
    X, Y = E.load("A")
    tr, va = S.dl_split(X.index)
    return va, Y.loc[va, "y_reg"], E.references("reg", X, Y, E.rain_today(X.index), tr, va)


def score_and_store(out, info, kind, transform, L, seed, refs_bundle, save=True):
    """Tidy rows (model, transform, lookback, seed, metric, value) for smeared/raw and, for log, naive predictions."""
    import experiment as E
    va, y, refs = refs_bundle
    assert out.index.equals(va) and np.allclose(out["y_true"], y), "predictions are not aligned with the DL split"
    rows = []
    variants = {"pred": transform} if transform == "mm" else {"pred": "log_smear", "pred_naive": "log_naive"}
    for col, label in variants.items():
        m = E.score("reg", y, out[col].to_numpy(), np.nan, refs, va)
        m.update(best_epoch=info["best_epoch"], smear_factor=info["smear_factor"], fit_seconds=info["fit_seconds"])
        rows += [{"model": kind, "target": label, "lookback": L, "seed": seed, "metric": k, "value": v} for k, v in m.items()]
    res = pd.DataFrame(rows)
    if save:
        keys = ["model", "target", "lookback", "seed"]
        table = res
        if RESULTS.exists():
            old = pd.read_csv(RESULTS)
            k = lambda d: d[keys].astype(str).agg("|".join, axis=1)
            table = pd.concat([old[~k(old).isin(set(k(res)))], res], ignore_index=True)
        table.to_csv(RESULTS, index=False)
    return res                                                     # only this run's rows
