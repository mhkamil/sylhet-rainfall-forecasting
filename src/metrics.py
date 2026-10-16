"""Stage 3 evaluation metrics for next-day rain occurrence (probabilities) and amount (mm)."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

WET_MM = 1.0
HEAVY_MM = (20.0, 50.0)


def _div(a, b):
    return a / b if b else np.nan


# ---------- classification ----------
def confusion(y, yhat):
    y, yhat = np.asarray(y).astype(bool), np.asarray(yhat).astype(bool)
    return {"tp": int((y & yhat).sum()), "fp": int((~y & yhat).sum()),
            "fn": int((y & ~yhat).sum()), "tn": int((~y & ~yhat).sum())}


def hard_metrics(y, yhat):
    """Scores for yes/no forecasts. FAR is the false alarm RATIO fp/(tp+fp) (meteorological convention)."""
    c = confusion(y, yhat)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn
    pod, prec = _div(tp, tp + fn), _div(tp, tp + fp)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return {**c, "accuracy": (tp + tn) / n,
            "balanced_accuracy": np.nanmean([pod, _div(tn, tn + fp)]),
            "precision": prec, "recall_POD": pod,
            "F1": _div(2 * tp, 2 * tp + fp + fn), "FAR": _div(fp, tp + fp),
            "CSI": _div(tp, tp + fp + fn), "HSS": _div(2 * (tp * tn - fp * fn), hss_den)}


def ece(y, p, n_bins=10):
    """Expected calibration error with equal-width probability bins (weighted |observed freq - mean prob|)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return sum((bins == b).mean() * abs(y[bins == b].mean() - p[bins == b].mean())
               for b in range(n_bins) if (bins == b).any())


def calibration(y, p, n_bins=10):
    """Reliability-diagram data: observed frequency vs mean forecast probability per bin."""
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_frequency": frac_pos})


def classification_metrics(y, p, threshold=0.5, p_ref=None):
    """Probabilistic scores + hard scores at `threshold`. BSS uses p_ref (e.g. climatology), else the sample base rate."""
    y, p = np.asarray(y, int), np.clip(np.asarray(p, float), 0, 1)
    brier = np.mean((p - y) ** 2)
    ref = np.full_like(p, y.mean()) if p_ref is None else np.asarray(p_ref, float)
    one_class = len(np.unique(y)) < 2
    out = {"ROC_AUC": np.nan if one_class else roc_auc_score(y, p),
           "PR_AUC": np.nan if one_class else average_precision_score(y, p),
           "Brier": brier, "BSS": 1 - brier / np.mean((ref - y) ** 2),
           "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]), "ECE": ece(y, p)}
    return {**out, **hard_metrics(y, p >= threshold), "threshold": threshold}


def choose_threshold(y, p, metric="CSI", grid=np.round(np.arange(0.01, 1.0, 0.01), 2)):
    """Probability threshold maximising CSI or F1 on the data given (use validation/training data, never test).
    The returned value is then applied unchanged to later data."""
    scores = [hard_metrics(y, np.asarray(p) >= t)[metric] for t in grid]
    return float(grid[int(np.nanargmax(scores))])


# ---------- regression ----------
def _corr(f, a, b):
    return np.nan if np.ptp(a) == 0 or np.ptp(b) == 0 else f(a, b)[0]


def regression_metrics(y, yhat, y_ref=None, wet_mm=WET_MM, heavy=HEAVY_MM):
    """Scores for rainfall amounts in mm. MSESS uses y_ref (e.g. climatology), else the sample mean."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = yhat - y
    mse = np.mean(err ** 2)
    ref = np.full_like(y, y.mean()) if y_ref is None else np.asarray(y_ref, float)
    wet = y >= wet_mm
    out = {"MAE": np.mean(np.abs(err)), "RMSE": np.sqrt(mse), "bias": np.mean(err),
           "pearson_r": _corr(stats.pearsonr, y, yhat), "spearman_rho": _corr(stats.spearmanr, y, yhat),
           "R2": 1 - mse / np.var(y), "MSESS": 1 - mse / np.mean((ref - y) ** 2),
           "MAE_wet": np.mean(np.abs(err[wet])), "RMSE_wet": np.sqrt(np.mean(err[wet] ** 2))}
    for h in heavy:
        c = confusion(y >= h, yhat >= h)
        out[f"recall_ge{int(h)}"] = _div(c["tp"], c["tp"] + c["fn"])
        out[f"precision_ge{int(h)}"] = _div(c["tp"], c["tp"] + c["fp"])
    return out


# ---------- uncertainty ----------
def block_bootstrap(metric_fn, dates, *arrays, n_boot=1000, alpha=0.05, seed=42):
    """Percentile CI for metric_fn(*arrays) resampling whole calendar year-month blocks with replacement
    (keeps day-to-day autocorrelation inside blocks). Returns (estimate, lower, upper)."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    block = pd.DatetimeIndex(dates).to_period("M").astype(str).to_numpy()
    ids = np.unique(block)
    rows = [np.flatnonzero(block == b) for b in ids]
    boot = []
    for _ in range(n_boot):
        take = np.concatenate([rows[i] for i in rng.integers(0, len(ids), len(ids))])
        boot.append(metric_fn(*[a[take] for a in arrays]))
    lo, hi = np.nanpercentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return metric_fn(*arrays), lo, hi
