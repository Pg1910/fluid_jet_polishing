"""
Evaluation harness.

Every model in this project goes through this file. That is the point:
comparisons are only meaningful when the protocol is held fixed.

Three protocols, each answering a different question:

  repeated_kfold  "How well does it predict a new run at a condition
                   similar to ones we have already sprayed?"
                  Repeated because with n=40 a single split is noise.

  loo             Leave-one-out. Same question, but each model sees 39 of
                  40 samples, so it is the least pessimistic estimate
                  available. Deterministic -- no seed, no error bar.
                  Caveat: LOO has low bias but high variance, and with a
                  small n its R2 can look erratic. Read it with MAE.

  grouped_sod     "How well does it predict a stand-off distance we never
                   tested?" Holds out one entire SOD level at a time.
                  This is extrapolation, and it is the honest answer to
                  the question a process engineer actually asks.

Leakage control: models must be sklearn Pipelines. A Pipeline refits every
step (scaler, polynomial expansion, the estimator) inside each fold, so
test-fold statistics never influence training. If you scale the full X
before calling this, you have already leaked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    RepeatedKFold,
    LeaveOneOut,
    LeaveOneGroupOut,
    cross_val_predict,
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------- metrics

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the three metrics we report everywhere.

    R2   fraction of variance explained, relative to predicting the mean.
         Can be negative: that means worse than a constant model.
    MAE  mean absolute error, in the physical units of the target.
         Robust to outliers and directly interpretable.
    RMSE root-mean-square error, same units, penalises large misses more.
    """
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# -------------------------------------------------------------- protocols

def repeated_kfold(model, X, y, n_splits=5, n_repeats=20, seed=0) -> dict:
    """Repeated k-fold, scoring each repeat separately.

    Why score per repeat rather than pooling all out-of-fold predictions?
    Because we want the *spread* across repeats. Pooling would give one
    number and hide exactly the instability we are trying to expose.
    """
    per_repeat = []
    for r in range(n_repeats):
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=1, random_state=seed + r)
        oof = cross_val_predict(clone(model), X, y, cv=cv)
        per_repeat.append(_metrics(y, oof))

    out = {}
    for k in ("R2", "MAE", "RMSE"):
        vals = np.array([m[k] for m in per_repeat])
        out[k] = vals.mean()
        out[k + "_std"] = vals.std()
    return out


def loo(model, X, y) -> dict:
    """Leave-one-out. Deterministic, so no standard deviation."""
    oof = cross_val_predict(clone(model), X, y, cv=LeaveOneOut())
    m = _metrics(y, oof)
    return {**m, "R2_std": np.nan, "MAE_std": np.nan, "RMSE_std": np.nan}


def grouped_sod(model, X, y, groups) -> dict:
    """Leave one stand-off distance out. Tests extrapolation.

    Each fold trains on three SOD levels and predicts the fourth. Note
    that two of the four folds (SOD=200 and SOD=320) require the model to
    extrapolate *beyond* the training range, which is strictly harder than
    the interior folds. Expect poor and uneven numbers -- that is the
    finding, not a bug.
    """
    oof = cross_val_predict(clone(model), X, y, cv=LeaveOneGroupOut(), groups=groups)
    m = _metrics(y, oof)
    return {**m, "R2_std": np.nan, "MAE_std": np.nan, "RMSE_std": np.nan}


# ------------------------------------------------------------ in-sample

def in_sample(model, X, y) -> dict:
    """Fit and score on the same data.

    This is the number the paper appears to report. We compute it ONLY to
    display alongside the honest numbers -- the gap between in-sample and
    out-of-fold is our overfitting diagnostic. Never quote it alone.
    """
    m = clone(model).fit(X, y)
    return _metrics(y, m.predict(X))

# --------------------------------------------------- paired comparison
 
def paired_compare(model_a, Xa, model_b, Xb, y,
                   n_splits=5, n_repeats=20, seed=0) -> dict:
    """Compare two (model, feature-set) combinations on IDENTICAL folds.
 
    Why paired
    ----------
    Most of the variance in a cross-validated score across repeats is fold
    *difficulty*, not model quality: an unlucky split hurts both models
    equally. Because we force both models through the same folds, that
    shared component cancels when we take the per-repeat difference. The
    paired standard deviation is typically far smaller than either model's
    marginal standard deviation, so this test is much more sensitive.
 
    The Nadeau-Bengio correction
    ----------------------------
    A plain paired t-test assumes the per-repeat differences are
    independent. They are not: training sets across folds and repeats
    overlap heavily, so the naive variance is too small and the naive
    t-statistic is too large. Nadeau & Bengio (2003) showed the variance
    should be inflated by a factor
 
        1/k + n_test / n_train
 
    where k is the number of folds. Without this, repeated CV produces
    p-values that are wrong by an order of magnitude -- an extremely
    common error in applied ML papers. We report both so the difference
    is visible.
 
    Returns
    -------
    dict with mean difference (B minus A), its raw and corrected
    standard error, the corrected t statistic, and the win rate.
    """
    from scipy import stats

    n = len(y)
    n_test = n/n_splits
    n_train = n-n_test

    diffs = []
    for r in range(n_repeats):
        cv = RepeatedKFold(n_splits = n_splits, n_repeats = 1, random_state = seed +r )
        oof_a = cross_val_predict(clone(model_a), Xa,y,cv=cv)
        oof_b = cross_val_predict(clone(model_b),Xb,y, cv = cv)
        diffs.append(r2_score(y, oof_b)- r2_score(y, oof_a))
    d = np.array(diffs)
    mean_d = d.mean()
    var_d = d.var(ddof=1)

    se_naive = np.sqrt(var_d / n_repeats)
    correction = (1.0 / n_splits)+(n_test / n_train)
    se_correccted = np.sqrt(var_d *correction)

    t_corr = mean_d / se_corrected if se_corrected > 0 else np.nan
    p_corr = 2* (1- stats.t.cdf(abs(t_corr), df = n_repeats -1))

    return {
        "mean_diff": mean_d,
        "sd_diff": np.sqrt(var_d),
        "se_naive": se_naive,
        "se_corrected": se_corrected,
        "t_corrected": t_corr,
        "p_corrected": p_corr,
        "win_rate": float((d>0).mean()),
    }



# --------------------------------------------------------------- runner

def benchmark(models: dict, X, y, groups=None, target_name="") -> pd.DataFrame:
    """Run every model through every protocol; return a tidy table."""
    rows = []
    for name, model in models.items():
        rk = repeated_kfold(model, X, y)
        lo = loo(model, X, y)
        ins = in_sample(model, X, y)
        row = {
            "target": target_name,
            "model": name,
            "kfold_R2": rk["R2"],
            "kfold_R2_sd": rk["R2_std"],
            "kfold_MAE": rk["MAE"],
            "loo_R2": lo["R2"],
            "loo_MAE": lo["MAE"],
            "insample_R2": ins["R2"],
            "overfit_gap": ins["R2"] - lo["R2"],
        }
        if groups is not None:
            gs = grouped_sod(model, X, y, groups)
            row["groupSOD_R2"] = gs["R2"]
            row["groupSOD_MAE"] = gs["MAE"]
        rows.append(row)
    return pd.DataFrame(rows)