"""Agreement metrics between automatic and human similarity judgments."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _clean(*arrays):
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= ~np.isnan(a)
    return [a[mask] for a in arrays]


def binary_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    """Precision, recall, F1, accuracy, balanced accuracy, Cohen's kappa, MCC (+ AUC/AP/r_pb)."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if y_score is not None:
        y_true, y_pred, y_score = _clean(y_true, y_pred, y_score)
    else:
        y_true, y_pred = _clean(y_true, y_pred)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan"),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred) if len(np.unique(np.r_[y_true, y_pred])) > 1 else float("nan"),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan"),
        "n": int(len(y_true)),
        "positive_rate_true": float(y_true.mean()) if len(y_true) else float("nan"),
        "positive_rate_pred": float(y_pred.mean()) if len(y_pred) else float("nan"),
    }
    if y_score is not None and len(np.unique(y_true)) > 1 and np.std(y_score) > 0:
        from scipy.stats import pointbiserialr

        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["average_precision"] = average_precision_score(y_true, y_score)
        out["point_biserial_r"] = float(pointbiserialr(y_true, y_score)[0])
    return {k: float(v) if k != "n" else v for k, v in out.items()}


def gradient_metrics(y_true, y_score, *, on_label_scale: bool = False) -> dict[str, float]:
    """Correlations (Pearson, Spearman, Kendall); MAE/RMSE only if scores are on the label scale."""
    from scipy.stats import kendalltau, pearsonr, spearmanr

    y_true, y_score = _clean(y_true, y_score)
    out: dict[str, float] = {"n": int(len(y_true))}
    if len(y_true) > 2 and np.std(y_true) > 0 and np.std(y_score) > 0:
        out["pearson_r"] = float(pearsonr(y_true, y_score)[0])
        out["spearman_rho"] = float(spearmanr(y_true, y_score)[0])
        out["kendall_tau"] = float(kendalltau(y_true, y_score)[0])
    else:
        out.update(pearson_r=float("nan"), spearman_rho=float("nan"), kendall_tau=float("nan"))
    if on_label_scale:
        err = y_score - y_true
        out["mae"] = float(np.mean(np.abs(err)))
        out["rmse"] = float(np.sqrt(np.mean(err**2)))
    return out


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true,
    y_hat,
    *,
    groups=None,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI; resamples whole ``groups`` (e.g. responses) if given."""
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(y_true))
    groups = np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        try:
            value = metric(y_true[idx], y_hat[idx])
        except ValueError:
            continue
        if not np.isnan(value):
            stats.append(value)
    if not stats:
        return float("nan"), float("nan")
    alpha = (1 - ci) / 2
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))
