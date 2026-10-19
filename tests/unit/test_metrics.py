import numpy as np
from sklearn.metrics import f1_score

from simchology.evaluation.metrics import binary_metrics, bootstrap_ci, gradient_metrics


def test_binary_metrics_match_sklearn():
    y = np.array([1, 0, 1, 1, 0, 0, 1])
    p = np.array([1, 0, 0, 1, 1, 0, 1])
    s = np.array([0.9, 0.1, 0.4, 0.8, 0.6, 0.2, 0.7])
    m = binary_metrics(y, p, y_score=s)
    assert np.isclose(m["f1"], f1_score(y, p))
    assert 0 <= m["roc_auc"] <= 1 and "cohen_kappa" in m and m["n"] == 7


def test_binary_metrics_ignore_missing():
    m = binary_metrics([1, 0, np.nan], [1, 0, 1])
    assert m["n"] == 2 and m["accuracy"] == 1.0


def test_gradient_metrics():
    y = np.array([0, 0.5, 1, 1, 0])
    s = np.array([0.1, 0.4, 0.9, 0.8, 0.2])
    m = gradient_metrics(y, s, on_label_scale=True)
    assert m["spearman_rho"] > 0.9 and "mae" in m
    assert "mae" not in gradient_metrics(y, s)


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.where(rng.random(200) < 0.8, y, 1 - y)
    lo, hi = bootstrap_ci(lambda a, b: f1_score(a, b), y, p, groups=np.arange(200) // 4, n_boot=200)
    assert lo < f1_score(y, p) < hi
