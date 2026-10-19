import numpy as np
import pandas as pd
import pytest

from simchology.baseline import (
    Baseline,
    BaselineQuantileThreshold,
    OptimizedThreshold,
    ReferenceThreshold,
    Threshold,
    build_baseline_pairs,
    make_threshold,
)
from simchology.baseline.scaling import GradientScaler, make_scaler
from simchology.baseline.thresholds import best_cut
from simchology.exceptions import ConfigurationError, SimchologyDataWarning


def _std(df):
    return pd.DataFrame({"left": df["text"], "right": df["info"], "label": df["label"].astype(float),
                         "group": df["topic"]})


def test_auto_baseline_with_labels_and_groups(demo_de):
    pairs = build_baseline_pairs(_std(demo_de), n_pairs=400, seed=1)
    assert pairs.attrs["strategy"] == "coded_negatives+cross_group"
    assert set(pairs["source"]) == {"coded_negative", "cross_group"}
    assert (pairs["source"] == "coded_negative").sum() <= 200
    std = _std(demo_de)
    positives = set(zip(std.left[std.label == 1], std.right[std.label == 1]))
    assert not positives & set(zip(pairs.left, pairs.right))
    cross = pairs[pairs.source == "cross_group"]
    ref_group = std.drop_duplicates("right").set_index("right")["group"]
    assert (cross["right"].map(ref_group) != cross["group_left"]).all()


def test_random_baseline_warns_without_groups(demo_de):
    std = _std(demo_de).drop(columns=["label", "group"])
    with pytest.warns(SimchologyDataWarning, match="random re-pairings"):
        pairs = build_baseline_pairs(std, n_pairs=100)
    assert pairs.attrs["strategy"] == "random" and len(pairs) == 100
    assert not set(zip(std.left, std.right)) & set(zip(pairs.left, pairs.right))


def test_baseline_strategy_requirements(demo_de):
    with pytest.raises(ConfigurationError):
        build_baseline_pairs(_std(demo_de).drop(columns="label"), strategy="coded_negatives")


def test_baseline_object():
    b = Baseline(np.arange(100) / 100, pd.DataFrame({"source": ["x"] * 100}), "m", "random", 0)
    assert b.quantile(0.95) == pytest.approx(0.9405)
    assert b.percentile_of([0.0, 2.0]).tolist() == [0.005, 1.0]
    assert len(b.summary()) == 1
    b2 = Baseline.from_arrays(b.to_arrays(), method="m", strategy="random", seed=0)
    assert np.allclose(b2.scores, b.scores)


def test_quantile_threshold_and_per_group():
    pairs = pd.DataFrame({"group_left": ["a"] * 50 + ["b"] * 50})
    b = Baseline(np.r_[np.linspace(0, 0.5, 50), np.linspace(0, 1, 50)], pairs, "m", "x", 0)
    with pytest.warns(SimchologyDataWarning):
        t = BaselineQuantileThreshold(0.95, per_group=True, min_group_n=10).fit(baseline=b)
    v = t.values(3, groups=["a", "b", "c"])
    assert v[0] < v[1] and v[2] == pytest.approx(t.value_)
    assert t.apply([0.49, 0.49, np.nan], groups=["a", "b", "a"]).tolist()[:2] == [1.0, 0.0]
    assert np.isnan(t.apply([np.nan])[0])
    assert Threshold.from_dict(t.to_dict()).values(1, groups=["a"])[0] == v[0]


def test_best_cut_maximises_f1():
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    labels = np.array([0, 0, 0, 1, 1, 1])
    cut = best_cut(scores, labels)
    assert 0.3 < cut < 0.6
    t = OptimizedThreshold().fit(scores=scores, labels=labels)
    assert t.apply(scores).tolist() == labels.tolist()


def _notebook_thresholds(M, topics, refs):
    """Oracle: the reference-threshold computation of the original notebook (similarity case)."""
    n = len(refs)
    fixed_overall = float(np.nanmax(M[~np.isnan(M)]))
    itemwise, topic_vals = {}, {}
    for a in range(n):
        same = [b for b in range(n) if b != a and topics[b] == topics[a]]
        val = float(np.nanmax(np.array([M[a, b] for b in same])))
        itemwise[refs[a]] = val
        topic_vals.setdefault(topics[a], []).append(val)
    topicwise = {t: float(np.nanmax(np.array(v))) for t, v in topic_vals.items()}
    fixed_topicwise = float(np.nanmax(np.array(list(topicwise.values()))))
    return fixed_overall, fixed_topicwise, topicwise, itemwise


def test_reference_threshold_matches_notebook():
    rng = np.random.default_rng(0)
    refs = [f"ref{i}" for i in range(9)]
    topics = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
    sim = rng.random((9, 9))
    lookup = {(refs[i], refs[j]): sim[i, j] for i in range(9) for j in range(9)}
    scorer = lambda left, right: np.array([lookup[(a, b)] for a, b in zip(left, right)])  # noqa: E731
    frame = pd.DataFrame({"right": refs, "group": topics})
    M = sim.copy()
    np.fill_diagonal(M, np.nan)
    overall, fixed_topicwise, topicwise, itemwise = _notebook_thresholds(M, topics, refs)
    fitted = {v: ReferenceThreshold(v).fit(references=frame, reference_scorer=scorer) for v in "abcd"}
    assert fitted["a"].values(1)[0] == pytest.approx(overall)
    assert fitted["b"].values(1)[0] == pytest.approx(fixed_topicwise)
    assert fitted["c"].values(1, groups=["B"])[0] == pytest.approx(topicwise["B"])
    assert fitted["d"].values(1, groups=["C"], refs=["ref7"])[0] == pytest.approx(itemwise["ref7"])
    # unknown reference falls back to its group, unknown group to the overall value
    assert fitted["d"].values(1, groups=["C"], refs=["new"])[0] == pytest.approx(topicwise["C"])
    assert fitted["c"].values(1, groups=["Z"])[0] == pytest.approx(overall)
    t = Threshold.from_dict(fitted["d"].to_dict())
    assert t.values(1, groups=["C"], refs=["ref7"])[0] == pytest.approx(itemwise["ref7"])


def test_make_threshold_specs():
    assert make_threshold("auto", native_output="probability").kind == "probability"
    assert make_threshold("auto").kind == "baseline_quantile"
    assert make_threshold(0.3).kind == "fixed"
    assert make_threshold("reference_itemwise").variant == "d"
    assert make_threshold("optimize_youden").metric == "youden"
    with pytest.raises(ConfigurationError):
        make_threshold("nonsense")


def test_scalers_round_trip():
    b = Baseline(np.linspace(0, 0.4, 200), pd.DataFrame(), "m", "x", 0)
    for spec in ("baseline_percentile", "baseline_z", "baseline_rescaled"):
        s = make_scaler(spec).fit(baseline=b, upper=1.0)
        out = s.transform([0.0, 0.2, 1.0])
        assert out[0] < out[1] < out[2] or spec == "baseline_percentile"
        assert np.allclose(GradientScaler.from_dict(s.to_dict()).transform([0.2]), s.transform([0.2]))
    platt = make_scaler("calibrated", label_kind="binary").fit(scores=[0.1, 0.2, 0.8, 0.9], labels=[0, 0, 1, 1])
    p = platt.transform([0.1, 0.9])
    assert p[0] < 0.5 < p[1]
    iso = make_scaler("calibrated", label_kind="gradient").fit(scores=[0.1, 0.5, 0.9], labels=[1, 3, 5])
    assert iso.transform([0.5])[0] == pytest.approx(3)
