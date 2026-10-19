"""End-to-end tests of assess()/SimilarityAssessor with the core (dependency-free) methods."""

import warnings

import numpy as np
import pandas as pd
import pytest

import simchology as sim
from simchology.exceptions import ConfigurationError, NotFittedError, SimchologyDataWarning

REGEX = {"backend": "regex"}


@pytest.fixture(autouse=True)
def _no_noise():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SimchologyDataWarning)
        yield


def test_binary_assess_adds_column_and_preserves_index(demo_de):
    data = demo_de.set_index("ID").sample(frac=1, random_state=0)
    out, a = sim.assess(data, method="tfidf", output="binary", coded=data, label_col="label",
                        group_col="topic", preprocess=REGEX, verbose=False, return_assessor=True)
    assert out.index.equals(data.index)
    assert str(out["predicted_similarity"].dtype) == "Int8"
    assert set(out["predicted_similarity"].dropna().unique()) <= {0, 1}
    assert a.report_.metrics["f1"] > 0.7
    assert a.threshold_.kind == "baseline_quantile"
    assert a.baseline_.strategy == "coded_negatives+cross_group"
    assert list(data.columns) == [c for c in out.columns if c != "predicted_similarity"]
    assert out.attrs["simchology"]["method"] == "tfidf"


def test_gradient_output_and_scales(demo_en):
    for scale in ("raw", "baseline_percentile", "baseline_z", "baseline_rescaled", "calibrated"):
        out, a = sim.assess(demo_en, method="jaccard", output="gradient", coded=demo_en,
                            label_col="label", group_col="topic", gradient_scale=scale,
                            preprocess=REGEX, verbose=False, return_assessor=True)
        values = out["predicted_similarity"].to_numpy()
        assert values.dtype == float and np.isfinite(values).all()
        if scale in ("baseline_percentile", "calibrated"):
            assert values.min() >= 0 and values.max() <= 1
        assert a.report_.metrics["spearman_rho"] > 0.4, scale
        assert a.report_.metrics["roc_auc"] > 0.8, scale


@pytest.mark.parametrize("threshold", ["baseline", "baseline_group", "optimize", "optimize_youden",
                                       "reference_a", "reference_b", "reference_c", "reference_d", 0.2])
def test_all_threshold_strategies(demo_en, threshold):
    out, a = sim.assess(demo_en, method="tfidf", output="binary", coded=demo_en, label_col="label",
                        group_col="topic", threshold=threshold, preprocess=REGEX, verbose=False,
                        return_assessor=True, keep_scores=True)
    assert {"similarity_score", "similarity_threshold"} <= set(out.columns)
    pred = out["predicted_similarity"].astype(float).to_numpy()
    assert np.array_equal(pred, (out["similarity_score"] > out["similarity_threshold"]).astype(float))
    assert a.report_.metrics["f1"] > 0.5, threshold


def test_reference_threshold_uses_external_references(demo_en):
    extra = pd.DataFrame({"info": ["Elephants are grey."], "topic": ["Giraffes"]})
    _, a = sim.assess(demo_en, method="jaccard", output="binary", coded=demo_en, label_col="label",
                      group_col="topic", threshold="reference_d", references=extra,
                      preprocess=REGEX, verbose=False, return_assessor=True)
    assert "Elephants are grey." in set(a.references_["right"])


def test_without_coded_data_uses_cross_group_baseline(demo_en):
    data = demo_en.drop(columns=["label", "label_gradient"])
    out, a = sim.assess(data, method="tfidf", output="binary", group_col="topic", preprocess=REGEX,
                        verbose=False, return_assessor=True)
    assert a.baseline_.strategy == "cross_group" and a.report_ is None
    assert out["predicted_similarity"].notna().all()


def test_label_col_in_data_is_used_as_coded(demo_en):
    data = demo_en.copy()
    data.loc[data.index[300:], "label"] = np.nan
    out, a = sim.assess(data, method="jaccard", output="binary", label_col="label",
                        group_col="topic", preprocess=REGEX, verbose=False, return_assessor=True)
    assert a.report_ is not None and len(out) == len(data)


def test_gradient_labels_with_binary_output(demo_en):
    _, a = sim.assess(demo_en, method="tfidf", output="binary", coded=demo_en,
                      label_col="label_gradient", positive_threshold=1.0, group_col="topic",
                      threshold="optimize", preprocess=REGEX, verbose=False, return_assessor=True)
    assert a.label_info_.kind == "gradient" and a.report_.metrics["f1"] > 0.6
    with pytest.raises(ConfigurationError, match="positive_threshold"):
        sim.assess(demo_en, method="tfidf", output="binary", coded=demo_en,
                   label_col="label_gradient", threshold="optimize", preprocess=REGEX, verbose=False)


def test_missing_texts_get_missing_predictions(demo_en):
    data = demo_en.copy()
    data.loc[data.index[:3], "text"] = None
    with pytest.warns(SimchologyDataWarning, match="missing"):
        out = sim.assess(data, method="jaccard", output="binary", coded=demo_en, label_col="label",
                         group_col="topic", preprocess=REGEX, verbose=False)
    assert out["predicted_similarity"].isna().sum() == 3


def test_save_load_round_trip(tmp_path, demo_de):
    for method in ("tfidf", "lsa", "lda", "bm25", "jaccard"):
        out, a = sim.assess(demo_de, method=method, output="binary", coded=demo_de, label_col="label",
                            group_col="topic", preprocess=REGEX, verbose=False, return_assessor=True,
                            keep_scores=True)
        path = a.save(tmp_path / method)
        b = sim.load(path)
        again = b.predict(demo_de, keep_scores=True)
        assert np.allclose(again["similarity_score"], out["similarity_score"]), method
        assert again["predicted_similarity"].equals(out["predicted_similarity"]), method
        assert b.report_.metrics == pytest.approx(a.report_.metrics)


def test_explain_and_describe(demo_en):
    _, a = sim.assess(demo_en, method="tfidf", output="binary", coded=demo_en, label_col="label",
                      group_col="topic", preprocess=REGEX, verbose=False, return_assessor=True)
    row = demo_en.iloc[0]
    exp = a.explain(row["text"], row["info"])
    assert isinstance(exp, sim.Explanation) and exp.score >= 0
    text = a.describe()
    assert "TF-IDF" in text and "held out" in text and "baseline" in text
    one = sim.explain("Giraffes sleep two hours.", "Giraffes sleep about two hours a day.",
                      method="jaccard", language="en")
    assert "giraffes" in one.shared


def test_evaluate_with_ci_and_robustness(demo_en):
    _, a = sim.assess(demo_en, method="jaccard", output="binary", coded=demo_en, label_col="label",
                      group_col="topic", preprocess=REGEX, verbose=False, return_assessor=True)
    report = a.evaluate(ci=0.95, n_boot=100, robustness=True)
    lo, hi = report.ci["f1"]
    assert lo <= report.metrics["f1"] <= hi
    assert set(report.robustness["variant"]) == {"original", "no_period", "cut_midword"}
    assert 0 <= report.consistency["flip_rate"] <= 1
    assert "robustness" in str(report)


def test_baseline_function(demo_en):
    b = sim.baseline(demo_en, method="jaccard", label_col="label", group_col="topic",
                     n_pairs=300, preprocess=REGEX, verbose=False)
    assert isinstance(b, sim.Baseline) and len(b) > 100
    assert b.summary()["source"].tolist()[0] == "all"


def test_compare_methods_fast(demo_en):
    table = sim.compare_methods(demo_en, "fast", label_col="label", group_col="topic", verbose=False)
    assert set(table["status"]) == {"ok"}
    assert table["f1"].iloc[0] >= table["f1"].iloc[-1]
    custom = sim.compare_methods(demo_en, [("jaccard", {"variant": "containment"}), "nonexistent"],
                                 label_col="label", group_col="topic", verbose=False)
    assert custom["status"].tolist()[0] == "ok"
    assert custom["status"].tolist()[1].startswith("failed")


def test_column_and_config_errors(demo_en):
    with pytest.raises(ConfigurationError, match="Did you mean"):
        sim.assess(demo_en, columns=("txt", "info"), method="jaccard", verbose=False)
    with pytest.raises(ConfigurationError):
        sim.assess(demo_en, method="jaccard", output="ordinal", verbose=False)
    with pytest.raises(NotFittedError):
        sim.SimilarityAssessor("jaccard").predict(demo_en)


def test_split_by_row_reproduces_row_split(demo_en):
    _, a = sim.assess(demo_en, method="jaccard", output="binary", coded=demo_en, label_col="label",
                      split_by="row", preprocess=REGEX, verbose=False, return_assessor=True)
    assert a.splits_.split_by is None and len(a.splits_.test) == 120
