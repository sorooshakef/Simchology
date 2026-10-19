"""Fine-tune tiny local models on the CPU: cross-encoders, LoRA classifiers, pretrained scoring."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from conftest import requires

import simchology as sim
from simchology.baseline.thresholds import ProbabilityThreshold
from simchology.exceptions import ConfigurationError, SimchologyHardwareWarning

pytestmark = requires("torch", "transformers", "accelerate")


@pytest.fixture(scope="module")
def subset(demo_en):
    keep = demo_en["participant"].unique()[:20]
    return demo_en[demo_en["participant"].isin(keep)].copy()


def _tolerance(assessor) -> float:
    """sim.load() picks the best device; off the CPU (MPS/CUDA, possibly bf16) results differ slightly."""
    return 1e-5 if assessor.runtime_.device == "cpu" else 2e-2


def _params(backbone, **kw):
    return {"backbone": backbone, "epochs": 1, "batch_size": 8, "grad_accum": 1, **kw}


def _assess(data, **kw):
    kw.setdefault("device", "cpu")
    kw.setdefault("check_hardware", "off")
    kw.setdefault("verbose", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sim.SimchologyDataWarning)
        return sim.assess(data, return_assessor=True, **kw)


@pytest.fixture(scope="module")
def fitted_binary(subset, tiny_bert_dir):
    return _assess(subset, method="cross_encoder", output="binary", coded=subset, label_col="label",
                   group_col="topic", method_params=_params(tiny_bert_dir))


def test_cross_encoder_binary(fitted_binary, subset, tmp_path):
    out, a = fitted_binary
    col = out["predicted_similarity"]
    assert str(col.dtype) == "Int8"
    assert set(col.dropna().unique()) <= {0, 1}
    assert out.index.equals(subset.index) and len(out) == len(subset)
    assert a.method_.native_output() == "probability"
    assert isinstance(a.threshold_, ProbabilityThreshold)
    assert a.report_ is not None and a.report_.n_test > 0 and "f1" in a.report_.metrics
    assert a.method_.training_summary_["metric"] == "f1"
    assert a.method_.max_length_ <= 256
    scores = a.score(subset).to_numpy()
    assert np.all((scores >= 0) & (scores <= 1))
    assert "cross-encoder" in a.describe()

    path = a.save(tmp_path / "ce")
    meta = json.loads((path / "method" / "simchology_model.json").read_text())
    assert meta["kind"] == "classification" and meta["max_length"] == a.method_.max_length_
    b = sim.load(path)
    assert np.allclose(b.score(subset).to_numpy(), scores, atol=_tolerance(b))
    if b.runtime_.device == "cpu":
        assert b.predict(subset)["predicted_similarity"].equals(col)


def test_cross_encoder_robustness_report(fitted_binary):
    _, a = fitted_binary
    report = a.evaluate(robustness=True)
    assert set(report.robustness["variant"]) == {"original", "no_period", "cut_midword"}
    assert 0.0 <= report.consistency["flip_rate"] <= 1.0


def test_hardware_warning_for_cpu_training(subset, tiny_bert_dir, monkeypatch):
    from simchology.hardware import checks

    # tiny jobs are not worth a warning; pretend this one would take seven hours
    monkeypatch.setattr(checks, "estimate_runtime", lambda *a, **k: 7 * 3600)
    with pytest.warns(SimchologyHardwareWarning, match="slow"):
        _assess(subset, method="cross_encoder", output="binary", coded=subset, label_col="label",
                method_params=_params(tiny_bert_dir), check_hardware="warn", test_size=0.2)


@pytest.mark.slow
def test_cross_encoder_gradient_labels(subset, tiny_bert_dir):
    out, a = _assess(subset, method="cross_encoder", output="gradient", coded=subset,
                     label_col="label_gradient", method_params=_params(tiny_bert_dir, epochs=2))
    assert a.method_.kind_ == "regression" and a.method_.native_output() == "score"
    assert a.method_.training_summary_["metric"] == "spearman"
    pred = out["predicted_similarity"].to_numpy(dtype=float)
    assert np.all((pred >= 0) & (pred <= 1))
    assert "mae" in a.report_.metrics and "rmse" in a.report_.metrics


@pytest.mark.slow
def test_cross_encoder_balanced_weights_without_dev(subset, tiny_bert_dir):
    _, a = _assess(subset, method="cross_encoder", output="binary", coded=subset, label_col="label",
                   dev_size=0.0, method_params=_params(tiny_bert_dir, class_weight="balanced"))
    assert a.method_.training_summary_["metric"] is None  # no development pairs: fixed epochs
    assert a.report_ is not None


def test_lora_classifier_round_trip(subset, tiny_qwen3_dir, tmp_path):
    out, a = _assess(subset, method="lora_classifier", output="binary", coded=subset,
                     label_col="label", method_params=_params(tiny_qwen3_dir, max_length=96))
    assert str(out["predicted_similarity"].dtype) == "Int8"
    assert a.method_.max_length_ == 96
    assert hasattr(a.method_.model_, "peft_config")
    scores = a.score(subset).to_numpy()
    path = a.save(tmp_path / "lora")
    assert (path / "method" / "adapter" / "adapter_config.json").exists()
    b = sim.load(path)
    assert np.allclose(b.score(subset).to_numpy(), scores, atol=_tolerance(b))
    b.save(tmp_path / "lora_again")  # merged model is saved in full
    c = sim.load(tmp_path / "lora_again")
    assert np.allclose(c.score(subset).to_numpy(), scores, atol=_tolerance(c))


@pytest.mark.slow
def test_lora_classifier_regression(subset, tiny_qwen3_dir):
    out, a = _assess(subset, method="lora_classifier", output="gradient", coded=subset,
                     label_col="label_gradient", method_params=_params(tiny_qwen3_dir, max_length=64))
    assert a.method_.kind_ == "regression"
    assert out["predicted_similarity"].between(0, 1).all()


def test_pretrained_cross_encoder(fitted_binary, subset, tiny_bert_dir, tmp_path):
    _, a = fitted_binary
    path = a.save(tmp_path / "trained")
    model_dir = str(path / "method" / "model")
    out = sim.assess(subset, method="pretrained_cross_encoder", output="gradient", device="cpu",
                     method_params={"model": model_dir, "pair_order": "response_first"}, verbose=False)
    expected = a.score(subset).to_numpy()
    assert np.allclose(out["predicted_similarity"].to_numpy(), expected, atol=1e-5)
    with pytest.raises(ConfigurationError, match="no trained classification head"):
        sim.assess(subset, method="pretrained_cross_encoder", device="cpu", verbose=False,
                   method_params={"model": tiny_bert_dir})


def test_compare_methods_skips_or_runs_expensive(subset, tiny_bert_dir):
    skipped = sim.compare_methods(subset, ["cross_encoder"], label_col="label", verbose=False)
    assert skipped.loc[0, "status"].startswith("skipped: expensive")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SimchologyHardwareWarning)
        ran = sim.compare_methods(
            subset, [("cross_encoder", _params(tiny_bert_dir))], label_col="label",
            include_expensive=True, check_hardware="off", verbose=False,
        )
    assert ran.loc[0, "status"] == "ok" and "f1" in ran.columns
