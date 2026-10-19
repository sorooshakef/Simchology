"""Fast tests of supervised pair scoring that do not train models."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import requires

import simchology as sim
from simchology.data.labels import LabelInfo
from simchology.exceptions import ConfigurationError, ModelNotAvailableError
from simchology.finetune.datasets import (
    PairDataset,
    batch_to_features,
    fit_template,
    pair_lengths,
    resolve_max_length,
)
from simchology.finetune.trainer import logits_to_scores
from simchology.methods.registry import get_method_class
from simchology.methods.supervised import (
    CrossEncoderMethod,
    LoRAClassifierMethod,
    PretrainedCrossEncoderMethod,
)


def test_registry_entries_and_aliases():
    assert get_method_class("finetune").info.name == "cross_encoder"
    assert get_method_class("crossencoder").info.name == "cross_encoder"
    assert get_method_class("lora").info.name == "lora_classifier"
    assert get_method_class("sts_cross_encoder").info.name == "pretrained_cross_encoder"
    table = sim.list_methods().set_index("method")
    for name in ("cross_encoder", "lora_classifier", "pretrained_cross_encoder"):
        assert table.loc[name, "family"] == "supervised"
    assert table.loc["cross_encoder", "needs_coded_data"]
    assert table.loc["cross_encoder", "cost"] == "heavy"
    assert not table.loc["pretrained_cross_encoder", "needs_coded_data"]


def test_importing_supervised_modules_does_not_import_torch():
    code = ("import sys, simchology.methods.supervised, simchology.finetune.trainer, "
            "simchology.finetune.lora; assert 'torch' not in sys.modules, 'torch imported'")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_parameter_validation():
    with pytest.raises(ConfigurationError, match="Unknown parameter"):
        CrossEncoderMethod(epoch=3)
    with pytest.raises(ConfigurationError):
        CrossEncoderMethod(class_weight="inverse")
    with pytest.raises(ConfigurationError):
        CrossEncoderMethod(bf16="yes")
    with pytest.raises(ConfigurationError, match="template"):
        LoRAClassifierMethod(template="Text: {response}")
    with pytest.raises(ConfigurationError):
        PretrainedCrossEncoderMethod(activation="tanh")
    with pytest.raises(ConfigurationError):
        PretrainedCrossEncoderMethod(pair_order="sideways")


def test_lora_defaults_differ_from_cross_encoder():
    lora = LoRAClassifierMethod()
    assert lora.params["learning_rate"] == 1e-4 and lora.params["epochs"] == 3
    assert lora.params["gradient_checkpointing"] is True
    assert lora.training_task == "lora"
    assert CrossEncoderMethod().training_task == "full_finetune"


def test_resolve_max_length():
    lengths = np.arange(1, 101)
    assert resolve_max_length(lengths, model_max=None, cap=512) == 96  # ceil(95th percentile)
    assert resolve_max_length(lengths, model_max=64, cap=512) == 64
    assert resolve_max_length(np.full(10, 3), model_max=None, cap=512) == 16  # lower bound
    assert resolve_max_length(lengths, model_max=256, cap=512, requested=300) == 256
    assert resolve_max_length(lengths, model_max=None, cap=512, requested=40) == 40


def test_native_output_and_range_follow_label_type():
    m = CrossEncoderMethod()
    m.runtime.label_info = LabelInfo("binary", 0.0, 1.0, 10, 0.5)
    assert m.native_output() == "probability" and m.score_range() == (0.0, 1.0)
    m.runtime.label_info = LabelInfo("gradient", 1.0, 5.0, 10)
    assert m.native_output() == "score"
    m.label_min_, m.label_max_ = 1.0, 5.0
    assert m.score_range() == (1.0, 5.0)
    assert PretrainedCrossEncoderMethod().native_output() == "score"


def test_label_scaling_round_trip():
    m = CrossEncoderMethod()
    m.label_min_, m.label_max_ = 1.0, 5.0
    y = np.array([1.0, 3.0, 5.0])
    scaled = m._targets(y, regression=True)
    assert np.allclose(scaled, [0, 0.5, 1])
    assert m._targets(np.array([0.0, 1.0]), regression=False).dtype.kind == "i"


def test_logits_to_scores():
    logits = np.array([[0.0, 0.0], [0.0, 10.0]])
    assert np.allclose(logits_to_scores(logits, "softmax"), [0.5, 1.0], atol=1e-4)
    assert np.allclose(logits_to_scores(np.array([[0.0], [100.0]]), "sigmoid"), [0.5, 1.0])
    assert np.allclose(logits_to_scores(np.array([[0.3], [-2.0]]), "identity"), [0.3, -2.0])


def test_pair_dataset_and_features():
    enc = {"input_ids": [[1, 2], [3]], "attention_mask": [[1, 1], [1]], "offset_mapping": [[0], [0]]}
    feats = batch_to_features(enc)
    assert feats == [{"input_ids": [1, 2], "attention_mask": [1, 1]}, {"input_ids": [3], "attention_mask": [1]}]
    ds = PairDataset(feats, [0, 1])
    assert len(ds) == 2 and ds[1]["labels"] == 1 and "labels" not in feats[1]
    with pytest.raises(ValueError):
        PairDataset(feats, [0])


def test_fit_requires_coded_pairs(small_pairs):
    with pytest.raises(ConfigurationError, match="coded"):
        CrossEncoderMethod().fit()
    with pytest.raises(ConfigurationError, match="coded"):
        sim.assess(small_pairs, method="cross_encoder", verbose=False)


@requires("torch", "transformers", "accelerate")
def test_single_class_training_data_is_rejected():
    train = pd.DataFrame({"left": ["a", "b"], "right": ["c", "d"], "label": [1.0, 1.0]})
    m = CrossEncoderMethod()
    m.runtime.label_info = LabelInfo("binary", 0.0, 1.0, 2, 1.0)
    with pytest.raises(ConfigurationError, match="one class"):
        m.fit(train=train)


@requires("torch", "transformers")
def test_architecture_check(tiny_bert_dir, tiny_qwen3_dir, tmp_path):
    from transformers import ViTConfig

    from simchology.finetune.trainer import check_sequence_classification

    assert check_sequence_classification(tiny_bert_dir).model_type == "bert"
    assert check_sequence_classification(tiny_qwen3_dir).model_type == "qwen3"
    ViTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
              intermediate_size=37).save_pretrained(tmp_path / "vit")
    with pytest.raises(ConfigurationError, match="cannot be used as a sequence classifier"):
        check_sequence_classification(str(tmp_path / "vit"))
    with pytest.raises((ModelNotAvailableError, ConfigurationError)):
        check_sequence_classification(str(tmp_path / "does-not-exist"))


@requires("torch", "transformers")
def test_fit_template_keeps_end_of_prompt(tiny_bert_dir):
    from simchology.finetune.trainer import load_tokenizer

    tok = load_tokenizer(tiny_bert_dir)
    template = "Text: {response}\nInformation: {reference}\nAnswer:"
    response = " ".join(["giraffes sleep two hours"] * 60)
    reference = "giraffes sleep about two hours a day ."
    text = fit_template(tok, template, response, reference, max_length=40)
    assert text.endswith("Answer:")
    assert reference in text
    assert len(tok(text)["input_ids"]) <= 40
    short = fit_template(tok, template, "giraffes", reference, max_length=40)
    assert short == template.format(response="giraffes", reference=reference)
    lengths = pair_lengths(tok, ["giraffes sleep", "a"], ["the moai", "b"])
    assert lengths.tolist()[0] > lengths.tolist()[1]


@requires("torch", "transformers", "huggingface_hub")
def test_resource_requests(tiny_bert_dir, tiny_qwen3_dir):
    from simchology.hardware import check_resources

    ce = CrossEncoderMethod(backbone=tiny_bert_dir, epochs=3, batch_size=4)
    req = ce.resource_request(100, ce.training_task)
    assert req.task == "full_finetune" and req.model_id == tiny_bert_dir
    assert (req.n_items, req.epochs, req.batch_size, req.seq_len) == (100, 3, 4, 512)
    assert "fine-tuning" in req.description
    inf = ce.resource_request(250, "inference")
    assert inf.task == "inference" and inf.batch_size == 16 and inf.n_items == 250
    verdict = check_resources(req, device="cpu", policy="off")
    assert verdict.level == "ok"
    verdict = check_resources(req, device="cpu")
    assert verdict.level in ("ok", "slow", "tight", "insufficient")
    assert verdict.estimate is not None and verdict.estimate.total_bytes > 0

    lora = LoRAClassifierMethod(backbone=tiny_qwen3_dir)
    lreq = lora.resource_request(80, lora.training_task)
    assert lreq.task == "lora" and lreq.seq_len == 1024 and lreq.gradient_checkpointing
