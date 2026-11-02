"""Sentence-Transformers, pooled HF and LLM embedding methods (tiny local models)."""

from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pytest
from conftest import requires

import simchology as sim
from simchology.config import DEFAULT_MODELS
from simchology.exceptions import ConfigurationError, ModelNotAvailableError, SimchologyDataWarning
from simchology.methods.base import Runtime
from simchology.methods.registry import make_method

pytestmark = requires("torch", "transformers", "sentence_transformers")

TEXTS = [
    "Giraffes are the tallest land animals in the world.",
    "Giraffes sleep only about two hours a day.",
    "The Romans built aqueducts to carry water into the cities.",
    "The moai are huge stone statues on Easter Island.",
]


def _bind(m):
    m.bind(None, Runtime(device="cpu", language="en", verbose=False))
    return m


@pytest.fixture
def sbert(tiny_st_dir):
    return _bind(make_method("sbert", {"model": tiny_st_dir}))


def _check_properties(m):
    left = [TEXTS[i] for i in (0, 1, 2, 3, 0)]
    right = [TEXTS[i] for i in (1, 2, 3, 0, 1)]
    s = m.score_pairs(left, right)
    assert s.shape == (5,) and np.all(np.isfinite(s))
    assert np.all(s >= -1 - 1e-9) and np.all(s <= 1 + 1e-9)
    assert s[0] == pytest.approx(s[4])
    np.testing.assert_allclose(s, m.score_pairs(right, left), atol=1e-6)
    same = m.score_pairs(TEXTS, TEXTS)
    np.testing.assert_allclose(same, 1.0, atol=1e-5)
    assert np.all(same >= s.max() - 1e-6)


def test_sbert_properties(sbert):
    assert sbert.info.preprocessing == "light" and sbert.info.cost_tier == "moderate"
    _check_properties(sbert)


@pytest.mark.parametrize("pooling", ["mean", "cls", "last_token"])
def test_hf_embedding_properties(tiny_bert_dir, pooling):
    _check_properties(_bind(make_method("hf_embedding", {"model": tiny_bert_dir, "pooling": pooling})))


def test_hf_embedding_layer_and_bad_pooling(tiny_bert_dir):
    m = _bind(make_method("hf_embedding", {"model": tiny_bert_dir, "layer": 1}))
    _check_properties(m)
    with pytest.raises(ConfigurationError):
        make_method("hf_embedding", {"pooling": "max"})


def test_embedding_cache_hit(sbert, monkeypatch):
    calls = []
    original = type(sbert)._embed_uncached

    def spy(self, texts):
        calls.append(list(texts))
        return original(self, texts)

    monkeypatch.setattr(type(sbert), "_embed_uncached", spy)
    sbert.score_pairs(TEXTS[:2], TEXTS[1:3])
    sbert.score_pairs(TEXTS[:3], TEXTS[:3])
    assert calls == [[TEXTS[0], TEXTS[1], TEXTS[2]]]  # second call fully cached


def test_disk_cache(tiny_st_dir, tmp_path, monkeypatch):
    first = _bind(make_method("sbert", {"model": tiny_st_dir, "cache_dir": str(tmp_path)}))
    expected = first.score_pairs(TEXTS[:2], TEXTS[2:])
    assert list(tmp_path.glob("*.npz"))
    second = _bind(make_method("sbert", {"model": tiny_st_dir, "cache_dir": str(tmp_path)}))
    monkeypatch.setattr(type(second), "_model", lambda self: pytest.fail("model should not load"))
    np.testing.assert_allclose(second.score_pairs(TEXTS[:2], TEXTS[2:]), expected, atol=1e-6)


def test_truncation_warning(tiny_st_dir):
    m = _bind(make_method("sbert", {"model": tiny_st_dir, "max_seq_length": 4}))
    with pytest.warns(SimchologyDataWarning, match="truncated"):
        m.score_pairs(TEXTS, TEXTS[::-1])


def test_no_truncation_warning_for_short_texts(sbert):
    with warnings.catch_warnings():
        warnings.simplefilter("error", SimchologyDataWarning)
        sbert.score_pairs(TEXTS, TEXTS[::-1])


def test_use_like_shortcut_and_metadata(sbert, tiny_st_dir):
    assert make_method("sbert", {"model": "use_like"}).model_id == DEFAULT_MODELS["use_like"]
    req = sbert.resource_request(100)
    assert req.task == "inference" and req.model_id == tiny_st_dir and req.n_items == 100
    n_params = sum(p.numel() for p in sbert._model().parameters())
    assert sbert.model_details() == {"embedding_model": {"name": tiny_st_dir, "source": "local",
                                                         "version": None, "parameters": n_params}}
    assert tiny_st_dir in sbert.describe()


def test_llm_embedding_uses_prompt_on_both_sides(tiny_st_dir):
    m = make_method("llm_embedding")
    assert m.model_id == DEFAULT_MODELS["llm_embedding"] and m.params["prompt"].startswith("Instruct:")
    llm = _bind(make_method("llm_embedding", {"model": tiny_st_dir}))
    plain = _bind(make_method("sbert", {"model": tiny_st_dir}))
    a, b = llm.encode(TEXTS[:2]), plain.encode(TEXTS[:2])
    assert not np.allclose(a, b)
    _check_properties(llm)


def test_missing_model_is_friendly():
    m = _bind(make_method("sbert", {"model": "simchology-test/does-not-exist"}))
    with pytest.raises(ModelNotAvailableError, match="does-not-exist"):
        m.score_pairs(["a"], ["b"])


def test_assess_end_to_end_binary(demo_en, tiny_st_dir):
    out, assessor = sim.assess(
        demo_en, columns=("text", "info"), method="sbert", method_params={"model": tiny_st_dir},
        device="cpu", output="binary", coded=demo_en, label_col="label", group_col="topic",
        language="en", verbose=False, return_assessor=True,
    )
    assert out["predicted_similarity"].dtype == "Int8"
    assert set(out["predicted_similarity"].dropna().unique()) <= {0, 1}
    assert assessor.baseline_ is not None and len(assessor.baseline_) > 100
    assert assessor.report_.metrics["n"] > 0
    assert "sentence embeddings" in assessor.describe()


def test_assess_gradient_baseline_percentile(demo_en, tiny_st_dir):
    out = sim.assess(
        demo_en, method="sbert", method_params={"model": tiny_st_dir}, device="cpu",
        coded=demo_en, label_col="label", group_col="topic", gradient_scale="baseline_percentile",
        language="en", verbose=False,
    )
    assert out["predicted_similarity"].between(0, 1).all()


def test_neural_modules_do_not_import_torch():
    code = (
        "import sys; import simchology.methods.contextual, simchology.methods.bertscore, "
        "simchology.methods.wmd, simchology.methods.static_embeddings, simchology.methods._torch_utils; "
        "import simchology; simchology.list_methods(); "
        "bad = [m for m in ('torch', 'transformers', 'sentence_transformers', 'ot', 'gensim') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
