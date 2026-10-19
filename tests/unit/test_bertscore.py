"""In-house BERTScore (tiny local BERT)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import requires

import simchology as sim
from simchology.exceptions import ConfigurationError
from simchology.methods.base import Runtime
from simchology.methods.bertscore import RECOMMENDED_LAYERS, recommended_layer
from simchology.methods.registry import make_method

pytestmark = requires("torch", "transformers")

LONG = "Giraffes are the tallest land animals and they sleep only about two hours a day."
SHORT = "Giraffes sleep two hours."
TEXTS = [
    "Giraffes are the tallest land animals in the world.",
    "Giraffes sleep only about two hours a day.",
    "The Romans built aqueducts to carry water into the cities.",
    "The moai are huge stone statues on Easter Island.",
]


def _make(model, **params):
    m = make_method("bertscore", {"model": model, **params})
    m.bind(None, Runtime(device="cpu", language="en", verbose=False))
    return m


def test_identical_texts_score_one(tiny_bert_dir):
    m = _make(tiny_bert_dir)
    a, b = m.encode([LONG, LONG])
    p, r, f, _ = m._prf(a, b)
    assert p == pytest.approx(1.0, abs=2e-3) and r == pytest.approx(1.0, abs=2e-3)
    assert f == pytest.approx(1.0, abs=2e-3)


def test_properties(tiny_bert_dir):
    m = _make(tiny_bert_dir)
    left = [TEXTS[i] for i in (0, 1, 2, 3, 0)]
    right = [TEXTS[i] for i in (1, 2, 3, 0, 1)]
    s = m.score_pairs(left, right)
    assert np.all(np.isfinite(s)) and np.all(np.abs(s) <= 1)
    assert s[0] == pytest.approx(s[4])
    np.testing.assert_allclose(s, m.score_pairs(right, left), atol=1e-6)  # F1 is symmetric
    same = m.score_pairs(TEXTS, TEXTS)
    np.testing.assert_allclose(same, 1.0, atol=2e-3)  # token vectors are cached in float16
    assert np.all(same >= s.max() - 2e-3)


def test_precision_recall_asymmetry(tiny_bert_dir):
    p = _make(tiny_bert_dir, score="precision")
    r = _make(tiny_bert_dir, score="recall")
    # precision(a, b) == recall(b, a)
    assert p.score_pairs([LONG], [SHORT])[0] == pytest.approx(r.score_pairs([SHORT], [LONG])[0], abs=1e-6)
    assert p.score_pairs([LONG], [SHORT])[0] != pytest.approx(r.score_pairs([LONG], [SHORT])[0], abs=1e-4)


def test_layer_resolution(tiny_bert_dir):
    assert recommended_layer("bert-base-multilingual-cased", 12) == (9, True)
    assert recommended_layer("FacebookAI/xlm-roberta-large", 24) == (17, True)
    assert recommended_layer("my/unknown-model", 10) == (6, False)
    assert RECOMMENDED_LAYERS["distilbert-base-multilingual-cased"] == 5
    m = _make(tiny_bert_dir)
    m.encode(["x"])
    assert m.layer_ == 1  # 60% of 2 layers
    explicit = _make(tiny_bert_dir, layer=2)
    explicit.encode(["x"])
    assert explicit.layer_ == 2
    assert explicit.score_pairs([TEXTS[0]], [TEXTS[1]])[0] != pytest.approx(m.score_pairs([TEXTS[0]], [TEXTS[1]])[0])


def test_explanation_is_word_level(tiny_bert_dir):
    m = _make(tiny_bert_dir)
    exp = m.explain_pair(LONG, SHORT)
    assert set(exp.components) == {"precision", "recall", "f1"}
    assert exp.score == pytest.approx(exp.components["f1"])
    a = exp.alignments
    assert set(a["side"]) == {"response", "reference"}
    assert not a["token"].str.startswith("##").any()  # sub-words merged into words
    assert "Giraffes" in set(a.loc[a.side == "reference", "token"])
    assert exp.matrix.shape == (len(exp.response_tokens), len(exp.reference_tokens))


def test_idf_weighting_and_persistence(tiny_bert_dir, tmp_path):
    corpus = TEXTS * 2 + ["giraffes giraffes giraffes"] * 5
    m = _make(tiny_bert_dir, idf=True).fit(corpus=corpus)
    plain = _make(tiny_bert_dir)
    assert m.score_pairs([LONG], [SHORT])[0] != pytest.approx(plain.score_pairs([LONG], [SHORT])[0], abs=1e-6)
    m.save(tmp_path)
    loaded = _make(tiny_bert_dir, idf=True)
    loaded.load(tmp_path)
    assert loaded.idf_ == pytest.approx(m.idf_)
    assert loaded.score_pairs([LONG], [SHORT])[0] == pytest.approx(m.score_pairs([LONG], [SHORT])[0])


def test_empty_text(tiny_bert_dir):
    assert _make(tiny_bert_dir).score_pairs([""], [SHORT])[0] == 0.0


def test_invalid_score():
    with pytest.raises(ConfigurationError):
        make_method("bertscore", {"score": "accuracy"})


def test_assess_and_explain_end_to_end(demo_en, tiny_bert_dir, tmp_path):
    out, assessor = sim.assess(
        demo_en, method="bertscore", method_params={"model": tiny_bert_dir}, device="cpu",
        output="binary", coded=demo_en, label_col="label", group_col="topic", language="en",
        verbose=False, return_assessor=True,
    )
    assert out["predicted_similarity"].notna().all()
    row = demo_en.iloc[0]
    exp = assessor.explain(row["text"], row["info"])
    assert exp.alignments is not None and len(exp.alignments)
    assert exp.score == pytest.approx(assessor.score(demo_en.iloc[:1]).iloc[0])
    assessor.save(tmp_path / "a")
    again = sim.load(tmp_path / "a")
    np.testing.assert_array_equal(again.predict(demo_en.head(20))["predicted_similarity"].to_numpy(),
                                  out.head(20)["predicted_similarity"].to_numpy())
