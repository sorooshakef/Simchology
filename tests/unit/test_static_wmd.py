"""Static word-vector and Word Mover's Distance methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import requires

import simchology as sim
from simchology.exceptions import ConfigurationError, ModelNotAvailableError
from simchology.methods.base import Runtime
from simchology.methods.registry import make_method
from simchology.methods.static_embeddings import load_vectors
from simchology.text.preprocess import PreprocessConfig, Preprocessor

TEXTS = [
    "Giraffes are the tallest land animals in the world.",
    "Giraffes sleep only about two hours a day.",
    "The Romans built aqueducts to carry water into the cities.",
    "The moai are huge stone statues on Easter Island.",
    "Most moai were carved from volcanic rock.",
]


def _bind(method, language="en"):
    method.bind(Preprocessor(PreprocessConfig(language=language), verbose=False),
                Runtime(device="cpu", language=language, verbose=False))
    return method


@pytest.fixture
def method_factory(spacy_vectors_nlp):
    def make(name, **params):
        params.setdefault("vectors", spacy_vectors_nlp)
        return _bind(make_method(name, params))

    return make


CONFIGS = [
    ("word_vectors", {}),
    ("word_vectors", {"weighting": "idf"}),
    pytest.param("wmd", {}, marks=requires("ot")),
    pytest.param("wmd", {"cost": "cosine"}, marks=requires("ot")),
]


@pytest.mark.parametrize("name,params", CONFIGS)
def test_properties(method_factory, name, params):
    m = method_factory(name, **params)
    if m.info.requires_corpus_fit:
        m.fit(corpus=TEXTS)
    left = [TEXTS[i] for i in (0, 1, 2, 3, 4, 0)]
    right = [TEXTS[i] for i in (1, 2, 3, 4, 0, 1)]
    s = m.score_pairs(left, right)
    lo, hi = m.score_range()
    assert s.shape == (6,) and np.all(np.isfinite(s))
    assert np.all(s >= lo - 1e-9) and np.all(s <= hi + 1e-9)
    assert s[0] == pytest.approx(s[5])  # duplicates get identical scores
    assert m.info.symmetric
    np.testing.assert_allclose(s, m.score_pairs(right, left), atol=1e-9)
    same = m.score_pairs(TEXTS, TEXTS)
    np.testing.assert_allclose(same, 1.0, atol=1e-6)  # identity is the maximum
    assert np.all(same >= s.max() - 1e-9)


def test_word_vectors_matches_manual_mean(method_factory, spacy_vectors_nlp):
    m = method_factory("word_vectors")
    a, b = "giraffes sleep", "romans built aqueducts"
    va = np.mean([spacy_vectors_nlp.vocab.get_vector(w) for w in ("giraffes", "sleep")], axis=0)
    vb = np.mean([spacy_vectors_nlp.vocab.get_vector(w) for w in ("romans", "built", "aqueducts")], axis=0)
    expected = va @ vb / np.linalg.norm(va) / np.linalg.norm(vb)
    assert m.score_pairs([a], [b])[0] == pytest.approx(expected, abs=1e-5)


def test_idf_weighting_changes_scores(method_factory):
    plain = method_factory("word_vectors")
    idf = method_factory("word_vectors", weighting="idf").fit(corpus=TEXTS * 3 + ["giraffes"] * 10)
    assert idf.idf_["giraffes"] < idf.idf_["romans"]
    s_plain = plain.score_pairs([TEXTS[0]], [TEXTS[1]])[0]
    s_idf = idf.score_pairs([TEXTS[0]], [TEXTS[1]])[0]
    assert s_plain != pytest.approx(s_idf)


def test_empty_or_unknown_text_scores_zero(method_factory):
    m = method_factory("word_vectors")
    assert m.score_pairs(["", "qwxz zzyq"], [TEXTS[0], TEXTS[0]]).tolist() == [0.0, 0.0]


@requires("ot")
def test_wmd_unknown_words_score_zero(method_factory):
    m = method_factory("wmd")
    assert m.score_pairs(["qwxz"], [TEXTS[0]])[0] == 0.0


def test_dict_vectors_and_capitalised_lookup():
    vectors = {"Giraffen": [1.0, 0.0], "schlafen": [0.0, 1.0], "Hals": [1.0, 0.1]}
    m = _bind(make_method("word_vectors", {"vectors": vectors}), language="de")
    s = m.score_pairs(["Giraffen schlafen"], ["Der Hals"])[0]
    assert 0 < s < 1
    toks, _ = m.token_vectors("Giraffen schlafen")
    assert toks == ["giraffen", "schlafen"]  # lower-cased tokens found via 'Giraffen'


@requires("gensim")
def test_keyed_vectors_and_files(tmp_path):
    from gensim.models import KeyedVectors

    kv = KeyedVectors(vector_size=3)
    kv.add_vectors(["giraffes", "sleep", "statues"], np.eye(3, dtype=np.float32))
    pre = Preprocessor(PreprocessConfig(language="en"), verbose=False)
    assert load_vectors(kv, pre).get("sleep").tolist() == [0.0, 1.0, 0.0]

    w2v = tmp_path / "vectors.txt"
    w2v.write_text("2 3\ngiraffes 1 0 0\nsleep 0 1 0\n", encoding="utf-8")
    glove = tmp_path / "glove.txt"
    glove.write_text("giraffes 1 0 0\nsleep 0 1 0\n", encoding="utf-8")
    for path in (w2v, glove):
        store = load_vectors(str(path), pre)
        assert store.dim == 3 and store.get("giraffes").tolist() == [1.0, 0.0, 0.0]
    m = _bind(make_method("word_vectors", {"vectors": str(w2v)}))
    assert m.score_pairs(["giraffes"], ["giraffes sleep"])[0] == pytest.approx(1 / np.sqrt(2), abs=1e-6)


def test_missing_spacy_vectors_message(monkeypatch):
    m = _bind(make_method("word_vectors"))
    monkeypatch.setattr(Preprocessor, "backend", property(lambda self: "regex"))
    with pytest.raises(ModelNotAvailableError, match="spacy download"):
        m.score_pairs(["a b"], ["c d"])


@requires("spacy")
def test_spacy_pipeline_without_vectors_message():
    import spacy

    m = _bind(make_method("word_vectors", {"vectors": spacy.blank("en")}))
    with pytest.raises(ModelNotAvailableError, match="no word vectors"):
        m.score_pairs(["giraffes"], ["sleep"])


def test_invalid_parameters():
    with pytest.raises(ConfigurationError):
        make_method("word_vectors", {"weighting": "tfidf"})
    with pytest.raises(ConfigurationError):
        _bind(make_method("word_vectors", {"vectors": 42})).score_pairs(["a"], ["b"])


def test_word_vectors_explanation(method_factory):
    m = method_factory("word_vectors")
    exp = m.explain_pair("Giraffes sleep two hours.", "Giraffes sleep about two hours a day.")
    assert isinstance(exp.alignments, pd.DataFrame) and len(exp.alignments)
    assert set(exp.alignments["side"]) == {"response", "reference"}
    assert exp.alignments.loc[exp.alignments.token == "giraffes", "similarity"].iloc[0] == pytest.approx(1.0)


@requires("ot")
def test_wmd_explanation_flows(method_factory):
    m = method_factory("wmd")
    exp = m.explain_pair("Giraffes sleep two hours.", "The giraffes sleep a lot.", top_k=0)
    flows = exp.alignments
    assert len(flows) and {"weight", "matched_token"} <= set(flows.columns)
    assert flows["weight"].sum() == pytest.approx(1.0, abs=1e-6)  # the whole mass is moved
    assert exp.components["distance"] >= 0
    assert exp.score == pytest.approx(1 / (1 + exp.components["distance"]))


@requires("ot")
def test_wmd_identical_bags_are_one(method_factory):
    m = method_factory("wmd")
    assert m.score_pairs(["sleep giraffes"], ["giraffes sleep"])[0] == pytest.approx(1.0)


def test_assess_end_to_end_word_vectors(demo_en, spacy_vectors_nlp):
    out, assessor = sim.assess(
        demo_en, columns=("text", "info"), method="word_vectors",
        method_params={"vectors": spacy_vectors_nlp}, output="binary", coded=demo_en,
        label_col="label", group_col="topic", language="en", verbose=False, return_assessor=True,
    )
    assert out["predicted_similarity"].dtype == "Int8"
    assert assessor.report_ is not None and 0 <= assessor.report_.metrics["f1"] <= 1
    exp = assessor.explain(demo_en["text"].iloc[0], demo_en["info"].iloc[0])
    assert exp.alignments is not None


@requires("ot")
def test_assess_end_to_end_wmd_gradient(demo_en, spacy_vectors_nlp):
    sub = demo_en[demo_en.topic == demo_en.topic.iloc[0]].head(40)
    out = sim.assess(sub, method="wmd", method_params={"vectors": spacy_vectors_nlp},
                     language="en", verbose=False)
    assert out["predicted_similarity"].between(0, 1).all()
