"""Property and value tests for the methods that need no optional dependencies."""

import numpy as np
import pytest

from simchology.methods.registry import make_method
from simchology.text.preprocess import PreprocessConfig, Preprocessor

CORE = ["jaccard", "levenshtein", "osa", "damerau_levenshtein", "bow", "tfidf", "bm25", "lsa", "lda"]


def _bound(name, corpus, **params):
    m = make_method(name, params)
    m.bind(Preprocessor(PreprocessConfig.from_options("en", {"backend": "regex"}), verbose=False),
           m.runtime)
    if m.info.requires_corpus_fit:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(corpus=corpus)
    return m


@pytest.fixture(scope="module")
def corpus(demo_en):
    return list(dict.fromkeys(demo_en["text"].tolist() + demo_en["info"].tolist()))


@pytest.mark.parametrize("name", CORE)
def test_properties(name, corpus, demo_en):
    m = _bound(name, corpus)
    left = demo_en["text"].head(40).tolist()
    right = demo_en["info"].head(40).tolist()
    s = m.score_pairs(left, right)
    assert s.shape == (40,) and np.all(np.isfinite(s))
    lo, hi = m.score_range()
    assert lo is None or s.min() >= lo - 1e-9
    assert hi is None or s.max() <= hi + 1e-9
    # duplicates get identical scores; order is preserved
    s2 = m.score_pairs(left[::-1] + left[:3], right[::-1] + right[:3])
    assert np.allclose(s2[:40], s[::-1]) and np.allclose(s2[40:], s[:3])
    if m.info.symmetric:
        assert np.allclose(m.score_pairs(right, left), s)
        same = m.score_pairs(right, right)
        assert np.all(same >= s - 1e-9)


def test_jaccard_values():
    m = _bound("jaccard", None)
    s = m.score_pairs(["giraffes are tall animals"], ["tall giraffes sleep"])
    assert s[0] == pytest.approx(2 / 4)  # {giraffes, tall, animals} vs {tall, giraffes, sleep}
    c = _bound("jaccard", None, variant="containment")
    assert c.score_pairs(["giraffes are tall animals"], ["tall giraffes sleep"])[0] == pytest.approx(2 / 3)
    exp = m.explain_pair("giraffes are tall animals", "tall giraffes sleep")
    assert exp.shared == ["giraffes", "tall"] and exp.only_reference == ["sleep"]


def test_edit_similarity_values():
    m = _bound("levenshtein", None, preprocess=False)
    assert m.score_pairs(["kitten"], ["sitting"])[0] == pytest.approx(1 - 3 / 7)
    raw = _bound("levenshtein", None, preprocess=False, normalize=False)
    assert raw.score_pairs(["kitten"], ["sitting"])[0] == -3
    dl = _bound("damerau_levenshtein", None, preprocess=False, normalize=False)
    lv = _bound("levenshtein", None, preprocess=False, normalize=False)
    assert dl.score_pairs(["ab"], ["ba"])[0] == -1 and lv.score_pairs(["ab"], ["ba"])[0] == -2
    tok = _bound("levenshtein", None, level="token", preprocess=False)
    assert tok.score_pairs(["giraffes are very tall"], ["giraffes are tall"])[0] == pytest.approx(3 / 4)
    # with preprocessing, stop words ("are", "very") are removed before comparing
    pre = _bound("levenshtein", None, level="token")
    assert pre.score_pairs(["giraffes are very tall"], ["giraffes are tall"])[0] == 1.0


def test_tfidf_matches_sklearn(corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer

    m = _bound("tfidf", corpus)
    prep = m.preprocessor
    vec = TfidfVectorizer(analyzer=lambda t: list(prep.tokens([t])[0]))
    x = vec.fit_transform(corpus)
    a, b = corpus[0], corpus[5]
    expected = float((x[0] @ x[5].T).toarray()[0, 0])
    assert m.score_pairs([a], [b])[0] == pytest.approx(expected)


def test_bm25_is_bounded_and_rewards_coverage(corpus):
    m = _bound("bm25", corpus)
    ref = "giraffes sleep two hours"
    full, partial = m.score_pairs(["giraffes sleep about two hours a day", "giraffes are tall"], [ref, ref])
    assert 0 < partial < full < 1


def test_lsa_clamps_components():
    m = make_method("lsa")
    m.bind(Preprocessor(PreprocessConfig.from_options("en", {"backend": "regex"}), verbose=False),
           m.runtime)
    with pytest.warns(Warning, match="reduced"):
        m.fit(corpus=["alpha beta gamma", "beta gamma delta", "gamma delta epsilon"])
    assert m.n_components_ <= 2


def test_explanations_for_lexical(corpus):
    m = _bound("tfidf", corpus)
    exp = m.explain_pair("giraffes sleep two hours", "giraffes sleep about two hours a day")
    assert set(exp.alignments["token"]) >= {"giraffes", "sleep"}
    assert exp.alignments["similarity"].sum() == pytest.approx(exp.score)
