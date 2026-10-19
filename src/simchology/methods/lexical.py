"""Bag-of-words, TF-IDF and BM25 representations compared with cosine similarity / BM25 scoring.

The vocabulary and document frequencies are learnt from a corpus (by default all
sentences of the responses plus the references, as in the original coding scripts)
and then frozen, so later predictions do not change earlier scores.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..exceptions import ConfigurationError
from ..explain.explanation import Explanation
from .base import MethodInfo, RepresentationMethod
from .registry import register_method


def _row_dot(a: sp.csr_matrix, b: sp.csr_matrix) -> np.ndarray:
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def _l2(m: sp.csr_matrix) -> sp.csr_matrix:
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms) @ m


class _VocabularyMethod(RepresentationMethod):
    """Shared vocabulary handling for count-based representations."""

    defaults = {"unit": "auto", "ngram_range": (1, 1), "min_df": 1}

    def __init__(self, **params):
        super().__init__(**params)
        lo, hi = tuple(self.params["ngram_range"])
        if not 1 <= lo <= hi:
            raise ConfigurationError("ngram_range must be (min_n, max_n) with 1 <= min_n <= max_n")
        self.vocabulary_: dict[str, int] = {}
        self.df_: np.ndarray = np.zeros(0)
        self.n_docs_: int = 0
        self.avgdl_: float = 0.0

    def _terms(self, texts: list[str]) -> list[list[str]]:
        lo, hi = tuple(self.params["ngram_range"])
        out = []
        for toks in self.preprocessor.tokens(texts, self.params["unit"]):
            terms = []
            for n in range(lo, hi + 1):
                terms += [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
            out.append(terms)
        return out

    def fit(self, corpus=None, train=None, dev=None):
        if corpus is None:
            if train is None:
                raise ConfigurationError(f"The method '{self.name}' needs a corpus to learn its vocabulary.")
            corpus = list(dict.fromkeys([*train["left"], *train["right"]]))
        docs = self._terms(list(corpus))
        df: Counter = Counter()
        for terms in docs:
            df.update(set(terms))
        min_df = int(self.params["min_df"])
        vocab = sorted(t for t, c in df.items() if c >= min_df)
        self.vocabulary_ = {t: i for i, t in enumerate(vocab)}
        self.df_ = np.array([df[t] for t in vocab], dtype=float)
        self.n_docs_ = len(docs)
        self.avgdl_ = float(np.mean([len(d) for d in docs])) if docs else 0.0
        self._fitted = True
        self._after_fit(docs)
        return self

    def _after_fit(self, docs: list[list[str]]) -> None:
        pass

    def _ensure_fitted(self, texts: list[str]) -> None:
        if not self._fitted:
            self.fit(corpus=texts)

    def _counts_from_terms(self, docs: list[list[str]]) -> sp.csr_matrix:
        rows, cols, vals = [], [], []
        for i, terms in enumerate(docs):
            for t, c in Counter(terms).items():
                j = self.vocabulary_.get(t)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(c)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocabulary_)), dtype=float)

    def _counts(self, texts: list[str]) -> sp.csr_matrix:
        return self._counts_from_terms(self._terms(texts))

    def describe(self) -> str:
        return f"{self.info.description} (vocabulary of {len(self.vocabulary_):,} terms)"

    # persistence: plain JSON + npz, no pickle
    def _extra_arrays(self) -> dict[str, np.ndarray]:
        return {}

    def _load_extra(self, arrays) -> None:
        pass

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        terms = sorted(self.vocabulary_, key=self.vocabulary_.get)
        (directory / "vocabulary.json").write_text(json.dumps(terms, ensure_ascii=False), encoding="utf-8")
        np.savez(directory / "state.npz", df=self.df_, n_docs=self.n_docs_, avgdl=self.avgdl_,
                 **self._extra_arrays())

    def load(self, directory: Path, *, trust: bool = False) -> None:
        terms = json.loads((directory / "vocabulary.json").read_text(encoding="utf-8"))
        self.vocabulary_ = {t: i for i, t in enumerate(terms)}
        with np.load(directory / "state.npz") as z:
            self.df_ = z["df"]
            self.n_docs_ = int(z["n_docs"])
            self.avgdl_ = float(z["avgdl"])
            self._load_extra(z)
        self._fitted = True

    def _term_contributions(self, left: str, right: str, a: sp.csr_matrix, b: sp.csr_matrix,
                            score: float, top_k: int) -> Explanation:
        terms = sorted(self.vocabulary_, key=self.vocabulary_.get)
        contrib = a.multiply(b).tocoo()
        rows = sorted(
            ({"side": "both", "token": terms[j], "matched_token": terms[j], "similarity": float(v)}
             for j, v in zip(contrib.col, contrib.data)),
            key=lambda r: -r["similarity"],
        )
        import pandas as pd

        exp = Explanation(self.name, left, right, score,
                          alignments=pd.DataFrame(rows[:top_k] if top_k else rows,
                                                  columns=["side", "token", "matched_token", "similarity"]))
        exp.notes.append("similarity = sum of the listed per-term contributions")
        return exp


@register_method
class BowCosineMethod(_VocabularyMethod):
    """Cosine similarity of bag-of-words count vectors (``binary=True`` for presence/absence)."""

    info = MethodInfo(
        name="bow", family="lexical", description="cosine similarity of bag-of-words vectors",
        requires_corpus_fit=True, supports_explain=True, aliases=("bag_of_words", "count"),
    )
    defaults = {**_VocabularyMethod.defaults, "binary": False}

    def encode(self, texts):
        self._ensure_fitted(texts)
        m = self._counts(texts)
        if self.params["binary"]:
            m.data[:] = 1.0
        return _l2(m.tocsr()).tocsr()

    def compare(self, a, b):
        return np.clip(_row_dot(a, b), 0.0, 1.0)

    def explain_pair(self, left, right, top_k=10):
        a, b = self.encode([left, right])
        return self._term_contributions(left, right, a, b, float(self.compare(a, b)[0]), top_k)


@register_method
class TfidfCosineMethod(_VocabularyMethod):
    """Cosine similarity of TF-IDF vectors (idf as in scikit-learn: ln((1+N)/(1+df)) + 1)."""

    info = MethodInfo(
        name="tfidf", family="lexical", description="cosine similarity of TF-IDF vectors",
        requires_corpus_fit=True, supports_explain=True, aliases=("tf_idf", "cosine", "tfidf_cosine"),
    )
    defaults = {**_VocabularyMethod.defaults, "sublinear_tf": False, "smooth_idf": True}

    def idf(self) -> np.ndarray:
        n, df = self.n_docs_, self.df_
        if self.params["smooth_idf"]:
            return np.log((1 + n) / (1 + df)) + 1
        return np.log(n / df) + 1

    def _weight(self, counts: sp.csr_matrix) -> sp.csr_matrix:
        m = counts.copy()
        if self.params["sublinear_tf"]:
            m.data = 1 + np.log(m.data)
        m = m @ sp.diags(self.idf())
        return _l2(m.tocsr()).tocsr()

    def encode(self, texts):
        self._ensure_fitted(texts)
        return self._weight(self._counts(texts))

    def compare(self, a, b):
        return np.clip(_row_dot(a, b), 0.0, 1.0)

    def explain_pair(self, left, right, top_k=10):
        a, b = self.encode([left, right])
        return self._term_contributions(left, right, a, b, float(self.compare(a, b)[0]), top_k)


@register_method
class BM25Method(_VocabularyMethod):
    """Okapi BM25 with the reference as query and the response as document.

    The score is divided by its maximum possible value (every query term saturated), so
    it lies in [0, 1): the share of the reference's (idf-weighted) terms the response
    contains, with term-frequency saturation (``k1``) and length normalisation (``b``).
    """

    info = MethodInfo(
        name="bm25", family="lexical",
        description="normalised Okapi BM25 score (reference as query)",
        symmetric=False, requires_corpus_fit=True, supports_explain=True, aliases=("okapi_bm25",),
    )
    defaults = {**_VocabularyMethod.defaults, "k1": 1.5, "b": 0.75}

    def idf(self) -> np.ndarray:
        n, df = self.n_docs_, self.df_
        return np.log(1 + (n - df + 0.5) / (df + 0.5))

    def encode(self, texts):
        self._ensure_fitted(texts)
        return self._counts(texts)

    def _parts(self, a: sp.csr_matrix, b: sp.csr_matrix):
        k1, bb = float(self.params["k1"]), float(self.params["b"])
        a = a.tocsr()
        dl = np.asarray(a.sum(axis=1)).ravel()
        avgdl = self.avgdl_ or 1.0
        denom = k1 * (1 - bb + bb * dl / avgdl)
        rows = np.repeat(np.arange(a.shape[0]), np.diff(a.indptr))
        sat = sp.csr_matrix((a.data * (k1 + 1) / (a.data + denom[rows]), a.indices, a.indptr), shape=a.shape)
        q = (b > 0).astype(float).multiply(self.idf()).tocsr()
        return sat.multiply(q).tocsr(), np.asarray(q.sum(axis=1)).ravel() * (k1 + 1)

    def compare(self, a, b):
        contrib, upper = self._parts(a, b)
        score = np.asarray(contrib.sum(axis=1)).ravel()
        return np.where(upper > 0, score / np.where(upper > 0, upper, 1), 0.0)

    def explain_pair(self, left, right, top_k=10):
        a, b = self.encode([left]), self.encode([right])
        contrib, upper = self._parts(a, b)
        score = float(self.compare(a, b)[0])
        return self._term_contributions(left, right, contrib / (upper[0] or 1), sp.csr_matrix(
            np.ones(contrib.shape)), score, top_k)
