"""Latent semantic analysis (LSA) and latent Dirichlet allocation (LDA)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._utils.warn import warn
from ..exceptions import ConfigurationError, SimchologyDataWarning
from ..explain.explanation import Explanation
from .base import MethodInfo
from .lexical import TfidfCosineMethod, _VocabularyMethod
from .registry import register_method


@register_method
class LSAMethod(TfidfCosineMethod):
    """Cosine similarity in a latent semantic space (TF-IDF reduced by truncated SVD).

    ``n_components`` (default 300, as is typical for LSA) is reduced automatically when
    the corpus is too small to support that many dimensions. LSA needs a reasonably
    large corpus; pass ``corpus=`` with additional texts from the same domain if the
    coded data are small.
    """

    info = MethodInfo(
        name="lsa", family="topic", description="cosine similarity in an LSA space",
        requires_corpus_fit=True, score_range=(-1.0, 1.0), aliases=("latent_semantic_analysis",),
    )
    defaults = {**TfidfCosineMethod.defaults, "n_components": 300}

    def _after_fit(self, docs):
        from sklearn.decomposition import TruncatedSVD

        corpus_matrix = self._weight(self._counts_from_terms(docs))
        k = int(self.params["n_components"])
        max_k = max(1, min(corpus_matrix.shape) - 1)
        if k > max_k:
            warn(
                f"LSA: the corpus supports at most {max_k} dimensions, so n_components was "
                f"reduced from {k} to {max_k}. For a stable LSA space, fit on a larger corpus "
                "(corpus=...).", SimchologyDataWarning,
            )
            k = max_k
        svd = TruncatedSVD(n_components=k, random_state=self.runtime.seed)
        svd.fit(corpus_matrix)
        self.components_ = svd.components_.astype(np.float64)
        self.n_components_ = k

    def encode(self, texts):
        tfidf = super().encode(texts)
        z = np.asarray(tfidf @ self.components_.T)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return z / norms

    def compare(self, a, b):
        return np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)

    def describe(self):
        return f"cosine similarity in a {getattr(self, 'n_components_', self.params['n_components'])}-dimensional LSA space"

    def explain_pair(self, left, right, top_k=10):
        raise ConfigurationError("LSA dimensions are not interpretable; use 'tfidf' to see term contributions.")

    def _extra_arrays(self):
        return {"components": self.components_}

    def _load_extra(self, z):
        self.components_ = z["components"]
        self.n_components_ = self.components_.shape[0]


@register_method
class LDAMethod(_VocabularyMethod):
    """Similarity of LDA topic distributions (1 - Jensen-Shannon divergence, or exp(-KL)).

    ``divergence="js"`` (default) is symmetric. ``divergence="kl"`` computes
    KL(reference || response), i.e. how badly the response's topic mixture accounts for
    the reference's, and is asymmetric. LDA needs longer texts and a large corpus; it
    is applied to whole responses rather than sentences.
    """

    info = MethodInfo(
        name="lda", family="topic", description="similarity of LDA topic distributions",
        requires_corpus_fit=True, segment_default=False, supports_explain=True,
        aliases=("latent_dirichlet_allocation", "topic_model"),
    )
    defaults = {**_VocabularyMethod.defaults, "n_topics": 20, "divergence": "js", "max_iter": 50}

    def __init__(self, **params):
        super().__init__(**params)
        if self.params["divergence"] not in ("js", "kl"):
            raise ConfigurationError("divergence must be 'js' or 'kl'")

    def _after_fit(self, docs):
        from sklearn.decomposition import LatentDirichletAllocation

        lengths = [len(d) for d in docs]
        if len(docs) < 200 or (lengths and np.median(lengths) < 20):
            warn(
                f"LDA was fitted on {len(docs)} texts with a median of "
                f"{np.median(lengths) if lengths else 0:.0f} words. LDA needs many, longer "
                "texts; with short texts its topic estimates are unreliable.",
                SimchologyDataWarning,
            )
        self._fitted = True
        counts = self._counts_from_terms(docs)
        lda = LatentDirichletAllocation(
            n_components=int(self.params["n_topics"]), max_iter=int(self.params["max_iter"]),
            learning_method="batch", random_state=self.runtime.seed,
        )
        lda.fit(counts)
        self.lda_ = lda

    def encode(self, texts):
        self._ensure_fitted(texts)
        theta = self.lda_.transform(self._counts(texts))
        return theta / theta.sum(axis=1, keepdims=True)

    def compare(self, p, q):
        eps = 1e-12
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        if self.params["divergence"] == "kl":
            kl = np.sum(q * np.log(q / p), axis=1)  # KL(reference || response)
            return np.exp(-kl)
        m = (p + q) / 2
        jsd = 0.5 * np.sum(p * np.log2(p / m), axis=1) + 0.5 * np.sum(q * np.log2(q / m), axis=1)
        return np.clip(1 - jsd, 0.0, 1.0)

    def describe(self):
        kind = "1 - Jensen-Shannon divergence" if self.params["divergence"] == "js" else "exp(-KL divergence)"
        return f"{kind} between {self.params['n_topics']}-topic LDA distributions"

    def topic_words(self, n: int = 8) -> list[list[str]]:
        """The ``n`` most probable words of each topic."""
        terms = sorted(self.vocabulary_, key=self.vocabulary_.get)
        return [[terms[j] for j in np.argsort(-row)[:n]] for row in self.lda_.components_]

    def explain_pair(self, left, right, top_k=10):
        p, q = self.encode([left, right])
        words = self.topic_words(6)
        exp = Explanation(self.name, left, right, float(self.compare(p[None], q[None])[0]))
        for side, dist in (("response", p), ("reference", q)):
            top = np.argsort(-dist)[:3]
            exp.notes.append(f"{side} topics: " + "; ".join(
                f"#{t} ({dist[t]:.0%}: {', '.join(words[t])})" for t in top))
        return exp

    # The fitted LDA model is stored as plain arrays and rebuilt on load (no pickle).
    def _extra_arrays(self):
        lda = self.lda_
        return {"components": lda.components_, "exp_dirichlet": lda.exp_dirichlet_component_,
                "doc_topic_prior": np.array(lda.doc_topic_prior_),
                "topic_word_prior": np.array(lda.topic_word_prior_)}

    def _load_extra(self, z):
        from sklearn.decomposition import LatentDirichletAllocation

        lda = LatentDirichletAllocation(
            n_components=int(self.params["n_topics"]), max_iter=int(self.params["max_iter"]),
            learning_method="batch", random_state=self.runtime.seed,
        )
        lda.components_ = z["components"]
        lda.exp_dirichlet_component_ = z["exp_dirichlet"]
        lda.doc_topic_prior_ = float(z["doc_topic_prior"])
        lda.topic_word_prior_ = float(z["topic_word_prior"])
        lda.n_features_in_ = lda.components_.shape[1]
        lda.n_batch_iter_ = 1
        lda.n_iter_ = 1
        lda.bound_ = 0.0
        self.lda_ = lda

    def save(self, directory: Path) -> None:
        super().save(directory)
