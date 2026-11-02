"""Static word embeddings (word2vec, GloVe, fastText, spaCy vectors) averaged per text.

Each text is represented by the (optionally idf-weighted) mean of the vectors of its
preprocessed words, and texts are compared with cosine similarity. Static embeddings
assign one vector per word type, so they capture synonymy but not word order or the
different senses of polysemous words.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._optional import require
from .._utils.provenance import spacy_model_record
from ..exceptions import ConfigurationError, ModelNotAvailableError
from ..explain.explanation import Explanation
from .base import MethodInfo, RepresentationMethod
from .registry import register_method


class VectorStore:
    """Uniform lookup of static word vectors from spaCy, gensim or a plain mapping."""

    def __init__(self, lookup, dim: int, description: str) -> None:
        self._lookup = lookup
        self.dim = int(dim)
        self.description = description
        self._cache: dict[str, np.ndarray | None] = {}

    def get(self, word: str) -> np.ndarray | None:
        """Vector of ``word``; tries the word as given, then capitalised and upper-cased.

        Preprocessed tokens are lower-cased, while many vector vocabularies (e.g. German
        spaCy vectors) store nouns capitalised.
        """
        if word in self._cache:
            return self._cache[word]
        vec = None
        for form in dict.fromkeys((word, word.capitalize(), word.upper())):
            vec = self._lookup(form)
            if vec is not None:
                break
        self._cache[word] = vec
        return vec


def _from_spacy(nlp, name: str) -> VectorStore:
    vocab = nlp.vocab
    dim = int(getattr(vocab, "vectors_length", 0) or 0)
    if dim == 0 or vocab.vectors.shape[0] == 0:
        raise ModelNotAvailableError(
            f"The spaCy pipeline {name!r} has no word vectors. Install a pipeline with vectors "
            "(the large ones, e.g. `python -m spacy download de_core_news_lg` or "
            "`python -m spacy download en_core_web_lg`), or pass method_params={'vectors': "
            "'<path to word2vec/GloVe/fastText file>'}."
        )

    def lookup(word: str):
        if vocab.has_vector(word):
            v = np.asarray(vocab.get_vector(word), dtype=np.float32)
            return v if np.any(v) else None
        return None

    return VectorStore(lookup, dim, f"spaCy vectors ({name})")


def _from_keyed_vectors(kv, name: str) -> VectorStore:
    ft = hasattr(kv, "buckets") and getattr(kv, "buckets", 0)  # fastText: subword vectors for any word

    def lookup(word: str):
        if word in kv.key_to_index or ft:
            try:
                return np.asarray(kv[word], dtype=np.float32)
            except KeyError:
                return None
        return None

    return VectorStore(lookup, kv.vector_size, f"word vectors ({name})")


def _from_mapping(mapping: Mapping[str, Any]) -> VectorStore:
    if not mapping:
        raise ConfigurationError("The dictionary of word vectors is empty.")
    first = np.asarray(next(iter(mapping.values())), dtype=np.float32)

    def lookup(word: str):
        v = mapping.get(word)
        return None if v is None else np.asarray(v, dtype=np.float32)

    return VectorStore(lookup, first.shape[0], f"word vectors ({len(mapping):,} words)")


def _read_text_vectors(path: Path) -> VectorStore:
    """word2vec text files (``<count> <dim>`` header) and GloVe files (no header)."""
    words: dict[str, int] = {}
    rows: list[np.ndarray] = []
    dim = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh):
            parts = line.rstrip().split(" ")
            if n == 0 and len(parts) == 2 and all(x.isdigit() for x in parts):
                continue  # word2vec header
            if len(parts) < 2:
                continue
            if dim is None:
                dim = len(parts) - 1
            if len(parts) - 1 != dim:
                continue  # skip malformed lines (e.g. words containing spaces)
            if parts[0] not in words:
                words[parts[0]] = len(rows)
                rows.append(np.asarray(parts[1:], dtype=np.float32))
    if not rows:
        raise ModelNotAvailableError(f"No word vectors could be read from {str(path)!r}.")
    matrix = np.vstack(rows)

    def lookup(word: str):
        i = words.get(word)
        return None if i is None else matrix[i]

    return VectorStore(lookup, matrix.shape[1], f"word vectors ({path.name})")


def _load_vector_file(path: str) -> VectorStore:
    p = Path(path)
    if not p.exists():
        raise ModelNotAvailableError(f"The word-vector file {path!r} does not exist.")
    suffix = p.suffix.lower()
    if suffix not in (".bin", ".kv", ".model"):
        return _read_text_vectors(p)
    require("gensim", feature="loading binary word-vector files", extra="static")
    from gensim.models import KeyedVectors

    if suffix == ".bin":
        try:
            from gensim.models.fasttext import load_facebook_vectors

            return _from_keyed_vectors(load_facebook_vectors(str(p)), p.name)
        except Exception:
            kv = KeyedVectors.load_word2vec_format(str(p), binary=True)
            return _from_keyed_vectors(kv, p.name)
    return _from_keyed_vectors(KeyedVectors.load(str(p)), p.name)


def load_vectors(source: Any, preprocessor) -> VectorStore:
    """Resolve the ``vectors`` parameter of word-vector based methods."""
    if isinstance(source, VectorStore):
        return source
    if source is None or source == "spacy":
        if preprocessor.backend != "spacy":
            model = preprocessor.spacy_model_name or "<pipeline>"
            raise ModelNotAvailableError(
                "Word vectors are taken from the spaCy pipeline by default, but no spaCy "
                f"pipeline for {preprocessor.config.language!r} is installed. Install one with "
                f"vectors, e.g. `python -m spacy download {model}`, or pass "
                "method_params={'vectors': '<path to word2vec/GloVe/fastText file>'}."
            )
        return _from_spacy(preprocessor.nlp(), preprocessor.spacy_model_name)
    if isinstance(source, Mapping):
        return _from_mapping(source)
    if hasattr(source, "vocab") and hasattr(source.vocab, "vectors"):
        return _from_spacy(source, getattr(source, "meta", {}).get("name", "custom pipeline"))
    if hasattr(source, "key_to_index"):
        return _from_keyed_vectors(source, "custom KeyedVectors")
    if isinstance(source, (str, os.PathLike)):
        s = str(source)
        if s.startswith("gensim:"):
            require("gensim", feature="downloading word vectors", extra="static")
            import gensim.downloader

            name = s.split(":", 1)[1]
            try:
                return _from_keyed_vectors(gensim.downloader.load(name), name)
            except Exception as exc:
                raise ModelNotAvailableError(f"Could not load gensim vectors {name!r}: {exc}") from exc
        return _load_vector_file(s)
    raise ConfigurationError(
        "vectors must be 'spacy', a path to a word-vector file, 'gensim:<name>', a spaCy "
        "pipeline, gensim KeyedVectors or a dict mapping words to vectors."
    )


class _WordVectorMixin:
    """Tokenisation + vector lookup shared by word_vectors and wmd."""

    _store: VectorStore | None = None

    def store(self) -> VectorStore:
        if self._store is None:
            self._store = load_vectors(self.params["vectors"], self.preprocessor)
        return self._store

    def model_details(self) -> dict[str, dict]:
        source = self.params["vectors"]
        if source is None or source == "spacy":
            if self.preprocessor.backend != "spacy":
                return {}
            return {"word_vectors": spacy_model_record(self.preprocessor.nlp(), self.preprocessor.spacy_model_name)}
        if hasattr(source, "vocab") and hasattr(source.vocab, "vectors"):
            return {"word_vectors": spacy_model_record(source)}
        if isinstance(source, (str, os.PathLike)):
            s = str(source)
            gensim = s.startswith("gensim:")
            record = {"name": s.split(":", 1)[1] if gensim else s, "source": "gensim" if gensim else "file",
                      "version": None}
            if self._store is not None:
                record["vector_dimensions"] = self._store.dim
            return {"word_vectors": record}
        return {}  # vectors passed as Python objects

    def token_vectors(self, text: str) -> tuple[list[str], np.ndarray]:
        """Tokens of ``text`` that have a vector, and their vectors (n x dim)."""
        cache = self.__dict__.setdefault("_tv_cache", {})
        if text in cache:
            return cache[text]
        store = self.store()
        toks, vecs = [], []
        for tok in self.preprocessor.tokens([text], "surface")[0]:
            v = store.get(tok)
            if v is not None:
                toks.append(tok)
                vecs.append(v)
        mat = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, store.dim), np.float32)
        cache[text] = (toks, mat)
        return toks, mat


def _unit(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


@register_method
class StaticEmbeddingMethod(_WordVectorMixin, RepresentationMethod):
    """Cosine similarity of averaged static word vectors.

    Parameters
    ----------
    vectors:
        ``"spacy"`` (default: the vectors of the language's spaCy pipeline, which must be
        a large ``*_lg`` pipeline), a path to a word2vec/GloVe text or binary file or a
        fastText ``.bin`` file, ``"gensim:<name>"`` (e.g. ``"gensim:glove-wiki-gigaword-300"``),
        or a spaCy pipeline, gensim ``KeyedVectors`` or dict of word vectors.
    weighting:
        ``"mean"`` (plain average, default) or ``"idf"`` (words weighted by their inverse
        document frequency in the corpus, so frequent words count less).
    """

    info = MethodInfo(
        name="word_vectors", family="static",
        description="cosine similarity of averaged static word vectors",
        requires_corpus_fit=True, score_range=(-1.0, 1.0), supports_explain=True,
        extras=("nlp",), aliases=("word", "static", "static_embeddings"),
    )
    defaults = {"vectors": "spacy", "weighting": "mean"}

    def __init__(self, **params):
        super().__init__(**params)
        if self.params["weighting"] not in ("mean", "idf"):
            raise ConfigurationError("weighting must be 'mean' or 'idf'")
        self.idf_: dict[str, float] = {}
        self.n_docs_ = 0

    def fit(self, corpus=None, train=None, dev=None):
        if self.params["weighting"] == "idf":
            if corpus is None and train is not None:
                corpus = list(dict.fromkeys([*train["left"], *train["right"]]))
            docs = self.preprocessor.tokens(list(corpus or []), "surface")
            df: Counter = Counter()
            for d in docs:
                df.update(set(d))
            self.n_docs_ = len(docs)
            self.idf_ = {w: math.log((1 + self.n_docs_) / (1 + c)) + 1 for w, c in df.items()}
        self._fitted = True
        return self

    def _weight(self, word: str) -> float:
        if self.params["weighting"] != "idf":
            return 1.0
        return self.idf_.get(word, math.log(1 + self.n_docs_) + 1)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(corpus=texts)
        out = np.zeros((len(texts), self.store().dim), dtype=np.float64)
        for i, t in enumerate(texts):
            toks, vecs = self.token_vectors(t)
            if len(toks):
                w = np.array([self._weight(x) for x in toks])
                out[i] = (w[:, None] * vecs).sum(axis=0) / w.sum()
        return _unit(out)

    def compare(self, a, b) -> np.ndarray:
        return np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)

    def describe(self) -> str:
        how = "idf-weighted average" if self.params["weighting"] == "idf" else "average"
        src = self._store.description if self._store is not None else "static word vectors"
        return f"cosine similarity of the {how} of {src}"

    def explain_pair(self, left: str, right: str, top_k: int = 10) -> Explanation:
        ta, va = self.token_vectors(left)
        tb, vb = self.token_vectors(right)
        score = float(self.score_pairs([left], [right])[0])
        exp = Explanation(self.name, left, right, score, response_tokens=ta, reference_tokens=tb)
        if not ta or not tb:
            exp.notes.append("one of the texts has no words with a vector")
            return exp
        sim = _unit(va) @ _unit(vb).T
        rows = [{"side": "response", "token": ta[i], "matched_token": tb[int(np.argmax(sim[i]))],
                 "similarity": float(sim[i].max())} for i in range(len(ta))]
        rows += [{"side": "reference", "token": tb[j], "matched_token": ta[int(np.argmax(sim[:, j]))],
                  "similarity": float(sim[:, j].max())} for j in range(len(tb))]
        exp.alignments = pd.DataFrame(rows, columns=["side", "token", "matched_token", "similarity"])
        exp.matrix = sim
        exp.notes.append("each word with its most similar word in the other text (cosine of word vectors)")
        return exp

    def save(self, directory: Path) -> None:
        if self.params["weighting"] == "idf":
            import json

            directory.mkdir(parents=True, exist_ok=True)
            (directory / "idf.json").write_text(
                json.dumps({"n_docs": self.n_docs_, "idf": self.idf_}, ensure_ascii=False), encoding="utf-8")

    def load(self, directory: Path, *, trust: bool = False) -> None:
        f = directory / "idf.json"
        if f.exists():
            import json

            d = json.loads(f.read_text(encoding="utf-8"))
            self.n_docs_, self.idf_ = d["n_docs"], d["idf"]
        self._fitted = True

