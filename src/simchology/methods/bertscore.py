"""BERTScore (Zhang et al., 2020): greedy alignment of contextual token embeddings.

Each token of one text is matched to the most similar token (cosine of contextual
embeddings) of the other text:

* **precision** - mean over the *response* tokens of their best match in the reference;
* **recall**    - mean over the *reference* tokens of their best match in the response,
  i.e. how much of the reference is covered by the response;
* **F1**        - their harmonic mean (symmetric).

Raw BERTScore values fall in a narrow, model-dependent range (often 0.6-0.9 even for
unrelated texts), so they should be interpreted relative to a baseline of dissimilar
pairs (the default for binary output), not in absolute terms. Use :meth:`explain` to
inspect which words were matched.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .._utils.logging import note
from .._utils.provenance import hf_model_record
from ..config import DEFAULT_MODELS
from ..exceptions import ConfigurationError
from ..explain.explanation import Explanation
from ._torch_utils import (
    auto_batch_size,
    batched,
    dtype_kwargs,
    import_torch,
    import_transformers,
    load_model,
    resolve_torch_device,
)
from .base import MethodInfo, RepresentationMethod
from .registry import register_method

# Recommended layers from the bert-score package (best correlation with human judgments).
RECOMMENDED_LAYERS: dict[str, int] = {
    "bert-base-uncased": 9, "bert-large-uncased": 18, "bert-base-cased": 9, "bert-large-cased": 18,
    "bert-base-multilingual-cased": 9, "bert-base-multilingual-uncased": 9, "bert-base-chinese": 8,
    "roberta-base": 10, "roberta-large": 17, "roberta-large-mnli": 19,
    "xlm-roberta-base": 9, "xlm-roberta-large": 17,
    "distilbert-base-uncased": 5, "distilbert-base-multilingual-cased": 5,
    "distilroberta-base": 5, "deberta-base": 9, "deberta-large": 16, "deberta-xlarge-mnli": 40,
    "deberta-v3-base": 9, "deberta-v3-large": 17, "scibert_scivocab_uncased": 8,
    "xlnet-base-cased": 5, "xlnet-large-cased": 7, "electra-base-discriminator": 9,
    "electra-large-discriminator": 24,
}


@dataclass
class _TokenEmbedding:
    vectors: np.ndarray  # (n_tokens, dim), l2-normalised, float16
    token_ids: np.ndarray  # (n_tokens,)
    word_ids: list[int | None]
    offsets: list[tuple[int, int]]
    text: str


@register_method
class BERTScoreMethod(RepresentationMethod):
    """BERTScore similarity (precision, recall or F1 of greedy token alignment).

    Parameters
    ----------
    model:
        Hugging Face encoder (default ``bert-base-multilingual-cased``; for higher quality
        ``xlm-roberta-large``).
    layer:
        Hidden layer whose token embeddings are compared; ``"auto"`` uses the layer
        recommended by the bert-score package for known models, else 60% of the depth.
    score:
        ``"f1"`` (symmetric, default), ``"precision"`` (how much of the response is
        supported by the reference) or ``"recall"`` (how much of the reference is covered
        by the response).
    idf:
        Weight tokens by inverse document frequency in the corpus, so rare words count more.
    """

    info = MethodInfo(
        name="bertscore", family="alignment",
        description="BERTScore (greedy alignment of contextual token embeddings)",
        requires_corpus_fit=True, score_range=(-1.0, 1.0), preprocessing="light",
        cost_tier="moderate", supports_explain=True, extras=("embeddings",),
        requires_modules=("torch", "transformers"), aliases=("bert_score",),
    )
    defaults = {"model": DEFAULT_MODELS["bertscore"], "layer": "auto", "score": "f1", "idf": False,
                "batch_size": "auto", "max_length": 510, "revision": None, "trust_remote_code": False}

    def __init__(self, **params):
        super().__init__(**params)
        if self.params["score"] not in ("f1", "precision", "recall"):
            raise ConfigurationError("score must be 'f1', 'precision' or 'recall'")
        self.idf_: dict[int, float] | None = None
        self.idf_default_: float = 1.0
        self._emb: dict[str, _TokenEmbedding] = {}
        self._hf = None
        self.layer_: int | None = None

    @property
    def model_id(self) -> str:
        return str(self.params["model"])

    # -- model -----------------------------------------------------------------------
    def _device(self) -> str:
        return resolve_torch_device(self.runtime.device)

    def _load(self):
        if self._hf is not None:
            return self._hf
        transformers = import_transformers("the method 'bertscore'")
        kw = {"revision": self.params["revision"], "trust_remote_code": bool(self.params["trust_remote_code"])}
        tok = load_model(lambda: transformers.AutoTokenizer.from_pretrained(self.model_id, **kw),
                         self.model_id, "tokenizer")
        model = load_model(lambda: transformers.AutoModel.from_pretrained(self.model_id, **kw, **dtype_kwargs(None)),
                           self.model_id, "model")
        model.eval().to(self._device())
        n_layers = _n_layers(model.config)
        if self.params["layer"] == "auto":
            layer, known = recommended_layer(self.model_id, n_layers)
            if not known:
                note(f"BERTScore: no recommended layer known for {self.model_id!r}; using layer "
                     f"{layer} of {n_layers}. Set method_params={{'layer': ...}} to choose.",
                     self.runtime.verbose)
        else:
            layer = int(self.params["layer"])
        if n_layers and layer > n_layers:
            layer = n_layers
        self.layer_ = layer
        self._hf = (tok, model)
        return self._hf

    def _max_tokens(self, tok, model) -> int:
        limit = int(self.params["max_length"]) + 2
        pos = getattr(model.config, "max_position_embeddings", None)
        if isinstance(pos, int) and pos > 0:
            limit = min(limit, pos)
        mml = getattr(tok, "model_max_length", None)
        if isinstance(mml, int) and 0 < mml < 10**6:
            limit = min(limit, mml)
        return limit

    def _embed(self, texts: list[str]) -> None:
        todo = [t for t in dict.fromkeys(texts) if t not in self._emb]
        if not todo:
            return
        torch = import_torch("the method 'bertscore'")
        tok, model = self._load()
        limit = self._max_tokens(tok, model)
        bs = self.params["batch_size"]
        bs = auto_batch_size(self._device()) if bs == "auto" else int(bs)
        fast = bool(getattr(tok, "is_fast", False))
        with torch.no_grad():
            for chunk in batched(todo, bs):
                kwargs = {"return_offsets_mapping": True} if fast else {}
                enc = tok(list(chunk), padding=True, truncation=True, max_length=limit,
                          return_tensors="pt", return_special_tokens_mask=True, **kwargs)
                offsets = enc.pop("offset_mapping", None)
                special = enc.pop("special_tokens_mask")
                inputs = {k: v.to(model.device) for k, v in enc.items()}
                out = model(**inputs, output_hidden_states=True)
                hidden = out.hidden_states[self.layer_] if out.hidden_states is not None else out.last_hidden_state
                hidden = hidden.float().cpu().numpy()
                for b, text in enumerate(chunk):
                    keep = (enc["attention_mask"][b].numpy() == 1) & (special[b].numpy() == 0)
                    vecs = hidden[b][keep]
                    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    word_ids_all = enc.word_ids(b) if fast else [None] * len(keep)
                    word_ids = [w for w, k in zip(word_ids_all, keep) if k]
                    offs = ([tuple(int(x) for x in o) for o, k in zip(offsets[b].tolist(), keep) if k]
                            if offsets is not None else [(0, 0)] * int(keep.sum()))
                    self._emb[text] = _TokenEmbedding(
                        (vecs / norms).astype(np.float16), enc["input_ids"][b].numpy()[keep],
                        word_ids, offs, text,
                    )

    # -- fitting (idf) -----------------------------------------------------------------
    def fit(self, corpus=None, train=None, dev=None):
        if self.params["idf"]:
            if corpus is None and train is not None:
                corpus = list(dict.fromkeys(train["right"]))
            self._fit_idf(list(corpus or []))
        self._fitted = True
        return self

    def _fit_idf(self, corpus: list[str]) -> None:
        tok, _ = self._load()
        df: Counter = Counter()
        for ids in tok(corpus, add_special_tokens=False, truncation=False)["input_ids"]:
            df.update(set(ids))
        m = len(corpus)
        self.idf_ = {int(k): math.log((m + 1) / (v + 1)) for k, v in df.items()}
        self.idf_default_ = math.log(m + 1)

    def _weights(self, emb: _TokenEmbedding) -> np.ndarray:
        if not self.params["idf"] or self.idf_ is None:
            return np.ones(len(emb.token_ids))
        return np.array([self.idf_.get(int(i), self.idf_default_) for i in emb.token_ids], dtype=float)

    # -- scoring -------------------------------------------------------------------------
    def encode(self, texts: list[str]) -> list[_TokenEmbedding]:
        if self.params["idf"] and self.idf_ is None:
            self._fit_idf(list(dict.fromkeys(texts)))
        self._embed(texts)
        return [self._emb[t] for t in texts]

    def _prf(self, a: _TokenEmbedding, b: _TokenEmbedding) -> tuple[float, float, float, np.ndarray]:
        if not len(a.token_ids) or not len(b.token_ids):
            return 0.0, 0.0, 0.0, np.zeros((len(a.token_ids), len(b.token_ids)))
        sim = np.clip(a.vectors.astype(np.float32) @ b.vectors.astype(np.float32).T, -1.0, 1.0)
        wa, wb = self._weights(a), self._weights(b)
        p = float((sim.max(axis=1) * wa).sum() / (wa.sum() or 1.0))
        r = float((sim.max(axis=0) * wb).sum() / (wb.sum() or 1.0))
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f, sim

    def compare(self, a, b) -> np.ndarray:
        which = {"precision": 0, "recall": 1, "f1": 2}[self.params["score"]]
        return np.clip(np.array([self._prf(x, y)[which] for x, y in zip(a, b)], dtype=float), -1.0, 1.0)

    # -- explanations ----------------------------------------------------------------------
    @staticmethod
    def _words(emb: _TokenEmbedding, tok) -> tuple[list[str], list[list[int]]]:
        """Group token positions into words (via the fast tokenizer's word ids)."""
        if any(w is not None for w in emb.word_ids):
            groups: dict[int, list[int]] = {}
            for pos, w in enumerate(emb.word_ids):
                groups.setdefault(w if w is not None else -1 - pos, []).append(pos)
            words, members = [], []
            for _, positions in groups.items():
                start = min(emb.offsets[p][0] for p in positions)
                end = max(emb.offsets[p][1] for p in positions)
                words.append(emb.text[start:end] if end > start else tok.convert_ids_to_tokens(
                    [int(emb.token_ids[positions[0]])])[0])
                members.append(positions)
            return words, members
        pieces = tok.convert_ids_to_tokens([int(i) for i in emb.token_ids])
        return list(pieces), [[i] for i in range(len(pieces))]

    def explain_pair(self, left: str, right: str, top_k: int = 10) -> Explanation:
        a, b = self.encode([left, right])
        p, r, f, sim = self._prf(a, b)
        tok, _ = self._load()
        wa, ma = self._words(a, tok)
        wb, mb = self._words(b, tok)
        word_sim = np.zeros((len(wa), len(wb)))
        for i, pa in enumerate(ma):
            for j, pb in enumerate(mb):
                word_sim[i, j] = sim[np.ix_(pa, pb)].max() if len(pa) and len(pb) else 0.0
        rows = []
        for i, w in enumerate(wa):
            if len(wb):
                j = int(np.argmax(word_sim[i]))
                rows.append({"side": "response", "token": w, "matched_token": wb[j], "similarity": float(word_sim[i, j])})
        for j, w in enumerate(wb):
            if len(wa):
                i = int(np.argmax(word_sim[:, j]))
                rows.append({"side": "reference", "token": w, "matched_token": wa[i], "similarity": float(word_sim[i, j])})
        score = {"precision": p, "recall": r, "f1": f}[self.params["score"]]
        exp = Explanation(
            self.name, left, right, float(score), components={"precision": p, "recall": r, "f1": f},
            alignments=pd.DataFrame(rows, columns=["side", "token", "matched_token", "similarity"]),
            matrix=word_sim, response_tokens=wa, reference_tokens=wb,
        )
        exp.notes.append("each word with its best-matching word in the other text; precision "
                         "averages the response rows, recall the reference rows (token level)")
        if self.layer_ is not None:
            exp.notes.append(f"{self.model_id}, layer {self.layer_}")
        return exp

    # -- metadata ------------------------------------------------------------------------
    def resource_request(self, n_pairs: int, task: str = "inference"):
        from ..hardware.checks import ResourceRequest

        return ResourceRequest(
            "inference", model_id=self.model_id, seq_len=min(512, int(self.params["max_length"]) + 2),
            batch_size=8, n_items=n_pairs, description=f"computing BERTScore with {self.model_id!r}",
        )

    def model_details(self) -> dict[str, dict]:
        return {"bertscore_model": hf_model_record(self.model_id, self.params["revision"])}

    def describe(self) -> str:
        layer = f", layer {self.layer_}" if self.layer_ is not None else ""
        idf = " with idf weighting" if self.params["idf"] else ""
        return f"BERTScore {self.params['score']}{idf} ({self.model_id}{layer})"

    def save(self, directory: Path) -> None:
        if self.idf_ is not None:
            import json

            directory.mkdir(parents=True, exist_ok=True)
            (directory / "idf.json").write_text(json.dumps(
                {"idf": {str(k): v for k, v in self.idf_.items()}, "default": self.idf_default_}), encoding="utf-8")

    def load(self, directory: Path, *, trust: bool = False) -> None:
        f = directory / "idf.json"
        if f.exists():
            import json

            d = json.loads(f.read_text(encoding="utf-8"))
            self.idf_ = {int(k): float(v) for k, v in d["idf"].items()}
            self.idf_default_ = float(d["default"])
        self._fitted = True


def recommended_layer(model_id: str, n_layers: int | None) -> tuple[int, bool]:
    """(layer, known): bert-score's recommended layer, else 60% of the model's depth."""
    key = str(model_id).rstrip("/").split("/")[-1]
    if key in RECOMMENDED_LAYERS:
        return RECOMMENDED_LAYERS[key], True
    return (max(1, round(0.6 * n_layers)) if n_layers else -1), False


def _n_layers(config) -> int | None:
    cfg = getattr(config, "text_config", None) or config
    for name in ("num_hidden_layers", "n_layers", "num_layers", "n_layer"):
        v = getattr(cfg, name, None)
        if isinstance(v, int):
            return v
    return None
