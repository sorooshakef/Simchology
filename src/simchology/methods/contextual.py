"""Contextualised sentence embeddings compared with cosine similarity.

* ``sbert``          - Sentence-Transformers models fine-tuned for sentence similarity
  (default: a multilingual paraphrase model);
* ``hf_embedding``   - any Hugging Face encoder with mean / CLS / last-token pooling;
* ``llm_embedding``  - embedding models built on large language models (default:
  Qwen3-Embedding-0.6B), more accurate but considerably more expensive.

Contextual embeddings occupy a narrow region of the vector space, so even unrelated
texts get clearly positive cosine similarities. Interpret scores relative to a baseline
(the default for binary output) rather than in absolute terms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._utils.hashing import stable_hash
from .._utils.warn import warn
from ..config import DEFAULT_MODELS
from ..exceptions import ConfigurationError, SimchologyDataWarning
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

# Instruction recommended for symmetric semantic-similarity tasks with Qwen3-Embedding.
LLM_EMBEDDING_PROMPT = "Instruct: Retrieve semantically similar text.\nQuery: "


class _EmbeddingCacheMixin:
    """In-memory (and optional on-disk) cache of one embedding per text."""

    def _cache(self) -> dict[str, np.ndarray]:
        return self.__dict__.setdefault("_emb_cache", {})

    def _cache_key(self) -> str:
        return stable_hash({k: v for k, v in self.params.items() if k not in ("batch_size", "cache_dir")}
                           | {"class": type(self).__name__})

    def _disk_path(self) -> Path | None:
        d = self.params.get("cache_dir")
        return None if not d else Path(d) / f"simchology-embeddings-{self._cache_key()}.npz"

    def _load_disk(self) -> None:
        path = self._disk_path()
        if path is None or self.__dict__.get("_disk_loaded"):
            return
        self._disk_loaded = True
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                for t, v in zip(z["texts"].tolist(), z["embeddings"]):
                    self._cache().setdefault(t, v)

    def _save_disk(self) -> None:
        path = self._disk_path()
        if path is None or not self._cache():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        texts = list(self._cache())
        np.savez(path, texts=np.array(texts, dtype=str), embeddings=np.vstack([self._cache()[t] for t in texts]))

    def _embed_uncached(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load_disk()
        cache = self._cache()
        todo = [t for t in dict.fromkeys(texts) if t not in cache]
        if todo:
            emb = np.asarray(self._embed_uncached(todo), dtype=np.float32)
            for t, v in zip(todo, emb):
                cache[t] = v
            self._save_disk()
        out = np.vstack([cache[t] for t in texts]).astype(np.float64)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def compare(self, a, b) -> np.ndarray:
        return np.clip(np.einsum("ij,ij->i", a, b), -1.0, 1.0)

    def _warn_truncation(self, lengths, limit: int | None) -> None:
        if not limit or not len(lengths):
            return
        share = float(np.mean(np.asarray(lengths) > limit))
        if share > 0.05:
            warn(
                f"{share:.0%} of the texts are longer than the model's limit of {limit} tokens "
                f"and were truncated, so their endings are ignored by '{self.name}'. Keep sentence "
                "segmentation on (segment='auto') or choose a model with a longer context.",
                SimchologyDataWarning,
            )


@register_method
class SentenceEmbeddingMethod(_EmbeddingCacheMixin, RepresentationMethod):
    """Cosine similarity of Sentence-Transformers embeddings (Sentence-BERT).

    Parameters
    ----------
    model:
        A Sentence-Transformers model id or local folder. Default:
        ``paraphrase-multilingual-mpnet-base-v2``. ``"use_like"`` selects
        ``distiluse-base-multilingual-cased-v2`` (used by the original coding scripts).
    prompt:
        Text prepended to both texts (for instruction-tuned embedding models).
    batch_size:
        ``"auto"`` or a number.
    max_seq_length:
        Override the model's maximum input length (in tokens).
    cache_dir:
        Folder for an on-disk embedding cache, so repeated runs do not re-encode texts.
    """

    info = MethodInfo(
        name="sbert", family="contextual",
        description="cosine similarity of Sentence-Transformers embeddings",
        score_range=(-1.0, 1.0), preprocessing="light", cost_tier="moderate",
        extras=("embeddings",), requires_modules=("torch", "transformers", "sentence_transformers"),
        aliases=("sentence", "sentence_bert", "sentence_transformers", "embedding"),
    )
    defaults = {
        "model": DEFAULT_MODELS["sbert"], "batch_size": "auto", "normalize": True, "prompt": None,
        "max_seq_length": None, "revision": None, "trust_remote_code": False, "cache_dir": None,
    }

    @property
    def model_id(self) -> str:
        m = self.params["model"]
        return DEFAULT_MODELS["use_like"] if m == "use_like" else str(m)

    def _device(self) -> str:
        return resolve_torch_device(self.runtime.device)

    def _batch_size(self) -> int:
        bs = self.params["batch_size"]
        return auto_batch_size(self._device()) if bs == "auto" else int(bs)

    def _model(self):
        model = self.__dict__.get("_st_model")
        if model is not None:
            return model
        import_torch(f"the method '{self.name}'")
        from .._optional import require

        st = require("sentence_transformers", feature=f"the method '{self.name}'", extra="embeddings")
        model = load_model(
            lambda: st.SentenceTransformer(
                self.model_id, device=self._device(), revision=self.params["revision"],
                trust_remote_code=bool(self.params["trust_remote_code"]),
            ),
            self.model_id, "Sentence-Transformers model",
        )
        if self.params["max_seq_length"]:
            model.max_seq_length = int(self.params["max_seq_length"])
        self._st_model = model
        return model

    def _token_lengths(self, model, texts: list[str]) -> list[int]:
        tok = getattr(model, "tokenizer", None)
        if tok is None:
            return []
        prompt = self.params["prompt"] or ""
        try:
            enc = tok([prompt + t for t in texts], add_special_tokens=True, truncation=False)
            return [len(ids) for ids in enc["input_ids"]]
        except Exception:
            return []

    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        model = self._model()
        self._warn_truncation(self._token_lengths(model, texts), getattr(model, "max_seq_length", None))
        kwargs = {"prompt": self.params["prompt"]} if self.params["prompt"] else {}
        return model.encode(
            list(texts), batch_size=self._batch_size(), convert_to_numpy=True,
            normalize_embeddings=bool(self.params["normalize"]), show_progress_bar=False, **kwargs,
        )

    def resource_request(self, n_pairs: int, task: str = "inference"):
        from ..hardware.checks import ResourceRequest

        return ResourceRequest(
            "inference", model_id=self.model_id, seq_len=int(self.params["max_seq_length"] or 256),
            batch_size=self._batch_size(), n_items=n_pairs,
            description=f"embedding texts with {self.model_id!r}",
        )

    def model_ids(self) -> dict[str, str]:
        rev = self.params["revision"]
        return {"embedding_model": self.model_id + (f"@{rev}" if rev else "")}

    def describe(self) -> str:
        return f"cosine similarity of sentence embeddings from {self.model_id}"


@register_method
class LLMEmbeddingMethod(SentenceEmbeddingMethod):
    """Cosine similarity of LLM-based embeddings (default: Qwen3-Embedding-0.6B).

    The same instruction (``prompt``) is prepended to both texts, as recommended for
    symmetric semantic-similarity tasks. These models are larger and slower than
    Sentence-BERT models; a GPU or Apple-silicon Mac is recommended.
    """

    info = MethodInfo(
        name="llm_embedding", family="contextual",
        description="cosine similarity of LLM-based text embeddings",
        score_range=(-1.0, 1.0), preprocessing="light", cost_tier="moderate",
        extras=("embeddings",), requires_modules=("torch", "transformers", "sentence_transformers"),
        aliases=("qwen_embedding", "llm_embeddings"),
    )
    defaults = {**SentenceEmbeddingMethod.defaults, "model": DEFAULT_MODELS["llm_embedding"],
                "prompt": LLM_EMBEDDING_PROMPT, "batch_size": "auto"}

    def _batch_size(self) -> int:
        bs = self.params["batch_size"]
        return auto_batch_size(self._device(), gpu=8, cpu=4) if bs == "auto" else int(bs)

    def resource_request(self, n_pairs: int, task: str = "inference"):
        req = super().resource_request(n_pairs, task)
        req.seq_len = int(self.params["max_seq_length"] or 512)
        return req


@register_method
class HFPoolingEmbeddingMethod(_EmbeddingCacheMixin, RepresentationMethod):
    """Cosine similarity of pooled hidden states of any Hugging Face encoder.

    Plain pretrained encoders (e.g. BERT) were not trained for sentence similarity, and
    simple pooling of their outputs is clearly worse than Sentence-BERT models; this
    method is mainly useful for comparisons or for domain-adapted encoders.

    Parameters
    ----------
    model:
        Hugging Face model id or local folder (default ``jhu-clsp/mmBERT-base``).
    pooling:
        ``"mean"`` over tokens (default), ``"cls"`` (first token) or ``"last_token"``
        (for decoder models).
    layer:
        Which hidden layer to pool (-1 = last).
    """

    info = MethodInfo(
        name="hf_embedding", family="contextual",
        description="cosine similarity of pooled transformer hidden states",
        score_range=(-1.0, 1.0), preprocessing="light", cost_tier="moderate",
        extras=("embeddings",), requires_modules=("torch", "transformers"),
        aliases=("pooled_embedding", "transformer_embedding"),
    )
    defaults = {
        "model": DEFAULT_MODELS["hf_embedding"], "pooling": "mean", "layer": -1, "max_length": 512,
        "batch_size": "auto", "revision": None, "trust_remote_code": False, "cache_dir": None,
    }

    def __init__(self, **params):
        super().__init__(**params)
        if self.params["pooling"] not in ("mean", "cls", "last_token"):
            raise ConfigurationError("pooling must be 'mean', 'cls' or 'last_token'")

    @property
    def model_id(self) -> str:
        return str(self.params["model"])

    def _device(self) -> str:
        return resolve_torch_device(self.runtime.device)

    def _load(self):
        if self.__dict__.get("_hf") is not None:
            return self._hf
        transformers = import_transformers(f"the method '{self.name}'")
        kw = {"revision": self.params["revision"], "trust_remote_code": bool(self.params["trust_remote_code"])}
        tok = load_model(lambda: transformers.AutoTokenizer.from_pretrained(self.model_id, **kw),
                         self.model_id, "tokenizer")
        model = load_model(lambda: transformers.AutoModel.from_pretrained(self.model_id, **kw, **dtype_kwargs(None)),
                           self.model_id, "model")
        model.eval().to(self._device())
        self._hf = (tok, model)
        return self._hf

    def _embed_uncached(self, texts: list[str]) -> np.ndarray:
        torch = import_torch(f"the method '{self.name}'")
        tok, model = self._load()
        limit = min(int(self.params["max_length"]),
                    int(getattr(model.config, "max_position_embeddings", 10**6) or 10**6))
        bs = self.params["batch_size"]
        bs = auto_batch_size(self._device()) if bs == "auto" else int(bs)
        lengths = [len(x) for x in tok(list(texts), add_special_tokens=True, truncation=False)["input_ids"]]
        self._warn_truncation(lengths, limit)
        out = []
        layer = int(self.params["layer"])
        with torch.no_grad():
            for chunk in batched(list(texts), bs):
                enc = tok(list(chunk), padding=True, truncation=True, max_length=limit, return_tensors="pt")
                enc = {k: v.to(model.device) for k, v in enc.items()}
                res = model(**enc, output_hidden_states=layer != -1)
                hidden = res.last_hidden_state if layer == -1 else res.hidden_states[layer]
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooling = self.params["pooling"]
                if pooling == "mean":
                    vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                elif pooling == "cls":
                    vec = hidden[:, 0]
                else:  # last non-padding token, for either padding side
                    am = enc["attention_mask"]
                    idx = am.shape[1] - 1 - torch.argmax(am.flip(1), dim=1)
                    vec = hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]
                out.append(vec.float().cpu().numpy())
        return np.vstack(out)

    def resource_request(self, n_pairs: int, task: str = "inference"):
        from ..hardware.checks import ResourceRequest

        return ResourceRequest("inference", model_id=self.model_id, seq_len=int(self.params["max_length"]),
                               batch_size=8, n_items=n_pairs,
                               description=f"embedding texts with {self.model_id!r}")

    def model_ids(self) -> dict[str, str]:
        return {"encoder": self.model_id}

    def describe(self) -> str:
        return f"cosine similarity of {self.params['pooling']}-pooled hidden states of {self.model_id}"
