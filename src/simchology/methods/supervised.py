"""Supervised pair scoring: cross-encoders fine-tuned on coded pairs (and LoRA-tuned LLMs).

Both texts are fed *jointly* into a transformer encoder, separated by a special token,
and a classification (or regression) head predicts the coders' judgment. Because the
model learns how similarity is operationalised in the coded data, this is often the
most valid approach when enough coded pairs exist (a few hundred or more).

* ``cross_encoder`` - fine-tune an encoder (default: multilingual mmBERT) on the codes.
* ``lora_classifier`` - fine-tune a decoder LLM (default: Qwen3-0.6B) with LoRA adapters.
* ``pretrained_cross_encoder`` - use an already trained cross-encoder (no coded data needed).

Binary labels give a probability of "similar"; gradient labels (ratings) are learnt by
regression and predictions are returned on the label scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .._optional import require
from .._utils.logging import note
from ..config import DEFAULT_MODELS, LABEL, LEFT, RIGHT
from ..data.labels import detect_label_type, scale_labels, unscale_labels
from ..exceptions import ConfigurationError, NotFittedError
from ..finetune.datasets import (
    PairDataset,
    batch_to_features,
    fit_template,
    model_max_positions,
    pair_lengths,
    resolve_max_length,
)
from .base import MethodInfo, PairMethod
from .registry import register_method

PRETRAINED_DEFAULT = DEFAULT_MODELS.get(
    "pretrained_cross_encoder", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)
_META_FILE = "simchology_model.json"


class _TransformerPairMethod(PairMethod):
    """Shared inference, persistence and hardware requests for transformer pair scorers."""

    _cap: int = 512  # upper bound of the automatic max_length
    _pair_encoding: bool = True  # tokenise (response, reference) as a text pair

    def __init__(self, **params):
        super().__init__(**params)
        self.model_ = None
        self.tokenizer_ = None
        self.max_length_: int | None = None
        self.kind_: str | None = None  # "classification" | "regression" | "pretrained"
        self.label_min_: float = 0.0
        self.label_max_: float = 1.0
        self.activation_: str = "softmax"
        self.training_summary_: dict = {}
        self._added_pad_token = False

    # -- configuration -------------------------------------------------------------
    def _model_id(self) -> str:
        return str(self.params.get("backbone") or self.params.get("model"))

    def _device(self) -> str:
        from ..hardware.device import resolve_device

        return resolve_device(self.runtime.device)

    def _expected_kind(self) -> str:
        if self.kind_ is not None:
            return self.kind_
        info = self.runtime.label_info
        return "regression" if info is not None and info.kind == "gradient" else "classification"

    def native_output(self) -> str:
        return "probability" if self._expected_kind() == "classification" else "score"

    def score_range(self):
        if self._expected_kind() == "classification":
            return (0.0, 1.0)
        if self._expected_kind() == "regression":
            return (self.label_min_, self.label_max_)
        return (0.0, 1.0) if self.activation_ in ("sigmoid", "softmax") else (None, None)

    def model_ids(self) -> dict[str, str]:
        rev = self.params.get("revision")
        return {"model": self._model_id() + (f"@{rev}" if rev else "")}

    # -- inputs ----------------------------------------------------------------------
    def _texts(self, left: list[str], right: list[str]):
        """Return (first, second) inputs for the tokenizer; ``second`` is None for templates."""
        return left, right

    def _tokenize(self, left: list[str], right: list[str], *, max_length: int | None, tensors: bool):
        first, second = self._texts(left, right)
        kwargs: dict[str, Any] = {"truncation": max_length is not None, "max_length": max_length}
        if tensors:
            kwargs.update(padding=True, return_tensors="pt")
        if second is None:
            return self.tokenizer_(list(first), **kwargs)
        return self.tokenizer_(list(first), list(second), **kwargs)

    def _lengths(self, left: list[str], right: list[str]) -> np.ndarray:
        first, second = self._texts(left, right)
        return pair_lengths(self.tokenizer_, first, second)

    def _resolve_max_length(self, config, left, right) -> int:
        lengths = self._lengths(left, right) if len(left) else np.array([], dtype=int)
        return resolve_max_length(
            lengths, model_max=model_max_positions(config, self.tokenizer_), cap=self._cap,
            requested=self.params.get("max_length", "auto"),
        )

    # -- inference -------------------------------------------------------------------
    def _score_unique_pairs(self, left: list[str], right: list[str]) -> np.ndarray:
        if self.model_ is None:
            if not self.info.requires_labels:
                self.fit()
            else:
                raise NotFittedError(f"The method '{self.name}' must be fitted on coded pairs first.")
        from ..finetune.trainer import logits_to_scores, predict_logits

        device = self._device()
        self.model_.to(device)
        order = np.argsort([-(len(a) + len(b)) for a, b in zip(left, right)], kind="mergesort")
        bs = int(self.params.get("eval_batch_size", 16))
        batches = (
            self._tokenize([left[i] for i in order[s:s + bs]], [right[i] for i in order[s:s + bs]],
                           max_length=self.max_length_, tensors=True)
            for s in range(0, len(order), bs)
        )
        logits = predict_logits(self.model_, batches, device=device)
        scores = logits_to_scores(logits, self.activation_)
        if self.kind_ == "regression":
            scores = np.clip(scores, 0.0, 1.0)
            scores = unscale_labels(scores, _Scale(self.label_min_, self.label_max_))
        out = np.empty(len(left))
        out[order] = scores
        return out

    # -- hardware --------------------------------------------------------------------
    def resource_request(self, n_pairs: int, task: str = "inference"):
        from ..hardware import ResourceRequest

        seq_len = self.max_length_ or (
            self.params["max_length"] if self.params.get("max_length", "auto") != "auto" else self._cap)
        model_id = self._model_id()
        if task == "inference":
            return ResourceRequest(
                task="inference", model_id=model_id, seq_len=int(seq_len),
                batch_size=int(self.params.get("eval_batch_size", 16)), n_items=int(n_pairs),
                description=f"scoring {n_pairs:,} pairs with the {self._label()} '{model_id}'",
            )
        return ResourceRequest(
            task=self.training_task, model_id=model_id, seq_len=int(seq_len),
            batch_size=int(self.params.get("batch_size", 2)), n_items=int(n_pairs),
            epochs=int(self.params.get("epochs", 1)),
            gradient_checkpointing=bool(self.params.get("gradient_checkpointing", False)),
            lora_fraction=0.02, weight_bytes=self._weight_bytes(),
            description=f"fine-tuning the {self._label()} '{model_id}'",
        )

    def _weight_bytes(self) -> float | None:
        return None

    def _label(self) -> str:
        return "cross-encoder"

    # -- persistence -----------------------------------------------------------------
    def _meta(self) -> dict:
        return {
            "method": self.name, "kind": self.kind_, "max_length": self.max_length_,
            "label_min": self.label_min_, "label_max": self.label_max_,
            "activation": self.activation_, "model": self._model_id(),
            "revision": self.params.get("revision"), "added_pad_token": self._added_pad_token,
            "training": self.training_summary_,
        }

    def save(self, directory: Path) -> None:
        if self.model_ is None:
            return
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._save_model(directory)
        (directory / _META_FILE).write_text(json.dumps(self._meta(), indent=2), encoding="utf-8")

    def _save_model(self, directory: Path) -> None:
        from ..finetune.trainer import quiet_transformers

        with quiet_transformers():
            self.model_.save_pretrained(directory / "model")
            self.tokenizer_.save_pretrained(directory / "model")

    def load(self, directory: Path, *, trust: bool = False) -> None:
        directory = Path(directory)
        meta_file = directory / _META_FILE
        if not meta_file.exists():
            self._fitted = not self.info.requires_labels
            return
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        self.kind_ = meta["kind"]
        self.max_length_ = meta["max_length"]
        self.label_min_, self.label_max_ = meta["label_min"], meta["label_max"]
        self.activation_ = meta["activation"]
        self._added_pad_token = meta.get("added_pad_token", False)
        self.training_summary_ = meta.get("training", {})
        self._load_model(directory, meta)
        self._fitted = True

    def _load_model(self, directory: Path, meta: dict) -> None:
        from ..finetune.trainer import load_sequence_classifier, load_tokenizer

        source = str(directory / "model")
        self.tokenizer_ = load_tokenizer(source)
        self.model_ = load_sequence_classifier(
            source, trust_remote_code=self.params.get("trust_remote_code", False))
        self.model_.to(self._device())


class _Scale:
    """Minimal stand-in for LabelInfo in (un)scaling helpers."""

    def __init__(self, lo: float, hi: float) -> None:
        self.min, self.max = lo, hi


@register_method
class CrossEncoderMethod(_TransformerPairMethod):
    """Fine-tune a transformer cross-encoder on coded pairs.

    Parameters
    ----------
    backbone:
        Hugging Face model id or local folder (default ``jhu-clsp/mmBERT-base``, a
        multilingual ModernBERT). Any model usable with
        ``AutoModelForSequenceClassification`` works.
    max_length:
        Maximum tokens per pair; ``"auto"`` = the 95th percentile of the training pairs'
        lengths (at most 512 and the model's limit).
    epochs, learning_rate, batch_size, grad_accum, weight_decay:
        Training settings (defaults as in the original fine-tuning notebook).
    early_stopping_patience:
        Stop after this many epochs without improvement on the development pairs.
    metric_for_best:
        ``"auto"`` (F1 for binary labels, Spearman correlation for gradient labels).
    class_weight:
        ``None`` or ``"balanced"`` (weights classes inversely to their frequency).
    gradient_checkpointing:
        Trade compute for memory (useful on small GPUs).
    output_dir:
        Keep training checkpoints here; by default a temporary folder that is removed.
    bf16:
        ``"auto"`` (use bf16 mixed precision where supported), True or False.
    """

    info = MethodInfo(
        name="cross_encoder", family="supervised",
        description="a cross-encoder fine-tuned on the coded pairs",
        symmetric=False, requires_labels=True, native_output="probability", granularity="pair",
        segment_default=False, preprocessing="light", cost_tier="heavy", extras=("finetune",),
        requires_modules=("torch", "transformers", "accelerate"),
        aliases=("crossencoder", "finetune", "fine_tuning", "bert_finetune"), deterministic=False,
    )
    defaults = {
        "backbone": DEFAULT_MODELS["cross_encoder"], "max_length": "auto", "epochs": 10,
        "learning_rate": 2e-5, "batch_size": 2, "grad_accum": 8, "eval_batch_size": 16,
        "weight_decay": 0.01, "early_stopping_patience": 3, "metric_for_best": "auto",
        "gradient_checkpointing": False, "class_weight": None, "output_dir": None,
        "revision": None, "trust_remote_code": False, "bf16": "auto",
    }
    training_task = "full_finetune"

    def __init__(self, **params):
        super().__init__(**params)
        if self.params["class_weight"] not in (None, "balanced"):
            raise ConfigurationError("class_weight must be None or 'balanced'")
        if self.params["bf16"] not in ("auto", True, False):
            raise ConfigurationError("bf16 must be 'auto', True or False")

    # model construction differs between encoder cross-encoders and LoRA decoders
    def _build(self, config, num_labels: int, regression: bool):
        from ..finetune.trainer import load_sequence_classifier

        return load_sequence_classifier(
            self._model_id(), num_labels=num_labels, regression=regression,
            revision=self.params["revision"], trust_remote_code=self.params["trust_remote_code"],
        )

    def _load_tokenizer(self) -> None:
        from ..finetune.trainer import load_tokenizer

        self.tokenizer_ = load_tokenizer(self._model_id(), revision=self.params["revision"],
                                         trust_remote_code=self.params["trust_remote_code"])

    def fit(self, corpus=None, train=None, dev=None):
        if train is None or not len(train):
            raise ConfigurationError(f"The method '{self.name}' needs coded training pairs.")
        transformers = require("transformers", feature=f"the method '{self.name}'", extra="finetune")
        require("torch", feature=f"the method '{self.name}'", extra="finetune")
        require("accelerate", feature=f"the method '{self.name}'", extra="finetune")
        from ..finetune.trainer import check_sequence_classification, release_memory, train_model

        info = self.runtime.label_info or detect_label_type(train[LABEL])
        regression = info.kind == "gradient"
        self.kind_ = "regression" if regression else "classification"
        self.label_min_, self.label_max_ = float(info.min), float(info.max)
        self.activation_ = "identity" if regression else "softmax"
        y_train = self._targets(train[LABEL].to_numpy(dtype=float), regression)
        if not regression and len(np.unique(y_train)) < 2:
            raise ConfigurationError("The coded training pairs contain only one class; "
                                     "a classifier needs examples of similar and dissimilar pairs.")

        config = check_sequence_classification(self._model_id(), revision=self.params["revision"],
                                               trust_remote_code=self.params["trust_remote_code"])
        transformers.set_seed(self.runtime.seed)
        self._load_tokenizer()
        left, right = train[LEFT].astype(str).tolist(), train[RIGHT].astype(str).tolist()
        self.max_length_ = self._resolve_max_length(config, left, right)
        # checked now that the real sequence length is known, before the weights are loaded
        self.check_hardware(self.training_task, len(train))
        model = self._build(config, 1 if regression else 2, regression)

        train_ds = PairDataset(self._features(left, right), y_train.tolist())
        dev_ds = None
        if dev is not None and len(dev):
            dev_ds = PairDataset(
                self._features(dev[LEFT].astype(str).tolist(), dev[RIGHT].astype(str).tolist()),
                self._targets(dev[LABEL].to_numpy(dtype=float), regression).tolist(),
            )
        weights = None
        if self.params.get("class_weight") == "balanced" and not regression:
            counts = np.bincount(y_train.astype(int), minlength=2).astype(float)
            weights = (len(y_train) / (2 * np.maximum(counts, 1))).tolist()
        device = self._device()
        note(f"Fine-tuning '{self._model_id()}' on {len(train):,} coded pairs "
             f"({'regression' if regression else 'classification'}, max_length={self.max_length_}, "
             f"device={device}){'' if dev_ds else '; no development pairs, so no early stopping'} ...",
             self.runtime.verbose)
        self.model_, self.training_summary_ = train_model(
            model, self.tokenizer_, train_ds, dev_ds, kind=self.kind_, params=self.params,
            device=device, seed=self.runtime.seed, verbose=self.runtime.verbose,
            class_weights=weights,
            gradient_checkpointing=bool(self.params.get("gradient_checkpointing", False)),
        )
        self.model_.to(device)
        release_memory(device)
        self._fitted = True
        return self

    def _targets(self, labels: np.ndarray, regression: bool) -> np.ndarray:
        if regression:
            return scale_labels(labels, _Scale(self.label_min_, self.label_max_)).astype(np.float32)
        return labels.astype(int)

    def _features(self, left: list[str], right: list[str]) -> list[dict]:
        return batch_to_features(self._tokenize(left, right, max_length=self.max_length_, tensors=False))

    def describe(self) -> str:
        what = "regression" if self.kind_ == "regression" else "classification"
        summary = self.training_summary_
        stop = (f"; the best of {summary.get('epochs_trained', 0):.0f} epochs by development "
                f"{summary.get('metric')} was kept" if summary.get("metric") else "")
        return (f"a cross-encoder ({self._model_id()}) fine-tuned for {what} on the coded "
                f"training pairs (max. {self.params['epochs']} epochs, learning rate "
                f"{self.params['learning_rate']:g}, max. {self.max_length_} tokens per pair{stop})")


@register_method
class LoRAClassifierMethod(CrossEncoderMethod):
    """Fine-tune a decoder LLM as a pair classifier with LoRA adapters (PEFT).

    The pair is written into a prompt (``template``) whose last token is read by the
    classification head; only small adapter matrices and the head are trained. Long
    responses are shortened so the end of the prompt is never cut off.

    Parameters (in addition to those of ``cross_encoder``): ``r``, ``alpha``,
    ``dropout`` and ``target_modules`` of the LoRA adapters, ``template`` (with
    ``{response}`` and ``{reference}``) and ``load_in_4bit`` (CUDA + bitsandbytes only).

    Vision-language backbones such as Qwen3.5 work too: with the default
    ``target_modules="all-linear"`` their vision encoder gets no adapters, since text-only
    training never updates it.
    """

    info = MethodInfo(
        name="lora_classifier", family="supervised",
        description="a decoder language model fine-tuned with LoRA adapters on the coded pairs",
        symmetric=False, requires_labels=True, native_output="probability", granularity="pair",
        segment_default=False, preprocessing="light", cost_tier="heavy", extras=("finetune",),
        requires_modules=("torch", "transformers", "accelerate", "peft"),
        aliases=("lora", "llm_finetune"), deterministic=False,
    )
    defaults = {
        **CrossEncoderMethod.defaults,
        "backbone": DEFAULT_MODELS["lora_classifier"], "learning_rate": 1e-4, "epochs": 3,
        "batch_size": 4, "grad_accum": 4, "eval_batch_size": 8, "gradient_checkpointing": True,
        "r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear",
        "template": "Text: {response}\nInformation: {reference}\nAnswer:", "load_in_4bit": False,
    }
    training_task = "lora"
    _cap = 1024
    _pair_encoding = False

    def __init__(self, **params):
        super().__init__(**params)
        template = self.params["template"]
        if "{response}" not in template or "{reference}" not in template:
            raise ConfigurationError("template must contain {response} and {reference}")

    def _label(self) -> str:
        return "LoRA classifier"

    def _weight_bytes(self) -> float | None:
        return 4.0 if self._device() == "cpu" else 2.0

    def _texts(self, left, right):
        limit = self.max_length_
        template = self.params["template"]
        if limit is None:
            return [template.format(response=a, reference=b) for a, b in zip(left, right)], None
        return [fit_template(self.tokenizer_, template, a, b, limit) for a, b in zip(left, right)], None

    def _load_tokenizer(self) -> None:
        pass  # the tokenizer is created together with the model in _build

    def _resolve_max_length(self, config, left, right) -> int:
        from ..finetune.lora import prepare_tokenizer
        from ..finetune.trainer import load_tokenizer

        if self.tokenizer_ is None:
            self.tokenizer_ = load_tokenizer(self._model_id(), revision=self.params["revision"],
                                             trust_remote_code=self.params["trust_remote_code"])
            prepare_tokenizer(self.tokenizer_)
        self.max_length_ = None
        return super()._resolve_max_length(config, left, right)

    def _build(self, config, num_labels: int, regression: bool):
        from ..finetune.lora import build_lora_model

        model, tokenizer, added = build_lora_model(
            self._model_id(), num_labels=num_labels, regression=regression, params=self.params,
            device=self._device(),
        )
        self.tokenizer_ = tokenizer
        self._added_pad_token = added
        return model

    def _save_model(self, directory: Path) -> None:
        if hasattr(self.model_, "peft_config"):
            from ..finetune.trainer import quiet_transformers

            with quiet_transformers():
                self.model_.save_pretrained(directory / "adapter")
                self.tokenizer_.save_pretrained(directory / "adapter")
        else:  # already merged (e.g. after loading): save the full model
            super()._save_model(directory)

    def _meta(self) -> dict:
        meta = super()._meta()
        meta["format"] = "adapter" if hasattr(self.model_, "peft_config") else "full"
        meta["template"] = self.params["template"]
        return meta

    def _load_model(self, directory: Path, meta: dict) -> None:
        if meta.get("format", "adapter") == "full":
            super()._load_model(directory, meta)
            return
        from ..finetune.lora import load_lora_model

        regression = meta["kind"] == "regression"
        self.model_, self.tokenizer_ = load_lora_model(
            meta["model"], directory / "adapter", num_labels=1 if regression else 2,
            regression=regression, params=self.params, added_pad_token=meta.get("added_pad_token", False),
            device=self._device(),
        )
        self.model_.to(self._device())

    def describe(self) -> str:
        what = "regression" if self.kind_ == "regression" else "classification"
        return (f"the language model {self._model_id()} fine-tuned for {what} on the coded "
                f"training pairs with LoRA adapters (r={self.params['r']}, alpha={self.params['alpha']}, "
                f"max. {self.params['epochs']} epochs, max. {self.max_length_} tokens per prompt)")


@register_method
class PretrainedCrossEncoderMethod(_TransformerPairMethod):
    """Score pairs with an already trained cross-encoder, without coded data.

    The default, ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``, is a small multilingual
    relevance model (Apache-2.0) trained to judge whether a passage answers a query; with
    ``pair_order="reference_first"`` the reference is the query and the response the
    passage, so the score estimates how well the response covers the reference. STS
    cross-encoders (e.g. ``cross-encoder/stsb-roberta-base``, English) can be used for
    symmetric semantic similarity; stronger multilingual rerankers such as
    ``BAAI/bge-reranker-v2-m3`` also work but are larger.

    Parameters
    ----------
    model:
        Hugging Face id or local folder of a cross-encoder with a trained head.
    activation:
        ``"auto"`` (sigmoid for one output, probability of the last class otherwise),
        ``"sigmoid"``, ``"softmax"`` or ``"identity"`` (raw logit).
    pair_order:
        ``"reference_first"`` (query = reference) or ``"response_first"``.
    """

    info = MethodInfo(
        name="pretrained_cross_encoder", family="supervised",
        description="a pre-trained cross-encoder (no fine-tuning)",
        symmetric=False, requires_labels=False, native_output="score", granularity="pair",
        segment_default=False, preprocessing="light", cost_tier="moderate", extras=("embeddings",),
        requires_modules=("torch", "transformers"), aliases=("sts_cross_encoder", "reranker"),
    )
    defaults = {"model": PRETRAINED_DEFAULT, "max_length": 512, "eval_batch_size": 16,
                "activation": "auto", "pair_order": "reference_first", "revision": None,
                "trust_remote_code": False}

    def __init__(self, **params):
        super().__init__(**params)
        if self.params["activation"] not in ("auto", "sigmoid", "softmax", "identity"):
            raise ConfigurationError("activation must be auto, sigmoid, softmax or identity")
        if self.params["pair_order"] not in ("reference_first", "response_first"):
            raise ConfigurationError("pair_order must be 'reference_first' or 'response_first'")
        self.kind_ = "pretrained"

    def native_output(self) -> str:
        return "score"

    def _texts(self, left, right):
        return (right, left) if self.params["pair_order"] == "reference_first" else (left, right)

    def fit(self, corpus=None, train=None, dev=None):
        require("torch", feature=f"the method '{self.name}'", extra="embeddings")
        from ..finetune.trainer import (
            check_sequence_classification,
            load_sequence_classifier,
            load_tokenizer,
        )

        if self.model_ is None:
            config = check_sequence_classification(self._model_id(), revision=self.params["revision"],
                                                   trust_remote_code=self.params["trust_remote_code"])
            self.tokenizer_ = load_tokenizer(self._model_id(), revision=self.params["revision"],
                                             trust_remote_code=self.params["trust_remote_code"])
            self.model_ = load_sequence_classifier(
                self._model_id(), revision=self.params["revision"],
                trust_remote_code=self.params["trust_remote_code"], require_head=True,
            )
            num_labels = int(getattr(self.model_.config, "num_labels", 1))
            act = self.params["activation"]
            self.activation_ = ("sigmoid" if num_labels == 1 else "softmax") if act == "auto" else act
            self.max_length_ = resolve_max_length(
                np.array([], dtype=int), model_max=model_max_positions(config, self.tokenizer_),
                cap=int(self.params["max_length"]), requested=int(self.params["max_length"]),
            )
            self.model_.to(self._device())
        self._fitted = True
        return self

    def _save_model(self, directory: Path) -> None:
        pass  # nothing was trained: reload the original model on load

    def _load_model(self, directory: Path, meta: dict) -> None:
        self.model_ = None
        self.fit()

    def describe(self) -> str:
        return f"the pre-trained cross-encoder {self._model_id()} (not fine-tuned)"
