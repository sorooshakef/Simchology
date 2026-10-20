"""Prompt-based inference: an LLM judges each (response, reference) pair.

The ``llm`` method asks a local model served by Ollama (https://ollama.com), so texts
never leave the computer (an Ollama server on another machine needs ``allow_remote=True``).

The model is asked a fixed question (yes/no or a rating) after a task description that
users can replace with their own coding manual (``instructions=...``). With ``mode="few_shot"``,
``n_examples`` coded training pairs are shown as worked examples; they never come from
the held-out test pairs. Few-shot prompting is not guaranteed to beat zero-shot
prompting, so compare both on your coded data.

Outputs: for binary output the answer is yes/no (as P(yes) when the server returns
token probabilities); for gradient output P(yes) is used if token probabilities are
available, otherwise a rating on ``scale`` mapped to [0, 1].
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._utils.logging import progress
from .._utils.warn import warn
from ..config import DEFAULT_MODELS, LABEL, LEFT, OLLAMA_HOST, RIGHT
from ..exceptions import ConfigurationError, SimchologyWarning
from ..llm.cache import ResponseCache
from ..llm.prompts import PromptBuilder, validate_scale, validate_task
from ..llm.schema import (
    answer_json,
    binary_schema,
    expected_rating_from_logprobs,
    parse_binary,
    parse_rating,
    prob_yes_from_logprobs,
    rating_schema,
    rating_to_unit,
)
from .base import MethodInfo, PairMethod
from .registry import register_method

_RESPONSES = ("auto", "yes_no", "probability", "rating")

_SHARED_DEFAULTS = {
    "mode": "zero_shot",
    "n_examples": 6,
    "task": "containment",
    "instructions": None,
    "language": None,
    "scale": (1, 5),
    "response": "auto",
    "temperature": 0.0,
    "seed": None,
    "max_retries": 2,
    "cache_dir": None,
    "client": None,
}


class _PromptJudge(PairMethod):
    """Shared logic of the LLM judges (prompting, few-shot examples, parsing, caching)."""

    defaults = dict(_SHARED_DEFAULTS)
    _backend_name = "llm"

    @classmethod
    def missing_modules(cls, params: dict | None = None) -> list[str]:
        # A client object passed in by the caller replaces the client library.
        if params and params.get("client") is not None:
            return []
        return super().missing_modules(params)
    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        p = self.params
        if p["mode"] not in ("zero_shot", "few_shot"):
            raise ConfigurationError("mode must be 'zero_shot' or 'few_shot'")
        if p["response"] not in _RESPONSES:
            raise ConfigurationError(f"response must be one of {_RESPONSES}")
        validate_task(p["task"])
        self.scale = validate_scale(p["scale"])
        self._examples: pd.DataFrame | None = None
        self._kind: str | None = None  # resolved: "binary" | "probability" | "rating"
        self._ready = False
        self._cache: ResponseCache | None = None
        self._last_logprobs: Any = None
        self._n_failed = 0
        self._fitted = True

    # -- configuration resolved at run time ------------------------------------------
    @property
    def language(self) -> str:
        return self.params["language"] or self.runtime.language or "en"

    @property
    def seed(self) -> int:
        return self.runtime.seed if self.params["seed"] is None else int(self.params["seed"])

    def builder(self) -> PromptBuilder:
        return PromptBuilder(self.language, self.params["task"], self.params["instructions"], self.scale)

    def _requested_kind(self) -> str:
        r = self.params["response"]
        if r == "yes_no":
            return "binary"
        if r in ("probability", "rating"):
            return r
        return "binary" if self.runtime.output == "binary" else "auto"

    def native_output(self) -> str:
        kind = self._kind or self._requested_kind()
        return "score" if kind == "rating" else "probability"

    def score_range(self):
        return (0.0, 1.0)

    # -- few-shot examples -------------------------------------------------------------
    def fit(self, corpus=None, train=None, dev=None):
        if self.params["mode"] == "few_shot":
            if train is None or not len(train):
                raise ConfigurationError(
                    "mode='few_shot' needs coded pairs to draw examples from: pass coded=... "
                    "(examples are taken from the training pairs only)."
                )
            self._examples = train[[LEFT, RIGHT, LABEL]].dropna().reset_index(drop=True)
        self._fitted = True
        return self

    def _is_positive(self, label: float) -> bool:
        info = self.runtime.label_info
        if info is not None and info.kind == "gradient":
            span = (info.max - info.min) or 1.0
            return (label - info.min) / span >= 0.5
        return label >= 0.5

    def _example_answer(self, label: float, kind: str) -> str:
        if kind == "rating":
            lo, hi = self.scale
            info = self.runtime.label_info
            if info is not None and info.kind == "gradient":
                span = (info.max - info.min) or 1.0
                value = lo + (label - info.min) / span * (hi - lo)
            else:
                value = hi if label >= 0.5 else lo
            return answer_json("rating", round(value))
        return answer_json("binary", self._is_positive(label))

    def _ordered_examples(self) -> tuple[list[int], list[int]]:
        if not hasattr(self, "_example_order"):
            ex = self._examples
            pos = [i for i, v in enumerate(ex[LABEL]) if self._is_positive(float(v))]
            neg = [i for i, v in enumerate(ex[LABEL]) if not self._is_positive(float(v))]
            rng = np.random.default_rng(self.seed)
            rng.shuffle(pos)
            rng.shuffle(neg)
            self._example_order = (pos, neg)
        return self._example_order

    def select_examples(self, query_response: str, kind: str) -> list[tuple[str, str, str]]:
        """Balanced examples (alternating similar / not similar) never sharing the query's response."""
        if self._examples is None or self.params["mode"] != "few_shot":
            return []
        n = int(self.params["n_examples"])
        pos, neg = self._ordered_examples()
        ex = self._examples
        ok = [i for i in pos if ex[LEFT].iat[i] != query_response]
        bad = [i for i in neg if ex[LEFT].iat[i] != query_response]
        n_pos = min(len(ok), (n + 1) // 2)
        n_neg = min(len(bad), n - n_pos)
        n_pos = min(len(ok), n - n_neg)
        chosen: list[int] = []
        for k in range(max(n_pos, n_neg)):
            if k < n_pos:
                chosen.append(ok[k])
            if k < n_neg:
                chosen.append(bad[k])
        return [(ex[LEFT].iat[i], ex[RIGHT].iat[i], self._example_answer(float(ex[LABEL].iat[i]), kind))
                for i in chosen]

    # -- the backend-specific part ---------------------------------------------------------
    def _ensure_ready(self, left: list[str], right: list[str]) -> None:
        raise NotImplementedError

    def _complete(self, messages: list[dict], kind: str) -> tuple[str, Any]:
        raise NotImplementedError

    def _prepare_batch(self, left: list[str], right: list[str]) -> None:
        """Called before every batch (e.g. to size the context window)."""

    def _cache_identity(self) -> dict:
        return {}

    # -- scoring ------------------------------------------------------------------------
    def _value(self, content: str, logprobs: Any, kind: str) -> float | None:
        lo, hi = self.scale
        if kind == "rating":
            expected = expected_rating_from_logprobs(logprobs, lo, hi) if logprobs else None
            if expected is not None:
                return rating_to_unit(expected, lo, hi)
            r = parse_rating(content, lo, hi)
            return None if r is None else rating_to_unit(r, lo, hi)
        p = prob_yes_from_logprobs(logprobs) if logprobs else None
        answer = parse_binary(content)
        if kind == "probability" and p is not None:
            return p
        if kind == "binary":
            if p is not None and answer is not None:
                return p
            return answer
        return answer

    def _judge(self, left: str, right: str, kind: str) -> float | None:
        b = self.builder()
        query_kind = "rating" if kind == "rating" else "binary"
        examples = self.select_examples(left, query_kind)
        for attempt in range(int(self.params["max_retries"]) + 1):
            messages = b.messages(left, right, query_kind, examples, reminder=attempt > 0)
            key = ResponseCache.key(backend=self._backend_name, messages=messages, kind=query_kind,
                                    **self._cache_identity())
            cached = self._cache.get(key)
            if cached is not None:
                content, logprobs = cached["content"], cached.get("logprobs")
            else:
                content, logprobs = self._complete(messages, query_kind)
                self._cache.put(key, {"content": content, "logprobs": _plain(logprobs)})
                logprobs = _plain(logprobs)
            self._last_logprobs = logprobs
            value = self._value(content, logprobs, kind)
            if value is not None:
                return value
        return None

    def _resolve_kind(self, left: str, right: str) -> str:
        kind = self._requested_kind()
        if kind != "auto":
            return kind
        # gradient output: P(yes) if the server returns token probabilities, else a rating
        self._last_logprobs = None
        self._judge(left, right, "binary")
        has_p = prob_yes_from_logprobs(self._last_logprobs) is not None
        return "probability" if has_p else "rating"

    def _score_unique_pairs(self, left: list[str], right: list[str]) -> np.ndarray:
        if self._cache is None:
            self._cache = ResponseCache(self.params["cache_dir"], name=f"{self._backend_name}_judgments")
        if not self._ready:
            self._ensure_ready(left, right)
            self._ready = True
        self._prepare_batch(left, right)
        if self._kind is None and len(left):
            self._kind = self._resolve_kind(left[0], right[0])
        failed = 0
        out = np.full(len(left), np.nan)
        items = progress(range(len(left)), total=len(left), desc=f"asking {self.name}",
                         enabled=self.runtime.verbose and len(left) > 20)
        for i in items:
            value = self._judge(left[i], right[i], self._kind)
            if value is None:
                failed += 1
            else:
                out[i] = value
        if failed:
            self._n_failed += failed
            warn(
                f"{failed} of {len(left)} judgments of '{self.name}' could not be read from the "
                "model's answer and are missing (NaN). Try a larger model or response='rating'.",
                SimchologyWarning,
            )
        return out


def _plain(logprobs: Any) -> Any:
    """Convert token log-probabilities to JSON-serialisable dicts (for the cache)."""
    if logprobs is None:
        return None
    out = []
    for pos in logprobs:
        def get(o, k):
            if isinstance(o, dict):
                return o.get(k)
            try:
                return o[k]
            except (TypeError, KeyError, IndexError):
                return getattr(o, k, None)

        tops = [{"token": get(t, "token"), "logprob": get(t, "logprob")}
                for t in (get(pos, "top_logprobs") or [])]
        out.append({"token": get(pos, "token"), "logprob": get(pos, "logprob"), "top_logprobs": tops})
    return out


@register_method
class OllamaJudge(_PromptJudge):
    """Ask a local LLM served by Ollama whether/how similar each pair is.

    Parameters
    ----------
    model:
        Ollama model name (default: a small multilingual instruction model). Download it
        once with ``ollama pull <model>`` or pass ``auto_pull=True``.
    host:
        Ollama server address (default ``http://localhost:11434``). Non-local hosts need
        ``allow_remote=True``.
    mode:
        ``"zero_shot"`` or ``"few_shot"`` (``n_examples`` coded training pairs as examples).
    task:
        ``"containment"`` (does the response contain the reference's information? -
        default, suits recall/coding studies) or ``"similarity"`` (similar meaning?).
    instructions:
        Your own construct definition / coding manual, replacing the default task text.
    scale:
        Rating scale for gradient output without token probabilities (default (1, 5)).
    response:
        ``"auto"``, ``"yes_no"``, ``"probability"`` (P(yes) from token probabilities) or ``"rating"``.
    num_ctx:
        Context window; ``"auto"`` sizes it to the longest prompt (Ollama otherwise
        silently truncates long prompts).
    cache_dir:
        Directory for a resumable cache of all answers.
    """

    info = MethodInfo(
        name="llm", family="prompt", description="judgments of a local LLM (Ollama)",
        symmetric=False, native_output="probability", score_range=(0.0, 1.0), granularity="pair",
        segment_default=False, preprocessing="light", cost_tier="heavy", extras=("llm",),
        requires_modules=("ollama",), deterministic=False, uses_training_pairs=True,
        aliases=("ollama", "prompt", "llm_judge"),
    )
    defaults = {
        **_SHARED_DEFAULTS,
        "model": DEFAULT_MODELS["ollama"],
        "host": OLLAMA_HOST,
        "num_ctx": "auto",
        "think": False,
        "keep_alive": "10m",
        "timeout": 120,
        "auto_pull": False,
    }
    _backend_name = "ollama"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._backend = None
        self._model_info: dict = {}
        self._num_ctx: int | None = None

    @property
    def backend(self):
        if self._backend is None:
            from ..llm.ollama_backend import OllamaBackend

            self._backend = OllamaBackend(
                self.params["host"], timeout=self.params["timeout"], client=self.params["client"],
                allow_remote=self.runtime.allow_remote, verbose=self.runtime.verbose,
            )
        return self._backend

    def _needed_ctx(self, left: list[str], right: list[str]) -> int:
        b = self.builder()
        kind = "rating" if self._requested_kind() == "rating" else "binary"
        longest = max(len(a) for a in left) if left else 0
        ref = max(len(r) for r in right) if right else 0
        examples = self.select_examples("", kind)
        chars = len(b.user_message("x" * longest, "x" * ref, kind)) + len(b.t["system"])
        chars += sum(len(b.user_message(e[0], e[1], kind)) + len(e[2]) for e in examples)
        tokens = chars / 3 + 256  # ~3 characters per token for German/English text, plus answer
        return int(np.ceil(tokens / 1024) * 1024)

    def _ensure_ready(self, left, right):
        self._model_info = self.backend.ensure_model(self.params["model"], auto_pull=self.params["auto_pull"])
        self._hardware_checked = False

    def _prepare_batch(self, left, right):
        from ..hardware.checks import check_ollama_model

        needed = self._needed_ctx(left, right)
        context = self._model_info.get("context_length")
        if self.params["num_ctx"] == "auto":
            num_ctx = max(2048, needed, self._num_ctx or 0)
            if context:
                if needed > context:
                    warn(
                        f"The longest prompt (~{needed} tokens) exceeds the context length of "
                        f"{self.params['model']!r} ({context} tokens); it will be truncated.",
                        SimchologyWarning,
                    )
                num_ctx = min(num_ctx, context)
        else:
            num_ctx = int(self.params["num_ctx"])
        self._num_ctx = num_ctx
        if not getattr(self, "_hardware_checked", False) and self.runtime.check_hardware != "off":
            self._hardware_checked = True
            check_ollama_model(
                self.params["model"], self._model_info.get("size"), num_ctx=num_ctx,
                n_layers=self._model_info.get("n_layers"),
                hidden_size=self._model_info.get("hidden_size"), policy=self.runtime.check_hardware,
            )

    def _cache_identity(self) -> dict:
        return {"model": self.params["model"], "digest": self._model_info.get("digest"),
                "options": {"temperature": self.params["temperature"], "seed": self.seed}}

    def _complete(self, messages, kind):
        lo, hi = self.scale
        schema = rating_schema(lo, hi) if kind == "rating" else binary_schema()
        options = {"temperature": float(self.params["temperature"]), "seed": self.seed,
                   "num_ctx": self._num_ctx, "num_predict": 64}
        return self.backend.chat(
            self.params["model"], messages, format=schema, options=options,
            think=self.params["think"], keep_alive=self.params["keep_alive"], logprobs=True,
        )

    def model_ids(self) -> dict[str, str]:
        digest = self._model_info.get("digest")
        model = self.params["model"]
        return {"llm": f"{model} (ollama, digest {digest[:12]})" if digest else f"{model} (ollama)"}

    def describe(self) -> str:
        mode = "few-shot" if self.params["mode"] == "few_shot" else "zero-shot"
        return f"{mode} judgments of the local LLM {self.params['model']} (Ollama, temperature {self.params['temperature']:g})"
