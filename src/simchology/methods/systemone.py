"""Zero-shot typed-decision classifiers via the System One protocol.

System One models (TypeSafe's hosted Jev, or open models such as Kev that run on your
own computer) take a *state* (here: the response text) plus typed questions and return
probabilities instead of generated text:

* ``noul`` questions (yes/no) give P(yes) - used for binary output;
* ``score`` questions place the text on ordered levels (e.g. not / partially / fully
  contained) - used for gradient output, normalised to [0, 1].

By default Simchology talks to a local server at ``http://localhost:8009``. To run Kev
locally (https://github.com/jaredpalmer/kev)::

    uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009

Hosted Jev needs ``base_url``, an API key and ``allow_remote=True``, because the texts
then leave your computer.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from .._optional import require
from .._utils.logging import progress
from ..config import SYSTEMONE_URL
from ..exceptions import (
    BackendUnavailableError,
    ConfigurationError,
    ModelNotAvailableError,
)
from ..llm.cache import ResponseCache
from ..llm.netpolicy import enforce_remote_policy
from ..llm.prompts import default_levels, systemone_question, validate_task
from ..text.segment import regex_sentences
from .base import MethodInfo, PairMethod
from .registry import register_method

_KEY_ENVS = ("SIMCHOLOGY_SYSTEMONE_API_KEY", "TYPESAFE_API_KEY", "KEV_API_KEY")
_URL_ENVS = ("SIMCHOLOGY_SYSTEMONE_URL", "TYPESAFE_BASE_URL")


class _StateTooLong(Exception):
    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class _SegmentTooLong(ConfigurationError):
    """A text that cannot be split further was rejected as too long."""


@register_method
class SystemOneJudge(PairMethod):
    """Ask a System One server (Kev locally, or hosted Jev) about each pair.

    Parameters
    ----------
    base_url:
        Server address without the endpoint path (default ``http://localhost:8009``; the
        environment variables ``SIMCHOLOGY_SYSTEMONE_URL`` or ``TYPESAFE_BASE_URL``
        override the default). Ollama also serves System One models at
        ``http://localhost:11434``.
    model:
        Model name on the server (default ``"kev-latest"``).
    api_key:
        Bearer token, if the server requires one (default: environment variables
        ``SIMCHOLOGY_SYSTEMONE_API_KEY``, ``TYPESAFE_API_KEY`` or ``KEV_API_KEY``).
    question:
        ``"auto"`` (``"noul"`` = yes/no for binary output, ``"score"`` = ordered levels
        for gradient output), ``"noul"`` or ``"score"``.
    task:
        ``"containment"`` (default) or ``"similarity"``.
    instructions:
        Your own question text; ``{reference}`` is replaced by the reference (otherwise
        the reference is appended).
    criteria:
        For ``noul`` questions: ``{"true": "...", "false": "..."}`` descriptions.
    levels:
        For ``score`` questions: level descriptions from lowest to highest.
    group_by_response:
        Ask all questions about one response in a single request (much faster).
    endpoint:
        ``"/v1/systemone"`` (questions share one forward pass) or
        ``"/v1/systemone/separate"`` (Kev: each question answered in isolation).
    """

    info = MethodInfo(
        name="systemone", family="prompt",
        description="zero-shot typed-decision classifier (System One protocol, e.g. Kev or Jev)",
        symmetric=False, native_output="probability", score_range=(0.0, 1.0), granularity="pair",
        segment_default=False, preprocessing="light", cost_tier="heavy", extras=("systemone",),
        requires_modules=("httpx",), deterministic=False,
        aliases=("system_one", "jev", "kev", "zero_shot_classifier"),
    )
    defaults = {
        "base_url": SYSTEMONE_URL,
        "model": "kev-latest",
        "api_key": None,
        "question": "auto",
        "task": "containment",
        "language": None,
        "instructions": None,
        "criteria": None,
        "levels": None,
        "group_by_response": True,
        "max_questions": 32,
        "endpoint": "/v1/systemone",
        "timeout": 60,
        "max_retries": 2,
        "cache_dir": None,
        "client": None,
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        p = self.params
        validate_task(p["task"])
        if p["question"] not in ("auto", "noul", "score"):
            raise ConfigurationError("question must be 'auto', 'noul' or 'score'")
        if p["levels"] is not None and len(list(p["levels"])) < 2:
            raise ConfigurationError("levels must list at least two level descriptions, lowest first.")
        self._client = None
        self._cache: ResponseCache | None = None
        self._ready = False
        self._card: dict | None = None  # the server's model card (from /v1/models)
        self._fitted = True

    # -- configuration -------------------------------------------------------------------
    @property
    def base_url(self) -> str:
        url = self.params["base_url"]
        if url == SYSTEMONE_URL:
            for env in _URL_ENVS:
                if os.environ.get(env):
                    url = os.environ[env]
                    break
        return str(url).rstrip("/")

    @property
    def api_key(self) -> str | None:
        if self.params["api_key"]:
            return self.params["api_key"]
        for env in _KEY_ENVS:
            if os.environ.get(env):
                return os.environ[env]
        return None

    @property
    def question_type(self) -> str:
        q = self.params["question"]
        if q != "auto":
            return q
        return "noul" if self.runtime.output == "binary" else "score"

    @property
    def language(self) -> str:
        return self.params["language"] or self.runtime.language or "en"

    def native_output(self) -> str:
        return "probability" if self.question_type == "noul" else "score"

    def score_range(self):
        return (0.0, 1.0)

    def levels(self) -> list[str]:
        return list(self.params["levels"]) if self.params["levels"] else default_levels(
            self.language, self.params["task"])

    def build_question(self, reference: str) -> dict:
        return systemone_question(
            reference, question=self.question_type, language=self.language,
            task=self.params["task"], instructions=self.params["instructions"],
            criteria=self.params["criteria"], levels=self.levels(),
        )

    # -- HTTP --------------------------------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            client = self.params["client"]
            if client is None or not hasattr(client, "post"):
                httpx = require("httpx", feature="the method 'systemone'", extra="systemone")
                client = httpx.Client(timeout=self.params["timeout"])
            self._client = client
        return self._client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _unavailable(self) -> BackendUnavailableError:
        from ..hardware.checks import recommend_kev_model

        model = recommend_kev_model()
        return BackendUnavailableError(
            f"No System One server is reachable at {self.base_url}. To run the open Kev model "
            f"on this computer (suggested size for your hardware: {model}), install Kev from "
            "https://github.com/jaredpalmer/kev and start it with:\n"
            f"    uv run --extra serve python -m kev.serve --run {model} --port 8009\n"
            "Hosted Jev (TypeSafe) instead needs method_params={'base_url': ..., 'api_key': ...} "
            "and allow_remote=True, because the texts then leave your computer."
        )

    def _is_connect_error(self, exc: Exception) -> bool:
        return type(exc).__name__ in ("ConnectError", "ConnectTimeout") or isinstance(exc, ConnectionError)

    def health(self) -> None:
        enforce_remote_policy(self.base_url, self.runtime.allow_remote, "The System One method")
        try:
            response = self.client.get(f"{self.base_url}/v1/models", headers=self._headers())
        except Exception as exc:
            if self._is_connect_error(exc):
                raise self._unavailable() from exc
            raise
        if response.status_code in (401, 403):
            raise ConfigurationError(
                f"The System One server at {self.base_url} rejected the request "
                f"(HTTP {response.status_code}); check the API key."
            )
        self._card = {}
        try:
            body = response.json()
        except Exception:
            return
        # TypeSafe/Kev: {"models": [{"name", "release_date", ...}]}; OpenAI style: {"data": [{"id"}]}
        cards = (body.get("models") or body.get("data")) if isinstance(body, dict) else None
        for card in cards if isinstance(cards, list) else []:
            if isinstance(card, dict) and self.params["model"] in (card.get("name"), card.get("id")):
                self._card = card

    def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}{self.params['endpoint']}"
        retries = int(self.params["max_retries"])
        for attempt in range(retries + 1):
            try:
                response = self.client.post(url, json=payload, headers=self._headers())
            except Exception as exc:
                if self._is_connect_error(exc):
                    raise self._unavailable() from exc
                if type(exc).__name__.endswith("Timeout") and attempt < retries:
                    time.sleep(0.5 * 2**attempt)
                    continue
                raise
            status = response.status_code
            if status == 200:
                return response.json()
            detail = response.text[:500]
            if status == 422:
                raise _StateTooLong(detail)
            if status in (401, 403):
                raise ConfigurationError(
                    f"The System One server rejected the request (HTTP {status}); check the API key. {detail}"
                )
            if status == 404:
                if "model" in detail.lower():
                    raise ModelNotAvailableError(
                        f"The model {self.params['model']!r} is not available at {self.base_url}: {detail}"
                    )
                raise ConfigurationError(f"{url} does not exist (HTTP 404); check base_url and endpoint.")
            if status >= 500 and attempt < retries:
                time.sleep(0.5 * 2**attempt)
                continue
            raise ConfigurationError(f"The System One server returned HTTP {status}: {detail}")
        raise AssertionError("unreachable")

    # -- answers -----------------------------------------------------------------------------
    def _value(self, answer: dict) -> float:
        if answer is None:
            return float("nan")
        if answer.get("type", self.question_type) == "noul" or "noul" in answer:
            return float(answer["noul"])
        return float(answer["score"]) / (len(self.levels()) - 1)

    def _ask(self, state: str, questions: dict[str, dict]) -> dict[str, float]:
        payload = {"state": state, "model": self.params["model"], "questions": questions}
        key = ResponseCache.key(backend="systemone", url=self.base_url, endpoint=self.params["endpoint"],
                                payload=payload)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            answers = self._post(payload).get("answers", {})
            values = {name: self._value(answers.get(name)) for name in questions}
        except _StateTooLong as exc:
            segments = regex_sentences(state)
            if len(segments) <= 1:
                raise _SegmentTooLong(
                    "The System One server rejected a response as too long (HTTP 422) and it "
                    f"cannot be split into sentences: {exc.detail}"
                ) from exc
            per_segment = []
            for seg in segments:
                try:
                    per_segment.append(self._ask(seg, questions))
                except _SegmentTooLong:
                    continue  # this sentence alone is still too long; use the others
            if not per_segment:
                raise ConfigurationError(
                    "The System One server rejected a response as too long (HTTP 422), even "
                    f"sentence by sentence: {exc.detail}"
                ) from exc
            values = {name: float(np.nanmax([v[name] for v in per_segment])) for name in questions}
        self._cache.put(key, values)
        return values

    def _score_unique_pairs(self, left: list[str], right: list[str]) -> np.ndarray:
        if self._cache is None:
            self._cache = ResponseCache(self.params["cache_dir"], name="systemone_judgments")
        if not self._ready:
            self.health()
            self._ready = True
        out = np.full(len(left), np.nan)
        if self.params["group_by_response"]:
            groups: dict[str, list[int]] = {}
            for i, text in enumerate(left):
                groups.setdefault(text, []).append(i)
            size = max(1, int(self.params["max_questions"]))
            batches = []
            for text, idx in groups.items():
                for start in range(0, len(idx), size):
                    batches.append((text, idx[start:start + size]))
        else:
            batches = [(left[i], [i]) for i in range(len(left))]
        for text, idx in progress(batches, total=len(batches), desc="asking System One",
                                  enabled=self.runtime.verbose and len(batches) > 20):
            questions = {f"q{k}": self.build_question(right[i]) for k, i in enumerate(idx)}
            values = self._ask(text, questions)
            for k, i in enumerate(idx):
                out[i] = values[f"q{k}"]
        return out

    def model_details(self) -> dict[str, dict]:
        if self._card is None:  # the server is only contacted when the first pair is scored
            try:
                self.health()
            except Exception:  # not running: record the name only
                pass
        card = self._card or {}
        record = {"name": self.params["model"], "source": "systemone", "url": self.base_url,
                  "version": card.get("release_date")}
        # Kev also reports the checkpoint it serves (e.g. "jaredpalmer/kev-4b") and its base model
        record.update({k: card[k] for k in ("run", "base") if card.get(k)})
        return {"systemone": record}

    def describe(self) -> str:
        kind = "yes/no (noul)" if self.question_type == "noul" else f"{len(self.levels())}-level score"
        return f"zero-shot {kind} questions to the System One model {self.params['model']}"
