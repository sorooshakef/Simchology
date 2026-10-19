"""OpenAI-compatible chat endpoints (opt-in): hosted APIs or local servers (LM Studio, vLLM, llama.cpp)."""

from __future__ import annotations

import os
from typing import Any

from .._optional import require
from ..exceptions import BackendUnavailableError, ConfigurationError, ModelNotAvailableError
from .netpolicy import enforce_remote_policy, is_local_url

OPENAI_URL = "https://api.openai.com/v1"


class OpenAICompatibleBackend:
    """Chat completions with structured output, falling back gracefully when a server
    does not support JSON schemas, log-probabilities, seeds or temperature."""

    def __init__(self, model: str, *, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 120, max_retries: int = 2, client: Any = None,
                 allow_remote: bool = False) -> None:
        if not model:
            raise ConfigurationError(
                "The method 'openai_compatible' needs a model name: method_params={'model': '...'}."
            )
        self.model = model
        self.base_url = base_url or OPENAI_URL
        enforce_remote_policy(self.base_url, allow_remote, "The OpenAI-compatible method")
        if client is None or not hasattr(client, "chat"):
            openai = require("openai", feature="the method 'openai_compatible'", extra="api")
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                if not is_local_url(self.base_url):
                    raise ConfigurationError(
                        "No API key given: pass method_params={'api_key': ...} or set the "
                        "environment variable OPENAI_API_KEY."
                    )
                key = "not-needed"  # local servers usually ignore the key
            client = openai.OpenAI(api_key=key, base_url=self.base_url, timeout=timeout,
                                   max_retries=max_retries)
        self.client = client
        self.features = {"json_schema": True, "json_object": True, "logprobs": True,
                         "seed": True, "temperature": True}

    def _kwargs(self, messages, schema, temperature, seed) -> dict:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if schema is not None and self.features["json_schema"]:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "judgment", "schema": schema, "strict": True}}
        elif schema is not None and self.features["json_object"]:
            kwargs["response_format"] = {"type": "json_object"}
        if self.features["logprobs"]:
            kwargs.update(logprobs=True, top_logprobs=5)
        if self.features["seed"] and seed is not None:
            kwargs["seed"] = seed
        if self.features["temperature"]:
            kwargs["temperature"] = temperature
        return kwargs

    def _downgrade(self, message: str) -> bool:
        m = message.lower()
        for feature, words in (("json_schema", ("json_schema", "schema")),
                               ("json_object", ("response_format", "json_object")),
                               ("logprobs", ("logprob",)), ("seed", ("seed",)),
                               ("temperature", ("temperature",))):
            if self.features[feature] and any(w in m for w in words):
                self.features[feature] = False
                return True
        return False

    def chat(self, messages: list[dict], *, schema: dict | None, temperature: float,
             seed: int | None) -> tuple[str, Any]:
        for _ in range(6):
            try:
                response = self.client.chat.completions.create(
                    **self._kwargs(messages, schema, temperature, seed))
                break
            except Exception as exc:
                name = type(exc).__name__
                if name in ("APIConnectionError", "APITimeoutError", "ConnectError"):
                    raise BackendUnavailableError(
                        f"Could not reach the server at {self.base_url}. Is it running?"
                    ) from exc
                if name == "NotFoundError":
                    raise ModelNotAvailableError(
                        f"The model {self.model!r} is not available at {self.base_url}."
                    ) from exc
                if name in ("AuthenticationError", "PermissionDeniedError"):
                    raise ConfigurationError(f"The server at {self.base_url} rejected the API key.") from exc
                if name in ("BadRequestError", "UnprocessableEntityError") and self._downgrade(str(exc)):
                    continue
                raise
        else:  # pragma: no cover
            raise RuntimeError("The server rejected every request variant.")
        choice = response.choices[0]
        content = getattr(choice.message, "content", "") or ""
        logprobs = getattr(choice, "logprobs", None)
        lp = getattr(logprobs, "content", None) if logprobs is not None else None
        return content, lp or None
