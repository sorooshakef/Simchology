"""Thin wrapper around the ``ollama`` Python client with user-friendly errors."""

from __future__ import annotations

import sys
from typing import Any

from .._optional import require
from .._utils.logging import note
from ..exceptions import BackendUnavailableError, ModelNotAvailableError
from .netpolicy import enforce_remote_policy


def _start_instructions(host: str) -> str:
    if sys.platform == "darwin" or sys.platform == "win32":
        start = "open the Ollama app (it runs in the menu bar / system tray)"
    else:
        start = "run `ollama serve` in a terminal (or `systemctl start ollama`)"
    return (
        f"Ollama is not running at {host}. Install it from https://ollama.com/download "
        f"and {start}. Ollama runs the model on your own computer, so no texts leave it."
    )


def _field(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        value = obj[key]
    except (TypeError, KeyError, IndexError):
        value = getattr(obj, key, default)
    return default if value is None else value


def _matches(name: str, wanted: str) -> bool:
    if name == wanted:
        return True
    if ":" not in wanted:
        return name == f"{wanted}:latest"
    return False


class OllamaBackend:
    """Talk to a (normally local) Ollama server.

    ``client`` may be any object with the methods ``list``, ``show``, ``chat`` and
    ``pull`` of :class:`ollama.Client` (tests inject fakes).
    """

    def __init__(self, host: str, *, timeout: float = 120, client: Any = None,
                 allow_remote: bool = False, verbose: bool = True) -> None:
        enforce_remote_policy(host, allow_remote, "The Ollama method")
        self.host = host
        self.verbose = verbose
        if client is None or not hasattr(client, "chat"):
            ollama = require("ollama", feature="the method 'llm' (local LLMs via Ollama)", extra="llm")
            client = ollama.Client(host=host, timeout=timeout)
        self.client = client
        self.supports_logprobs: bool | None = None
        self.supports_think: bool = True

    # -- server and model checks ------------------------------------------------------
    def _call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ConnectionError as exc:
            raise BackendUnavailableError(_start_instructions(self.host)) from exc
        except Exception as exc:
            if type(exc).__name__ in ("ConnectError", "ConnectTimeout"):
                raise BackendUnavailableError(_start_instructions(self.host)) from exc
            raise

    def models(self) -> dict[str, dict]:
        response = self._call(self.client.list)
        out = {}
        for m in _field(response, "models", []) or []:
            name = _field(m, "model") or _field(m, "name")
            if name:
                out[name] = {"digest": _field(m, "digest"), "size": _field(m, "size"),
                             "details": _field(m, "details")}
        return out

    def health(self) -> None:
        self.models()

    def ensure_model(self, model: str, *, auto_pull: bool = False) -> dict:
        """Make sure ``model`` is available locally and return its size/context information."""
        available = self.models()
        name = next((n for n in available if _matches(n, model)), None)
        if name is None:
            if not auto_pull:
                raise ModelNotAvailableError(
                    f"The Ollama model {model!r} has not been downloaded yet. Download it once "
                    f"with `ollama pull {model}` in a terminal, or pass "
                    "method_params={'auto_pull': True}."
                )
            note(f"Downloading the Ollama model {model!r} (this can take a while) ...", self.verbose)
            self._call(self.client.pull, model)
            available = self.models()
            name = next((n for n in available if _matches(n, model)), model)
        info = dict(available.get(name, {}))
        info["name"] = name
        try:
            shown = self._call(self.client.show, name)
        except Exception:
            shown = None
        modelinfo = _field(shown, "modelinfo", {}) or {}
        for key, value in dict(modelinfo).items():
            if key.endswith(".context_length"):
                info["context_length"] = int(value)
            elif key.endswith(".block_count"):
                info["n_layers"] = int(value)
            elif key.endswith(".embedding_length"):
                info["hidden_size"] = int(value)
        details = _field(shown, "details") or info.get("details")
        info["parameter_size"] = _field(details, "parameter_size")
        info["quantization"] = _field(details, "quantization_level")
        info.pop("details", None)
        return info

    # -- generation -------------------------------------------------------------------
    def chat(self, model: str, messages: list[dict], *, format: dict | None, options: dict,
             think: bool | None, keep_alive: Any, logprobs: bool) -> tuple[str, Any]:
        """Return (content, token log-probabilities or None)."""
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "format": format,
                                  "options": options, "keep_alive": keep_alive}
        if think is not None and self.supports_think:
            kwargs["think"] = think
        want_lp = logprobs and self.supports_logprobs is not False
        if want_lp:
            kwargs.update(logprobs=True, top_logprobs=5)
        for _ in range(3):
            try:
                response = self._call(self.client.chat, **kwargs)
                break
            except TypeError as exc:  # older client without logprobs/think arguments
                msg = str(exc)
                if "logprobs" in msg:
                    kwargs.pop("logprobs", None)
                    kwargs.pop("top_logprobs", None)
                    self.supports_logprobs = False
                elif "think" in msg:
                    kwargs.pop("think", None)
                    self.supports_think = False
                else:
                    raise
            except Exception as exc:
                msg = str(exc).lower()
                status = getattr(exc, "status_code", None)
                if status == 404 or ("not found" in msg and "model" in msg):
                    raise ModelNotAvailableError(
                        f"The Ollama model {model!r} is not available: `ollama pull {model}`."
                    ) from exc
                if "think" in msg and "think" in kwargs:
                    kwargs.pop("think")
                    self.supports_think = False
                elif "logprob" in msg and "logprobs" in kwargs:
                    kwargs.pop("logprobs")
                    kwargs.pop("top_logprobs", None)
                    self.supports_logprobs = False
                else:
                    raise
        else:  # pragma: no cover - three consecutive feature downgrades
            raise RuntimeError("Ollama rejected the request.")
        message = _field(response, "message")
        content = _field(message, "content", "") or ""
        lp = _field(response, "logprobs")
        if want_lp:
            if lp:
                self.supports_logprobs = True
            elif self.supports_logprobs is None:
                self.supports_logprobs = False
        return content, lp or None
