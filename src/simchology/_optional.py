"""Lazy imports of optional dependencies with friendly installation hints.

Heavy packages (torch, transformers, spaCy, POT, ollama, ...) are never imported at
module level anywhere in Simchology. Code that needs one calls :func:`require`, which
either returns the module or raises :class:`MissingDependencyError` telling the user
exactly which ``pip install`` command fixes the problem.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

from packaging.version import Version

from .exceptions import MissingDependencyError

# top-level import name -> Simchology extra that provides it
EXTRA_FOR: dict[str, str] = {
    "spacy": "nlp",
    "wtpsplit": "sat",
    "torch": "embeddings",
    "transformers": "embeddings",
    "sentence_transformers": "embeddings",
    "huggingface_hub": "embeddings",
    "safetensors": "embeddings",
    "gensim": "static",
    "ot": "wmd",
    "peft": "finetune",
    "accelerate": "finetune",
    "ollama": "llm",
    "openai": "api",
    "httpx": "systemone",
}

# top-level import name -> name on PyPI, where they differ
PIP_NAME: dict[str, str] = {
    "ot": "POT",
    "sentence_transformers": "sentence-transformers",
    "huggingface_hub": "huggingface-hub",
}


def is_available(module: str) -> bool:
    """Return True if ``module`` can be imported, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def install_hint(module: str, extra: str | None = None) -> str:
    extra = extra or EXTRA_FOR.get(module)
    if extra:
        return f'pip install "simchology[{extra}]"'
    return f"pip install {PIP_NAME.get(module, module)}"


def require(
    module: str,
    *,
    feature: str,
    extra: str | None = None,
    min_version: str | None = None,
) -> ModuleType:
    """Import and return ``module`` or raise a helpful :class:`MissingDependencyError`.

    Parameters
    ----------
    module:
        Top-level import name, e.g. ``"torch"`` or ``"ot"``.
    feature:
        Human-readable description of what needs the module, used in the message,
        e.g. ``"the method 'wmd'"``.
    extra:
        Simchology extra providing the module; looked up in :data:`EXTRA_FOR` if omitted.
    min_version:
        Minimum version required, checked against ``module.__version__`` if present.
    """
    pip_name = PIP_NAME.get(module, module)
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(
            f"{feature[0].upper()}{feature[1:]} needs the optional package '{pip_name}', "
            f"which is not installed. Install it with:\n    {install_hint(module, extra)}"
        ) from exc
    if min_version is not None:
        installed = getattr(mod, "__version__", None)
        if installed is not None and Version(installed.split("+")[0]) < Version(min_version):
            raise MissingDependencyError(
                f"{feature[0].upper()}{feature[1:]} needs {pip_name}>={min_version}, "
                f"but version {installed} is installed. Upgrade it with:\n"
                f'    pip install --upgrade "{pip_name}>={min_version}"'
            )
    return mod
