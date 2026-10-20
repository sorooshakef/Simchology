"""Lazy imports of optional dependencies with friendly installation hints.

Heavy packages (torch, transformers, spaCy, POT, ollama, ...) are never imported at
module level anywhere in Simchology. Code that needs one calls :func:`require`, which
either returns the module or raises :class:`MissingDependencyError` telling the user
exactly which ``pip install`` command fixes the problem.
"""

from __future__ import annotations

import importlib
import importlib.util
from functools import cache
from importlib import metadata
from types import ModuleType

from packaging.requirements import Requirement
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
    "httpx": "systemone",
}

# top-level import name -> name on PyPI, where they differ
PIP_NAME: dict[str, str] = {
    "ot": "POT",
    "sentence_transformers": "sentence-transformers",
    "huggingface_hub": "huggingface-hub",
}


_DISPLAY_NAME = {name.lower(): name for name in PIP_NAME.values()}


def is_available(module: str) -> bool:
    """Return True if ``module`` can be imported, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@cache
def extra_packages(extra: str) -> tuple[str, ...]:
    """PyPI names of the packages an extra (as declared in pyproject.toml) installs."""
    try:
        requirements = metadata.requires("simchology") or []
    except metadata.PackageNotFoundError:
        return ()
    names: list[str] = []
    for line in requirements:
        req = Requirement(line)
        if req.marker is None or not req.marker.evaluate({"extra": extra}):
            continue
        if req.name == "simchology":  # an extra that includes other extras
            for sub in sorted(req.extras):
                names += [n for n in extra_packages(sub) if n not in names]
        else:
            # metadata may normalise names to lower case ("POT" -> "pot"); show the usual spelling
            name = _DISPLAY_NAME.get(req.name.lower(), req.name)
            if name not in names:
                names.append(name)
    return tuple(names)


def install_hint(module: str, extra: str | None = None) -> str:
    """The pip command that installs what ``module`` (or the whole ``extra``) needs."""
    extra = extra or EXTRA_FOR.get(module)
    packages = extra_packages(extra) if extra else ()
    return "pip install " + " ".join(packages or (PIP_NAME.get(module, module),))


def source_hint(extra: str) -> str:
    """The command installing an extra from the Simchology source folder."""
    return f'pip install -e ".[{extra}]"'


def install_message(module: str, extra: str | None = None) -> str:
    """Multi-line installation instructions for error messages."""
    extra = extra or EXTRA_FOR.get(module)
    text = f"    {install_hint(module, extra)}"
    if extra:
        text += f"\nor, to install everything this feature needs from the Simchology folder:\n    {source_hint(extra)}"
    return text


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
            f"which is not installed. Install it with:\n{install_message(module, extra)}"
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
