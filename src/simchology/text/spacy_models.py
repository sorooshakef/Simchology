"""Loading (and explaining how to install) spaCy pipelines."""

from __future__ import annotations

from functools import lru_cache

from .._optional import is_available, require
from ..config import SPACY_MODELS
from ..exceptions import ModelNotAvailableError


def default_spacy_model(language: str) -> str | None:
    return SPACY_MODELS.get(language)


def spacy_model_installed(name: str | None) -> bool:
    if not name or not is_available("spacy"):
        return False
    import spacy.util

    return bool(spacy.util.is_package(name)) or is_available(name)


@lru_cache(maxsize=8)
def load_spacy(language: str, model: str | None = None):
    """Load a spaCy pipeline (NER disabled, as it is not needed)."""
    spacy = require("spacy", feature="spaCy-based preprocessing", extra="nlp")
    name = model or default_spacy_model(language)
    if name is None:
        raise ModelNotAvailableError(
            f"No default spaCy pipeline is known for language {language!r}. Pass "
            "method_params={'spacy_model': '<name>'} or preprocess={'backend': 'regex'}."
        )
    if not spacy_model_installed(name):
        raise ModelNotAvailableError(
            f"The spaCy pipeline {name!r} is not installed. Install it once with:\n"
            f"    python -m spacy download {name}\n"
            f"or from Python: simchology.download_resources(language={language!r})"
        )
    return spacy.load(name, disable=["ner"])
