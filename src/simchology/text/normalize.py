"""Text normalisation: Unicode, whitespace and unification of symbols/abbreviations.

The paper recommends replacing symbols and abbreviations that should be treated as
equivalent (``50%`` vs ``50 Prozent``, ``10m`` vs ``10 Meter``) before string- or
vector-based similarity. Simchology applies a small language-specific rule set by
default and lets users add their own replacements.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

# (regex, replacement). Applied in order.
LANGUAGE_RULES: dict[str, list[tuple[str, str]]] = {
    "de": [(r"\s*%\s*", " Prozent "), (r"\s*&\s*", " und "), (r"\s*€\s*", " Euro ")],
    "en": [(r"\s*%\s*", " percent "), (r"\s*&\s*", " and ")],
    "fr": [(r"\s*%\s*", " pour cent "), (r"\s*&\s*", " et ")],
    "es": [(r"\s*%\s*", " por ciento "), (r"\s*&\s*", " y ")],
    "it": [(r"\s*%\s*", " per cento "), (r"\s*&\s*", " e ")],
    "nl": [(r"\s*%\s*", " procent "), (r"\s*&\s*", " en ")],
    # Exact replication of the original R coding script (coding.R).
    "r_compat": [("ō", "o"), (r"\s*%\s*", " Prozent ")],
}

_WS = re.compile(r"\s+")


class Normalizer:
    """Callable that normalises a text.

    Parameters
    ----------
    language:
        Language whose symbol rules (``%``, ``&``, currency) are applied; ``None`` for none.
        Use ``"r_compat"`` to reproduce the original R coding script exactly.
    replacements:
        Extra literal replacements, e.g. ``{"10m": "10 Meter", "z.B.": "zum Beispiel"}``.
        Keys may also be compiled regular expressions.
    language_rules:
        Whether to apply the language's built-in rules.
    """

    def __init__(
        self,
        language: str | None = None,
        replacements: Mapping | None = None,
        language_rules: bool = True,
    ) -> None:
        self.language = language
        self.language_rules = language_rules
        self.replacements = dict(replacements or {})
        rules: list[tuple[re.Pattern, str]] = []
        for key, value in self.replacements.items():
            pattern = key if isinstance(key, re.Pattern) else re.compile(re.escape(str(key)))
            rules.append((pattern, str(value)))
        if language_rules and language in LANGUAGE_RULES:
            rules += [(re.compile(p), r) for p, r in LANGUAGE_RULES[language]]
        self._rules = rules

    def __call__(self, text: str) -> str:
        text = unicodedata.normalize("NFC", str(text))
        for pattern, repl in self._rules:
            text = pattern.sub(repl, text)
        return _WS.sub(" ", text).strip()


def light_normalize(text: str) -> str:
    """NFC + whitespace only: what neural, supervised and prompt-based methods receive."""
    return _WS.sub(" ", unicodedata.normalize("NFC", str(text))).strip()
