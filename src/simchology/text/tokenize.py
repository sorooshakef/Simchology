"""Tokenisation, stop words and language detection that work without spaCy models."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache

from .._optional import is_available

WORD_RE = re.compile(r"\w+(?:[-'’]\w+)*", re.UNICODE)

# Used only when spaCy is not installed.
_GERMAN_STOPWORDS = frozenset("""
aber alle allem allen aller alles als also am an ander andere anderem anderen anderer anderes
anderm andern anderr anders auch auf aus bei bin bis bist da damit dann das dass dasselbe dazu
dein deine deinem deinen deiner deines dem demselben den denn denselben der derer derselbe
derselben des desselben dessen dich die dies diese dieselbe dieselben diesem diesen dieser dieses
dir doch dort du durch ein eine einem einen einer eines einig einige einigem einigen einiger
einiges einmal er es etwas euch euer eure eurem euren eurer eures für gegen gewesen hab habe
haben hat hatte hatten hier hin hinter ich ihm ihn ihnen ihr ihre ihrem ihren ihrer ihres im in
indem ins ist jede jedem jeden jeder jedes jene jenem jenen jener jenes jetzt kann kein keine
keinem keinen keiner keines können könnte machen man manche manchem manchen mancher manches mein
meine meinem meinen meiner meines mich mir mit muss musste nach nicht nichts noch nun nur ob oder
ohne sehr sein seine seinem seinen seiner seines selbst sich sie sind so solche solchem solchen
solcher solches soll sollte sondern sonst über um und uns unser unsere unserem unseren unserer
unseres unter viel vom von vor während war waren warst was weg weil weiter welche welchem welchen
welcher welches wenn werde werden wie wieder will wir wird wirst wo wollen wollte würde würden zu
zum zur zwar zwischen
""".split())

_FALLBACK_LANGS = ("de", "en", "fr", "es", "it", "nl", "pt")


def regex_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text)


@lru_cache(maxsize=32)
def stopwords_for(language: str) -> frozenset[str]:
    """Stop words for ``language`` (spaCy's lists if spaCy is installed)."""
    if is_available("spacy"):
        try:
            from spacy.util import get_lang_class

            return frozenset(w.lower() for w in get_lang_class(language)().Defaults.stop_words)
        except Exception:
            pass
    if language == "en":
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        return frozenset(ENGLISH_STOP_WORDS)
    if language == "de":
        return _GERMAN_STOPWORDS
    return frozenset()


def detect_language(texts: Sequence[str], sample: int = 300) -> str | None:
    """Guess the language of ``texts`` from stop-word frequencies; ``None`` if unsure."""
    words: Counter = Counter()
    for t in list(texts)[:sample]:
        words.update(w.lower() for w in regex_tokens(str(t)))
    if not words:
        return None
    scores = {}
    for lang in _FALLBACK_LANGS:
        sw = stopwords_for(lang)
        scores[lang] = sum(c for w, c in words.items() if w in sw)
    best = max(scores, key=scores.get)
    total = sum(words.values())
    return best if scores[best] >= 0.1 * total else None


def has_alnum(text: str) -> bool:
    return any(ch.isalnum() for ch in text)
