"""Preprocessing for string-, lexical-, topic- and static-embedding-based methods.

Following the paper's data-preparation recommendations, texts are (1) normalised
(symbols/abbreviations unified), (2) tokenised and optionally lemmatised, and (3)
stripped of punctuation and stop words. Sentence segmentation is handled separately
by :class:`~simchology.text.segment.Segmenter`.

Neural, supervised and prompt-based methods do *not* use this preprocessing: they
receive the text with only Unicode and whitespace normalisation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

from .._utils.logging import note
from ..exceptions import ConfigurationError, MissingDependencyError
from .normalize import Normalizer
from .spacy_models import default_spacy_model, load_spacy, spacy_model_installed
from .tokenize import has_alnum, regex_tokens, stopwords_for


@dataclass(frozen=True)
class PreprocessConfig:
    language: str = "de"
    lowercase: bool = True
    unit: str = "auto"  # "auto" | "surface" | "lemma"
    remove_stopwords: bool = True
    remove_punct: bool = True
    language_rules: bool = True
    replacements: tuple[tuple[str, str], ...] = ()
    extra_stopwords: tuple[str, ...] = ()
    spacy_model: str | None = None
    backend: str = "auto"  # "auto" | "spacy" | "regex"

    _FIELDS = ("lowercase", "unit", "remove_stopwords", "remove_punct", "language_rules",
               "replacements", "extra_stopwords", "spacy_model", "backend")

    @classmethod
    def from_options(cls, language: str, options: Mapping | None = None) -> PreprocessConfig:
        options = dict(options or {})
        unknown = set(options) - set(cls._FIELDS)
        if unknown:
            raise ConfigurationError(
                f"Unknown preprocessing option(s) {sorted(unknown)}; valid options are {list(cls._FIELDS)}."
            )
        if isinstance(options.get("replacements"), Mapping):
            options["replacements"] = tuple((str(k), str(v)) for k, v in options["replacements"].items())
        if "extra_stopwords" in options:
            options["extra_stopwords"] = tuple(options["extra_stopwords"])
        cfg = cls(language=language, **options)
        if cfg.unit not in ("auto", "surface", "lemma"):
            raise ConfigurationError("preprocess unit must be 'auto', 'surface' or 'lemma'")
        if cfg.backend not in ("auto", "spacy", "regex"):
            raise ConfigurationError("preprocess backend must be 'auto', 'spacy' or 'regex'")
        return cfg

    def to_dict(self) -> dict:
        d = asdict(self)
        d["replacements"] = [list(r) for r in self.replacements]
        d["extra_stopwords"] = list(self.extra_stopwords)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PreprocessConfig:
        d = dict(d)
        d["replacements"] = tuple(tuple(r) for r in d.get("replacements", ()))
        d["extra_stopwords"] = tuple(d.get("extra_stopwords", ()))
        return cls(**d)


@dataclass(frozen=True)
class TokenAnalysis:
    surface: tuple[str, ...]
    lemmas: tuple[str, ...]
    text: str = field(default="", compare=False)

    def units(self, unit: str) -> tuple[str, ...]:
        return self.lemmas if unit == "lemma" else self.surface


class Preprocessor:
    """Normalise, tokenise, lemmatise and filter texts, caching results per string."""

    def __init__(self, config: PreprocessConfig | None = None, *, verbose: bool = True) -> None:
        self.config = config or PreprocessConfig()
        self.verbose = verbose
        self.normalizer = Normalizer(
            self.config.language, dict(self.config.replacements), self.config.language_rules
        )
        self._backend: str | None = None
        self._cache: dict[str, TokenAnalysis] = {}
        self._stop = stopwords_for(self.config.language) | {w.lower() for w in self.config.extra_stopwords}

    # -- resolution of "auto" settings -------------------------------------------------
    @property
    def spacy_model_name(self) -> str | None:
        return self.config.spacy_model or default_spacy_model(self.config.language)

    @property
    def backend(self) -> str:
        if self._backend is None:
            cfg = self.config
            if cfg.backend == "auto":
                if spacy_model_installed(self.spacy_model_name):
                    self._backend = "spacy"
                else:
                    self._backend = "regex"
                    note(
                        f"No spaCy pipeline for {cfg.language!r} is installed, so texts are "
                        "tokenised with a simple rule and not lemmatised. For lemmatisation run "
                        f"`python -m spacy download {self.spacy_model_name or '<model>'}`.",
                        self.verbose and cfg.unit != "surface",
                    )
            else:
                self._backend = cfg.backend
            if self._backend == "regex" and cfg.unit == "lemma":
                raise MissingDependencyError(
                    "Lemmatisation (unit='lemma') needs spaCy and the pipeline "
                    f"{self.spacy_model_name!r}: pip install \"simchology[nlp]\" and "
                    f"python -m spacy download {self.spacy_model_name}"
                )
        return self._backend

    @property
    def unit(self) -> str:
        if self.config.unit != "auto":
            return self.config.unit
        return "lemma" if self.backend == "spacy" else "surface"

    def nlp(self):
        """The spaCy pipeline in use, or None with the regex backend."""
        return load_spacy(self.config.language, self.config.spacy_model) if self.backend == "spacy" else None

    def describe(self) -> dict:
        return {
            "backend": self.backend,
            "spacy_model": self.spacy_model_name if self.backend == "spacy" else None,
            "unit": self.unit,
            **{k: v for k, v in self.config.to_dict().items() if k not in ("backend", "unit", "spacy_model")},
        }

    # -- processing --------------------------------------------------------------------
    def normalize(self, text: str) -> str:
        return self.normalizer(text)

    def _keep(self, text: str, lower: str, lemma: str, is_punct: bool) -> bool:
        if self.config.remove_punct and (is_punct or not has_alnum(text)):
            return False
        return not (self.config.remove_stopwords and (lower in self._stop or lemma in self._stop))

    def _analyze_uncached(self, texts: list[str]) -> list[TokenAnalysis]:
        lc = self.config.lowercase
        normalized = [self.normalize(t) for t in texts]
        out: list[TokenAnalysis] = []
        if self.backend == "spacy":
            for raw, doc in zip(texts, self.nlp().pipe(normalized, batch_size=64)):
                surface, lemmas = [], []
                for tok in doc:
                    if tok.is_space:
                        continue
                    lower = tok.lower_
                    lemma = (tok.lemma_ or tok.text).lower()
                    if self._keep(tok.text, lower, lemma, tok.is_punct):
                        surface.append(lower if lc else tok.text)
                        lemmas.append(lemma if lc else (tok.lemma_ or tok.text))
                out.append(TokenAnalysis(tuple(surface), tuple(lemmas), raw))
        else:
            for raw, text in zip(texts, normalized):
                toks = [t for t in regex_tokens(text) if self._keep(t, t.lower(), t.lower(), False)]
                toks = [t.lower() for t in toks] if lc else toks
                out.append(TokenAnalysis(tuple(toks), tuple(toks), raw))
        return out

    def analyze(self, texts: Sequence[str]) -> list[TokenAnalysis]:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            for t, a in zip(todo, self._analyze_uncached(todo)):
                self._cache[t] = a
        return [self._cache[t] for t in texts]

    def tokens(self, texts: Sequence[str], unit: str | None = None) -> list[tuple[str, ...]]:
        unit = self.unit if unit in (None, "auto") else unit
        if unit == "lemma" and self.backend != "spacy":
            raise MissingDependencyError(
                "Lemmatisation (unit='lemma') needs spaCy and the pipeline "
                f"{self.spacy_model_name!r}: pip install \"simchology[nlp]\" and "
                f"python -m spacy download {self.spacy_model_name}"
            )
        return [a.units(unit) for a in self.analyze(texts)]
