"""Base classes shared by all similarity methods.

Every method returns, for row-aligned lists of response and reference texts, one
similarity score per pair where **higher always means more similar**. Distances are
converted to similarities inside the method.

* :class:`RepresentationMethod` - texts are represented once (vectors, token sets, ...)
  and then compared (string, lexical, topic, embedding methods).
* :class:`PairMethod` - both texts are processed jointly (cross-encoders, prompted LLMs,
  System One classifiers); unique pairs are scored once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._optional import is_available
from ..data.labels import LabelInfo
from ..data.validation import did_you_mean
from ..exceptions import ConfigurationError
from ..explain.explanation import Explanation
from ..text.preprocess import PreprocessConfig, Preprocessor


@dataclass(frozen=True)
class MethodInfo:
    name: str
    family: str  # string | lexical | topic | static | alignment | contextual | supervised | prompt
    description: str
    symmetric: bool = True
    requires_corpus_fit: bool = False
    requires_labels: bool = False
    native_output: str = "score"  # "score" (needs a threshold for binary output) | "probability"
    score_range: tuple[float | None, float | None] = (0.0, 1.0)
    granularity: str = "text"  # "text" | "pair"
    segment_default: bool = True  # split responses into sentences by default
    preprocessing: str = "full"  # "full" (normalise + tokenise) | "light" (whitespace only)
    cost_tier: str = "light"  # light | moderate | heavy
    extras: tuple[str, ...] = ()
    requires_modules: tuple[str, ...] = ()
    supports_explain: bool = False
    aliases: tuple[str, ...] = ()
    deterministic: bool = True
    uses_training_pairs: bool = False  # e.g. few-shot examples for prompts


@dataclass
class Runtime:
    """Settings shared by the assessor with the method it runs."""

    device: str = "auto"
    seed: int = 42
    check_hardware: str = "warn"
    allow_remote: bool = False
    verbose: bool = True
    language: str = "de"
    output: str = "gradient"
    label_info: LabelInfo | None = None


def take(rep: Any, idx: np.ndarray) -> Any:
    """Row-select from an encoded representation (ndarray, sparse matrix or list)."""
    if isinstance(rep, np.ndarray) or sp.issparse(rep):
        return rep[idx]
    return [rep[i] for i in idx]


class SimilarityMethod(ABC):
    """Base class of all methods. Subclasses define ``info`` and ``defaults``."""

    info: ClassVar[MethodInfo]
    defaults: ClassVar[dict[str, Any]] = {}
    training_task: ClassVar[str] = "full_finetune"  # used for hardware checks of supervised methods

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.defaults)
        if unknown:
            first = sorted(unknown)[0]
            raise ConfigurationError(
                f"Unknown parameter(s) {sorted(unknown)} for method '{self.info.name}'."
                f"{did_you_mean(first, list(self.defaults))} Valid parameters: "
                f"{', '.join(sorted(self.defaults)) or '(none)'}."
            )
        self.params: dict[str, Any] = {**self.defaults, **params}
        self.runtime = Runtime()
        self._preprocessor: Preprocessor | None = None
        self._fitted = not (self.info.requires_corpus_fit or self.info.requires_labels)

    # -- wiring ----------------------------------------------------------------------
    def bind(self, preprocessor: Preprocessor | None, runtime: Runtime) -> None:
        self._preprocessor = preprocessor
        self.runtime = runtime

    @property
    def preprocessor(self) -> Preprocessor:
        if self._preprocessor is None:
            self._preprocessor = Preprocessor(
                PreprocessConfig(language=self.runtime.language), verbose=self.runtime.verbose
            )
        return self._preprocessor

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def fitted(self) -> bool:
        return self._fitted

    def native_output(self) -> str:
        """"score" (needs a threshold for binary output) or "probability"."""
        return self.info.native_output

    def score_range(self) -> tuple[float | None, float | None]:
        return self.info.score_range

    # -- the interface -----------------------------------------------------------------
    def fit(
        self,
        corpus: Sequence[str] | None = None,
        train: pd.DataFrame | None = None,
        dev: pd.DataFrame | None = None,
    ) -> SimilarityMethod:
        """Fit on an unlabelled ``corpus`` (vocabularies, idf, topics) and/or labelled pairs.

        ``train``/``dev`` are frames with columns ``left``, ``right``, ``label`` (and
        ``group`` if available). Methods that need nothing simply return ``self``.
        """
        self._fitted = True
        return self

    @abstractmethod
    def score_pairs(self, left: Sequence[str], right: Sequence[str]) -> np.ndarray:
        """Similarity of ``left[i]`` (response) and ``right[i]`` (reference) for every i."""

    def explain_pair(self, left: str, right: str, top_k: int = 10) -> Explanation:
        raise ConfigurationError(f"The method '{self.name}' does not provide explanations.")

    def resource_request(self, n_pairs: int, task: str = "inference"):
        """A :class:`~simchology.hardware.ResourceRequest` for hardware checks, or None."""
        return None

    def check_hardware(self, task: str, n_items: int) -> None:
        """Warn (or raise, with ``check_hardware="error"``) if the machine looks insufficient."""
        policy = self.runtime.check_hardware
        if policy == "off":
            return
        req = self.resource_request(n_items, task)
        if req is None:
            return
        from ..hardware.checks import check_resources, emit

        emit(check_resources(req, device=self.runtime.device, policy=policy), policy)

    def describe(self) -> str:
        """Short description used in methods-section summaries."""
        return self.info.description

    def model_ids(self) -> dict[str, str]:
        """Models (and revisions) used, for provenance records."""
        return {}

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)

    # -- persistence -------------------------------------------------------------------
    def save(self, directory: Path) -> None:  # noqa: B027 - optional hook
        """Write fitted state to ``directory`` (nothing for stateless methods)."""

    def load(self, directory: Path, *, trust: bool = False) -> None:
        """Restore fitted state written by :meth:`save`."""
        self._fitted = True

    # -- helpers -----------------------------------------------------------------------
    @classmethod
    def missing_modules(cls, params: dict | None = None) -> list[str]:
        """Optional modules this method needs (with these ``params``) that are not installed."""
        return [m for m in cls.info.requires_modules if not is_available(m)]

    def __repr__(self) -> str:
        changed = {k: v for k, v in self.params.items() if self.defaults.get(k) != v}
        args = ", ".join(f"{k}={v!r}" for k, v in changed.items())
        return f"{type(self).__name__}({args})"


class RepresentationMethod(SimilarityMethod):
    """Encode each unique text once, then compare encodings row by row."""

    @abstractmethod
    def encode(self, texts: list[str]) -> Any:
        """Encode texts; the result must support row selection (see :func:`take`)."""

    @abstractmethod
    def compare(self, a: Any, b: Any) -> np.ndarray:
        """Row-aligned similarity of two encodings."""

    def score_pairs(self, left: Sequence[str], right: Sequence[str]) -> np.ndarray:
        if len(left) != len(right):
            raise ValueError("left and right must have the same length")
        if not len(left):
            return np.zeros(0)
        uniq = list(dict.fromkeys([*left, *right]))
        pos = {t: i for i, t in enumerate(uniq)}
        rep = self.encode(uniq)
        ia = np.fromiter((pos[t] for t in left), dtype=np.int64, count=len(left))
        ib = np.fromiter((pos[t] for t in right), dtype=np.int64, count=len(right))
        return np.asarray(self.compare(take(rep, ia), take(rep, ib)), dtype=float)


class PairMethod(SimilarityMethod):
    """Process both texts jointly; each unique (response, reference) pair is scored once."""

    def score_pairs(self, left: Sequence[str], right: Sequence[str]) -> np.ndarray:
        if len(left) != len(right):
            raise ValueError("left and right must have the same length")
        if not len(left):
            return np.zeros(0)
        keys = list(zip(left, right))
        uniq = list(dict.fromkeys(keys))
        scores = np.asarray(
            self._score_unique_pairs([a for a, _ in uniq], [b for _, b in uniq]), dtype=float
        )
        lookup = dict(zip(uniq, scores))
        return np.array([lookup[k] for k in keys], dtype=float)

    @abstractmethod
    def _score_unique_pairs(self, left: list[str], right: list[str]) -> np.ndarray:
        """Score pairs that are guaranteed to be unique."""
