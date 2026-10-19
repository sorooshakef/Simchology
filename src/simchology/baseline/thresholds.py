"""Decision rules that turn similarity scores into binary judgments.

A pair is judged similar when its score is *greater than* its threshold.

* :class:`BaselineQuantileThreshold` (default for score-based methods): the q-th
  quantile (default 0.95) of the scores of known-dissimilar pairs.
* :class:`OptimizedThreshold`: the cut that maximises F1 (or Youden's J, balanced
  accuracy, Cohen's kappa) on the coded *training* pairs.
* :class:`ReferenceThreshold`: variants (a)-(d) of the original coding scripts, derived
  from how similar *distinct* references are to each other:

  - (a) ``overall``         - the highest similarity between any two distinct references;
  - (b) ``fixed_groupwise`` - the highest similarity between two references of the same group;
  - (c) ``groupwise``       - per group, the highest within-group reference similarity;
  - (d) ``itemwise``        - per reference, its highest similarity to another reference
    of its group.

* :class:`FixedThreshold` / :class:`ProbabilityThreshold`: a fixed value (0.5 for
  probabilities).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .._utils.warn import warn
from ..config import GROUP, RIGHT
from ..exceptions import ConfigurationError, NotFittedError, SimchologyDataWarning
from .baseline import Baseline

_ALL = "__all__"


def _as_str(values) -> np.ndarray | None:
    if values is None:
        return None
    return np.asarray(["" if pd.isna(v) else str(v) for v in values], dtype=object)


class Threshold(ABC):
    kind: ClassVar[str]
    needs: ClassVar[str | None] = None  # "baseline" | "labels" | "references" | None

    @abstractmethod
    def fit(self, **kwargs: Any) -> Threshold: ...

    @abstractmethod
    def values(self, n: int, groups=None, refs=None) -> np.ndarray: ...

    @abstractmethod
    def describe(self) -> str: ...

    @abstractmethod
    def to_dict(self) -> dict: ...

    def apply(self, scores, groups=None, refs=None) -> np.ndarray:
        """1.0 where ``score > threshold``, 0.0 otherwise, NaN for missing scores."""
        s = np.asarray(scores, dtype=float)
        thr = self.values(len(s), groups, refs)
        return np.where(np.isnan(s), np.nan, (s > thr).astype(float))

    @staticmethod
    def from_dict(d: dict) -> Threshold:
        cls = {c.kind: c for c in (FixedThreshold, ProbabilityThreshold, BaselineQuantileThreshold,
                                   OptimizedThreshold, ReferenceThreshold)}[d["kind"]]
        return cls._from_dict(d)


class FixedThreshold(Threshold):
    kind = "fixed"

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def fit(self, **kwargs):
        return self

    def values(self, n, groups=None, refs=None):
        return np.full(n, self.value)

    def describe(self):
        return f"similar if score > {self.value:g} (fixed threshold)"

    def to_dict(self):
        return {"kind": self.kind, "value": self.value}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["value"])


class ProbabilityThreshold(FixedThreshold):
    kind = "probability"

    def __init__(self, value: float = 0.5) -> None:
        super().__init__(value)

    def describe(self):
        return f"similar if predicted probability > {self.value:g}"


class BaselineQuantileThreshold(Threshold):
    kind = "baseline_quantile"
    needs = "baseline"

    def __init__(self, q: float = 0.95, per_group: bool = False, min_group_n: int = 30) -> None:
        if not 0 < q < 1:
            raise ConfigurationError("baseline_quantile must be between 0 and 1, e.g. 0.95")
        self.q = float(q)
        self.per_group = per_group
        self.min_group_n = int(min_group_n)
        self.value_: float | None = None
        self.group_values_: dict[str, float] = {}
        self.n_baseline_: int = 0

    def fit(self, *, baseline: Baseline | None = None, **kwargs):
        if baseline is None or not len(baseline):
            raise ConfigurationError("A baseline-quantile threshold needs a non-empty baseline.")
        self.value_ = baseline.quantile(self.q)
        self.n_baseline_ = len(baseline)
        if len(baseline) < 200 and self.q >= 0.95:
            warn(
                f"The baseline has only {len(baseline)} pairs, so its {self.q:.0%} quantile rests "
                f"on about {max(1, round(len(baseline) * (1 - self.q)))} values and is unstable. "
                "Provide more coded data or a group column for re-pairing.",
                SimchologyDataWarning,
            )
        if self.per_group and "group_left" in baseline.pairs:
            for g in baseline.pairs["group_left"].dropna().astype(str).unique():
                sub = baseline.for_group(g)
                if sub is not None and len(sub) >= self.min_group_n:
                    self.group_values_[g] = sub.quantile(self.q)
        return self

    def values(self, n, groups=None, refs=None):
        if self.value_ is None:
            raise NotFittedError("threshold not fitted")
        if not self.per_group or groups is None:
            return np.full(n, self.value_)
        g = _as_str(groups)
        return np.array([self.group_values_.get(x, self.value_) for x in g], dtype=float)

    def describe(self):
        where = " per group" if self.per_group else ""
        val = f" = {self.value_:.4g}" if self.value_ is not None and not self.per_group else ""
        return (f"similar if score > the {self.q:.0%} quantile of {self.n_baseline_:,} "
                f"dissimilar baseline pairs{where}{val}")

    def to_dict(self):
        return {"kind": self.kind, "q": self.q, "per_group": self.per_group,
                "min_group_n": self.min_group_n, "value": self.value_,
                "group_values": self.group_values_, "n_baseline": self.n_baseline_}

    @classmethod
    def _from_dict(cls, d):
        t = cls(d["q"], d["per_group"], d["min_group_n"])
        t.value_, t.group_values_, t.n_baseline_ = d["value"], d["group_values"], d["n_baseline"]
        return t


def best_cut(scores, labels, metric: str = "f1") -> float:
    """Threshold maximising ``metric`` for the rule ``score > threshold``."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    mask = ~(np.isnan(s) | np.isnan(y))
    s, y = s[mask], y[mask]
    if len(s) == 0 or len(np.unique(y)) < 2:
        raise ConfigurationError("Optimising a threshold needs coded pairs of both classes.")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    pos, n = y.sum(), len(y)
    fn = pos - tp
    tn = (n - pos) - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "f1":
            m = 2 * tp / (2 * tp + fp + fn)
        elif metric == "youden":
            m = tp / pos - fp / (n - pos)
        elif metric == "balanced_accuracy":
            m = 0.5 * (tp / pos + tn / (n - pos))
        elif metric == "kappa":
            po = (tp + tn) / n
            pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
            m = (po - pe) / (1 - pe)
        else:
            raise ConfigurationError("metric must be f1, youden, balanced_accuracy or kappa")
    m = np.nan_to_num(m, nan=-np.inf)
    valid = np.r_[s[:-1] > s[1:], True]  # only cut between distinct scores
    m[~valid] = -np.inf
    k = int(np.argmax(m))
    if k < n - 1:
        return float((s[k] + s[k + 1]) / 2)
    return float(s[k] - 1e-9)


class OptimizedThreshold(Threshold):
    kind = "optimized"
    needs = "labels"

    def __init__(self, metric: str = "f1", per_group: bool = False, min_group_n: int = 30) -> None:
        self.metric = metric
        self.per_group = per_group
        self.min_group_n = min_group_n
        self.value_: float | None = None
        self.group_values_: dict[str, float] = {}

    def fit(self, *, scores=None, labels=None, groups=None, **kwargs):
        if scores is None or labels is None:
            raise ConfigurationError("An optimised threshold needs coded training pairs.")
        self.value_ = best_cut(scores, labels, self.metric)
        if self.per_group and groups is not None:
            g = _as_str(groups)
            for grp in np.unique(g):
                m = g == grp
                if m.sum() >= self.min_group_n and len(np.unique(np.asarray(labels)[m])) == 2:
                    self.group_values_[grp] = best_cut(np.asarray(scores)[m], np.asarray(labels)[m], self.metric)
        return self

    def values(self, n, groups=None, refs=None):
        if self.value_ is None:
            raise NotFittedError("threshold not fitted")
        if not self.per_group or groups is None:
            return np.full(n, self.value_)
        return np.array([self.group_values_.get(x, self.value_) for x in _as_str(groups)], dtype=float)

    def describe(self):
        where = " per group" if self.per_group else ""
        val = f" = {self.value_:.4g}" if self.value_ is not None and not self.per_group else ""
        return f"similar if score > the threshold maximising {self.metric} on the coded training pairs{where}{val}"

    def to_dict(self):
        return {"kind": self.kind, "metric": self.metric, "per_group": self.per_group,
                "min_group_n": self.min_group_n, "value": self.value_, "group_values": self.group_values_}

    @classmethod
    def _from_dict(cls, d):
        t = cls(d["metric"], d["per_group"], d["min_group_n"])
        t.value_, t.group_values_ = d["value"], d["group_values"]
        return t


_VARIANTS = {"a": "overall", "b": "fixed_groupwise", "c": "groupwise", "d": "itemwise"}


class ReferenceThreshold(Threshold):
    kind = "reference"
    needs = "references"

    def __init__(self, variant: str = "c", max_references: int = 2000) -> None:
        variant = {v: k for k, v in _VARIANTS.items()}.get(variant, variant)
        if variant not in _VARIANTS:
            raise ConfigurationError("reference threshold variant must be a, b, c or d "
                                     "(overall, fixed_groupwise, groupwise, itemwise)")
        self.variant = variant
        self.max_references = max_references
        self.overall_: float | None = None
        self.fixed_groupwise_: float | None = None
        self.groupwise_: dict[str, float] = {}
        self.itemwise_: dict[str, float] = {}
        self.n_references_ = 0

    def fit(self, *, references: pd.DataFrame | None = None,
            reference_scorer: Callable[[list[str], list[str]], np.ndarray] | None = None,
            seed: int = 42, **kwargs):
        if references is None or reference_scorer is None:
            raise ConfigurationError("A reference-based threshold needs the set of references.")
        refs = references.drop_duplicates(RIGHT)
        if len(refs) > self.max_references:
            warn(f"Using a random sample of {self.max_references} of {len(refs)} references "
                          "for the reference-based threshold.", SimchologyDataWarning)
            refs = refs.sample(self.max_references, random_state=seed)
        texts = refs[RIGHT].astype(str).tolist()
        groups = (refs[GROUP].astype(str).tolist() if GROUP in refs else [_ALL] * len(texts))
        n = len(texts)
        if n < 2:
            raise ConfigurationError("A reference-based threshold needs at least two distinct references.")
        ia, ib = np.where(~np.eye(n, dtype=bool))
        scores = reference_scorer([texts[i] for i in ia], [texts[j] for j in ib])
        mat = np.full((n, n), np.nan)
        mat[ia, ib] = scores
        self.overall_ = float(np.nanmax(mat))
        g = np.array(groups, dtype=object)
        for a in range(n):
            same = (g == g[a]) & (np.arange(n) != a)
            if same.any() and not np.all(np.isnan(mat[a, same])):
                self.itemwise_[texts[a]] = float(np.nanmax(mat[a, same]))
        for grp in np.unique(g):
            vals = [self.itemwise_[texts[a]] for a in np.flatnonzero(g == grp) if texts[a] in self.itemwise_]
            if vals:
                self.groupwise_[str(grp)] = float(np.max(vals))
        self.fixed_groupwise_ = max(self.groupwise_.values()) if self.groupwise_ else self.overall_
        self.n_references_ = n
        return self

    def values(self, n, groups=None, refs=None):
        if self.overall_ is None:
            raise NotFittedError("threshold not fitted")
        v = self.variant
        if v == "a":
            return np.full(n, self.overall_)
        if v == "b":
            return np.full(n, self.fixed_groupwise_)
        g = _as_str(groups) if groups is not None else np.full(n, _ALL, dtype=object)
        group_vals = np.array([self.groupwise_.get(x, self.overall_) for x in g], dtype=float)
        if v == "c" or refs is None:
            return group_vals
        r = _as_str(refs)
        return np.array([self.itemwise_.get(x, gv) for x, gv in zip(r, group_vals)], dtype=float)

    def describe(self):
        text = {
            "a": "the highest similarity between any two distinct references",
            "b": "the highest similarity between two distinct references of the same group",
            "c": "per group, the highest similarity between two distinct references of that group",
            "d": "per reference, its highest similarity to another reference of its group",
        }[self.variant]
        return f"similar if score > {text} (reference-based variant {self.variant}, {self.n_references_} references)"

    def to_dict(self):
        return {"kind": self.kind, "variant": self.variant, "max_references": self.max_references,
                "overall": self.overall_, "fixed_groupwise": self.fixed_groupwise_,
                "groupwise": self.groupwise_, "itemwise": self.itemwise_, "n_references": self.n_references_}

    @classmethod
    def _from_dict(cls, d):
        t = cls(d["variant"], d["max_references"])
        t.overall_, t.fixed_groupwise_ = d["overall"], d["fixed_groupwise"]
        t.groupwise_, t.itemwise_, t.n_references_ = d["groupwise"], d["itemwise"], d["n_references"]
        return t


def make_threshold(spec: Any, *, native_output: str = "score", baseline_quantile: float = 0.95) -> Threshold:
    """Create a threshold from a user specification (see :func:`simchology.assess`)."""
    if isinstance(spec, Threshold):
        return spec
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return FixedThreshold(spec)
    if spec in (None, "auto"):
        if native_output == "probability":
            return ProbabilityThreshold(0.5)
        return BaselineQuantileThreshold(baseline_quantile)
    key = str(spec).lower()
    if key in ("baseline", "baseline_quantile"):
        return BaselineQuantileThreshold(baseline_quantile)
    if key in ("baseline_group", "baseline_per_group"):
        return BaselineQuantileThreshold(baseline_quantile, per_group=True)
    if key in ("optimize", "optimise", "optimize_f1"):
        return OptimizedThreshold("f1")
    if key.startswith("optimize_"):
        return OptimizedThreshold(key.split("_", 1)[1])
    if key in ("probability", "0.5"):
        return ProbabilityThreshold(0.5)
    if key.startswith("reference"):
        variant = key.split("_", 1)[1] if "_" in key else "c"
        return ReferenceThreshold(variant)
    raise ConfigurationError(
        f"Unknown threshold {spec!r}. Use 'auto', 'baseline', 'baseline_group', 'optimize', "
        "'optimize_youden', 'reference_a' ... 'reference_d', or a number."
    )
