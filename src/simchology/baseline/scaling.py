"""Transformations of gradient scores that make them interpretable across methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ..exceptions import ConfigurationError
from .baseline import Baseline

SCALES = ("raw", "baseline_percentile", "baseline_z", "baseline_rescaled", "calibrated")


class GradientScaler(ABC):
    kind: ClassVar[str]
    needs: ClassVar[str | None] = None  # "baseline" | "labels"

    def fit(self, **kwargs) -> GradientScaler:
        return self

    @abstractmethod
    def transform(self, scores) -> np.ndarray: ...

    def describe(self) -> str:
        return ""

    def to_dict(self) -> dict:
        return {"kind": self.kind}

    @staticmethod
    def from_dict(d: dict) -> GradientScaler:
        cls = {c.kind: c for c in (RawScaler, BaselinePercentileScaler, BaselineZScaler,
                                   BaselineRescaler, PlattCalibrator, IsotonicCalibrator)}[d["kind"]]
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d):
        return cls()


class RawScaler(GradientScaler):
    kind = "raw"

    def transform(self, scores):
        return np.asarray(scores, dtype=float)


class BaselinePercentileScaler(GradientScaler):
    """Position of a score among the dissimilar baseline pairs (0.5 = typical unrelated pair).

    Note that all scores above the baseline maximum map to 1; use ``baseline_z`` or
    ``baseline_rescaled`` to keep distinctions among clearly similar pairs.
    """

    kind = "baseline_percentile"
    needs = "baseline"

    def __init__(self):
        self.sorted_: np.ndarray | None = None

    def fit(self, *, baseline: Baseline | None = None, **kwargs):
        if baseline is None or not len(baseline):
            raise ConfigurationError("gradient_scale='baseline_percentile' needs a baseline.")
        self.sorted_ = baseline._sorted.copy()
        return self

    def transform(self, scores):
        s = np.asarray(scores, dtype=float)
        lo = np.searchsorted(self.sorted_, s, side="left")
        hi = np.searchsorted(self.sorted_, s, side="right")
        return np.where(np.isnan(s), np.nan, (lo + hi) / (2 * len(self.sorted_)))

    def describe(self):
        return f"percentile of the score among {len(self.sorted_):,} dissimilar baseline pairs"

    def to_dict(self):
        return {"kind": self.kind, "sorted": self.sorted_.tolist()}

    @classmethod
    def _from_dict(cls, d):
        s = cls()
        s.sorted_ = np.asarray(d["sorted"], dtype=float)
        return s


class BaselineZScaler(GradientScaler):
    """(score - baseline mean) / baseline SD: how many SDs above unrelated pairs."""

    kind = "baseline_z"
    needs = "baseline"

    def __init__(self, mean: float | None = None, sd: float | None = None):
        self.mean_, self.sd_ = mean, sd

    def fit(self, *, baseline: Baseline | None = None, **kwargs):
        if baseline is None or len(baseline) < 2:
            raise ConfigurationError("gradient_scale='baseline_z' needs a baseline with at least two pairs.")
        self.mean_, self.sd_ = baseline.mean, baseline.sd or 1.0
        return self

    def transform(self, scores):
        return (np.asarray(scores, dtype=float) - self.mean_) / self.sd_

    def describe(self):
        return "baseline z-score (SDs above the mean of dissimilar pairs)"

    def to_dict(self):
        return {"kind": self.kind, "mean": self.mean_, "sd": self.sd_}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["mean"], d["sd"])


class BaselineRescaler(GradientScaler):
    """(score - baseline median) / (maximum score - baseline median).

    0 is a typical unrelated pair, 1 a perfect match; like BERTScore's baseline
    rescaling, but estimated from the user's own data.
    """

    kind = "baseline_rescaled"
    needs = "baseline"

    def __init__(self, median: float | None = None, upper: float = 1.0):
        self.median_, self.upper_ = median, upper

    def fit(self, *, baseline: Baseline | None = None, upper: float | None = None, **kwargs):
        if baseline is None or not len(baseline):
            raise ConfigurationError("gradient_scale='baseline_rescaled' needs a baseline.")
        self.median_ = baseline.median
        self.upper_ = 1.0 if upper is None else float(upper)
        return self

    def transform(self, scores):
        span = (self.upper_ - self.median_) or 1.0
        return (np.asarray(scores, dtype=float) - self.median_) / span

    def describe(self):
        return "score rescaled so that 0 = median dissimilar pair and 1 = identical texts"

    def to_dict(self):
        return {"kind": self.kind, "median": self.median_, "upper": self.upper_}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["median"], d["upper"])


class PlattCalibrator(GradientScaler):
    """Logistic regression from score to P(similar), fitted on binary coded training pairs."""

    kind = "calibrated_platt"
    needs = "labels"

    def __init__(self, coef: float | None = None, intercept: float | None = None):
        self.coef_, self.intercept_ = coef, intercept

    def fit(self, *, scores=None, labels=None, **kwargs):
        from sklearn.linear_model import LogisticRegression

        s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
        m = ~(np.isnan(s) | np.isnan(y))
        if len(np.unique(y[m])) < 2:
            raise ConfigurationError("Calibration needs coded pairs of both classes.")
        lr = LogisticRegression(C=1e4).fit(s[m].reshape(-1, 1), y[m].astype(int))
        self.coef_, self.intercept_ = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def transform(self, scores):
        z = self.coef_ * np.asarray(scores, dtype=float) + self.intercept_
        return 1 / (1 + np.exp(-z))

    def describe(self):
        return "probability of being similar (logistic calibration on coded training pairs)"

    def to_dict(self):
        return {"kind": self.kind, "coef": self.coef_, "intercept": self.intercept_}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["coef"], d["intercept"])


class IsotonicCalibrator(GradientScaler):
    """Monotone mapping from score onto the label scale, fitted on gradient coded pairs."""

    kind = "calibrated_isotonic"
    needs = "labels"

    def __init__(self, x=None, y=None):
        self.x_ = None if x is None else np.asarray(x, dtype=float)
        self.y_ = None if y is None else np.asarray(y, dtype=float)

    def fit(self, *, scores=None, labels=None, **kwargs):
        from sklearn.isotonic import IsotonicRegression

        s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
        m = ~(np.isnan(s) | np.isnan(y))
        iso = IsotonicRegression(out_of_bounds="clip").fit(s[m], y[m])
        self.x_, self.y_ = iso.X_thresholds_, iso.y_thresholds_
        return self

    def transform(self, scores):
        s = np.asarray(scores, dtype=float)
        return np.where(np.isnan(s), np.nan, np.interp(s, self.x_, self.y_))

    def describe(self):
        return "score mapped onto the coding scale (isotonic calibration on coded training pairs)"

    def to_dict(self):
        return {"kind": self.kind, "x": self.x_.tolist(), "y": self.y_.tolist()}

    @classmethod
    def _from_dict(cls, d):
        return cls(d["x"], d["y"])


def make_scaler(spec: str, *, label_kind: str | None = None) -> GradientScaler:
    if spec in (None, "raw"):
        return RawScaler()
    if spec == "baseline_percentile":
        return BaselinePercentileScaler()
    if spec == "baseline_z":
        return BaselineZScaler()
    if spec == "baseline_rescaled":
        return BaselineRescaler()
    if spec == "calibrated":
        if label_kind is None:
            raise ConfigurationError("gradient_scale='calibrated' needs coded data.")
        return PlattCalibrator() if label_kind == "binary" else IsotonicCalibrator()
    raise ConfigurationError(f"gradient_scale must be one of {SCALES}; got {spec!r}")
