"""Detect whether coded labels are binary or gradient and convert them to numbers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..config import LABEL_COL_CANDIDATES
from ..exceptions import ConfigurationError


@dataclass(frozen=True)
class LabelInfo:
    kind: str  # "binary" | "gradient"
    min: float
    max: float
    n: int
    positive_rate: float | None = None
    positive_label: object | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["positive_label"] is not None and not isinstance(d["positive_label"], (int, float, str, bool)):
            d["positive_label"] = str(d["positive_label"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> LabelInfo:
        return cls(**d)


def guess_label_col(df: pd.DataFrame) -> str | None:
    for name in LABEL_COL_CANDIDATES:
        if name in df.columns:
            return name
    return None


def to_numeric_labels(y: pd.Series, positive_label: object | None = None) -> np.ndarray:
    """Convert labels to floats. Booleans become 0/1; two string values need ``positive_label``."""
    s = pd.Series(y)
    if s.dtype == bool or str(s.dtype) == "boolean":
        return s.astype(float).to_numpy()
    numeric = pd.to_numeric(s, errors="coerce")
    non_numeric = numeric.isna() & s.notna()
    if not non_numeric.any():
        return numeric.to_numpy(dtype=float)
    values = sorted(s.dropna().astype(str).unique())
    if len(values) != 2:
        raise ConfigurationError(
            f"Labels must be numbers, booleans or two categories; found {len(values)} "
            f"text values ({', '.join(values[:5])}{', ...' if len(values) > 5 else ''})."
        )
    if positive_label is None:
        raise ConfigurationError(
            f"Labels are the text values {values}. Tell Simchology which one means "
            f"'similar' with positive_label=..., e.g. positive_label={values[-1]!r}."
        )
    return np.where(s.isna(), np.nan, (s.astype(str) == str(positive_label)).astype(float))


def detect_label_type(y: pd.Series, *, positive_label: object | None = None) -> LabelInfo:
    values = to_numeric_labels(y, positive_label)
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        raise ConfigurationError("The label column contains no usable values.")
    uniq = np.unique(finite)
    if set(uniq.tolist()) <= {0.0, 1.0}:
        return LabelInfo("binary", 0.0, 1.0, int(finite.size), float(finite.mean()), positive_label)
    return LabelInfo("gradient", float(finite.min()), float(finite.max()), int(finite.size), None, positive_label)


def binarize_labels(
    values: np.ndarray, info: LabelInfo, positive_threshold: float | None
) -> np.ndarray:
    """0/1 labels for binary evaluation. Gradient labels need ``positive_threshold``."""
    values = np.asarray(values, dtype=float)
    if info.kind == "binary":
        return values
    if positive_threshold is None:
        raise ConfigurationError(
            "The coded labels are gradient (e.g. ratings) but binary output was requested. "
            "Set positive_threshold=... to say which label values count as 'similar' "
            f"(labels range from {info.min:g} to {info.max:g}; values >= the threshold count as similar)."
        )
    return np.where(np.isnan(values), np.nan, (values >= positive_threshold).astype(float))


def scale_labels(values: np.ndarray, info: LabelInfo) -> np.ndarray:
    """Min-max scale gradient labels to [0, 1]."""
    span = info.max - info.min
    return (np.asarray(values, dtype=float) - info.min) / (span if span else 1.0)


def unscale_labels(values: np.ndarray, info: LabelInfo) -> np.ndarray:
    return np.asarray(values, dtype=float) * (info.max - info.min) + info.min
