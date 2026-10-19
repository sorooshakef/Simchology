"""The distribution of similarity scores among dissimilar pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Baseline:
    """Scores of known-dissimilar pairs, used to set thresholds and interpret scores.

    A score at the baseline's 95th percentile means "as similar as the most similar 5%
    of unrelated pairs". Use :meth:`summary` to see what values to expect for unrelated
    texts with the chosen method.
    """

    scores: np.ndarray
    pairs: pd.DataFrame
    method: str
    strategy: str
    seed: int

    def __post_init__(self) -> None:
        self.scores = np.asarray(self.scores, dtype=float)
        finite = ~np.isnan(self.scores)
        self._sorted = np.sort(self.scores[finite])

    def __len__(self) -> int:
        return len(self._sorted)

    @property
    def mean(self) -> float:
        return float(np.mean(self._sorted)) if len(self) else float("nan")

    @property
    def sd(self) -> float:
        return float(np.std(self._sorted, ddof=1)) if len(self) > 1 else float("nan")

    @property
    def median(self) -> float:
        return float(np.median(self._sorted)) if len(self) else float("nan")

    def quantile(self, q: float) -> float:
        return float(np.quantile(self._sorted, q)) if len(self) else float("nan")

    def percentile_of(self, scores) -> np.ndarray:
        """Mid-rank position of ``scores`` within the baseline, in [0, 1]."""
        s = np.asarray(scores, dtype=float)
        if not len(self):
            return np.full(s.shape, np.nan)
        lo = np.searchsorted(self._sorted, s, side="left")
        hi = np.searchsorted(self._sorted, s, side="right")
        out = (lo + hi) / (2 * len(self._sorted))
        return np.where(np.isnan(s), np.nan, out)

    def for_group(self, group) -> Baseline | None:
        if "group_left" not in self.pairs:
            return None
        mask = (self.pairs["group_left"].astype(str) == str(group)).to_numpy()
        if not mask.any():
            return None
        return Baseline(self.scores[mask], self.pairs[mask].reset_index(drop=True), self.method,
                        self.strategy, self.seed)

    def summary(self) -> pd.DataFrame:
        """n, mean, sd and upper quantiles of the baseline, overall and per source."""
        rows = []
        sources = [("all", np.ones(len(self.scores), dtype=bool))]
        if "source" in self.pairs and self.pairs["source"].nunique() > 1:
            sources += [(s, (self.pairs["source"] == s).to_numpy()) for s in self.pairs["source"].unique()]
        for name, mask in sources:
            v = self.scores[mask]
            v = v[~np.isnan(v)]
            rows.append({
                "source": name, "n": len(v),
                "mean": np.mean(v) if len(v) else np.nan,
                "sd": np.std(v, ddof=1) if len(v) > 1 else np.nan,
                **{f"q{int(q * 100)}": (np.quantile(v, q) if len(v) else np.nan) for q in (0.5, 0.9, 0.95, 0.99)},
                "max": np.max(v) if len(v) else np.nan,
            })
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (f"Baseline(method={self.method!r}, strategy={self.strategy!r}, n={len(self)}, "
                f"mean={self.mean:.3f}, q95={self.quantile(0.95):.3f})")

    # persistence
    def to_arrays(self) -> dict[str, np.ndarray]:
        arrays = {"scores": self.scores}
        for col in ("left", "right", "source", "group_left"):
            if col in self.pairs:
                arrays[col] = np.asarray(self.pairs[col].astype(str).tolist(), dtype=str)
        return arrays

    @classmethod
    def from_arrays(cls, arrays, *, method: str, strategy: str, seed: int) -> Baseline:
        cols = {c: arrays[c] for c in ("left", "right", "source", "group_left") if c in arrays}
        return cls(np.asarray(arrays["scores"]), pd.DataFrame(cols), method, strategy, seed)
