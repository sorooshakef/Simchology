"""The evaluation report shown after assessing similarity against coded data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_HEADLINE = {
    "binary": ["precision", "recall", "f1", "accuracy", "cohen_kappa", "roc_auc"],
    "gradient": ["pearson_r", "spearman_rho", "kendall_tau", "mae", "rmse", "roc_auc"],
}


@dataclass
class EvaluationReport:
    method: str
    output: str  # "binary" | "gradient"
    label_kind: str  # "binary" | "gradient"
    n_test: int
    metrics: dict[str, float]
    threshold: str = ""
    ci: dict[str, tuple[float, float]] | None = None
    confusion: pd.DataFrame | None = None
    per_group: pd.DataFrame | None = None
    robustness: pd.DataFrame | None = None
    consistency: dict[str, float] | None = None
    predictions: pd.DataFrame | None = field(default=None, repr=False)
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = [{"metric": k, "value": v} for k, v in self.metrics.items()]
        df = pd.DataFrame(rows)
        if self.ci:
            df["ci_low"] = [self.ci.get(k, (np.nan, np.nan))[0] for k in df["metric"]]
            df["ci_high"] = [self.ci.get(k, (np.nan, np.nan))[1] for k in df["metric"]]
        return df

    def __str__(self) -> str:
        lines = [
            f"Evaluation of '{self.method}' on {self.n_test:,} held-out coded pairs "
            f"({self.output} output, {self.label_kind} labels)"
        ]
        if self.threshold:
            lines.append(f"  decision rule: {self.threshold}")
        for key in _HEADLINE[self.output]:
            if key in self.metrics and not np.isnan(self.metrics[key]):
                ci = ""
                if self.ci and key in self.ci and not np.isnan(self.ci[key][0]):
                    ci = f"  [{self.ci[key][0]:.3f}, {self.ci[key][1]:.3f}]"
                lines.append(f"  {key:<14s} {self.metrics[key]:.4f}{ci}")
        if self.consistency:
            lines.append(
                f"  robustness: predictions agree for {self.consistency['agreement']:.1%} of "
                f"perturbed twins (flip rate {self.consistency['flip_rate']:.1%})"
            )
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)

    __repr__ = __str__

    def to_dict(self) -> dict:
        def frame(df):
            return None if df is None else json.loads(df.to_json(orient="split"))

        return {
            "method": self.method, "output": self.output, "label_kind": self.label_kind,
            "n_test": self.n_test, "metrics": self.metrics, "threshold": self.threshold,
            "ci": self.ci, "confusion": frame(self.confusion), "per_group": frame(self.per_group),
            "robustness": frame(self.robustness), "consistency": self.consistency, "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvaluationReport:
        def frame(x):
            return None if x is None else pd.DataFrame(x["data"], index=x["index"], columns=x["columns"])

        ci = {k: tuple(v) for k, v in d["ci"].items()} if d.get("ci") else None
        return cls(d["method"], d["output"], d["label_kind"], d["n_test"], d["metrics"],
                   d.get("threshold", ""), ci, frame(d.get("confusion")), frame(d.get("per_group")),
                   frame(d.get("robustness")), d.get("consistency"), None, d.get("notes", []))
