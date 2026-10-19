"""Robustness of predictions to perturbed response endings (per-variant metrics, flip rate)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.augmentation import ORIG_ID_COL, VARIANT_COL
from .metrics import binary_metrics, gradient_metrics


def robustness_report(
    df: pd.DataFrame, *, label: np.ndarray, prediction: np.ndarray, output: str,
    on_label_scale: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Metrics per perturbation variant and agreement between originals and their twins.

    ``df`` is an augmented frame (see :func:`simchology.augment`); ``label`` and
    ``prediction`` are aligned with its rows.
    """
    variants = df[VARIANT_COL].to_numpy()
    rows = []
    for variant in pd.unique(variants):
        m = variants == variant
        if output == "binary":
            metrics = binary_metrics(label[m], prediction[m])
        else:
            metrics = gradient_metrics(label[m], prediction[m], on_label_scale=on_label_scale)
        rows.append({"variant": variant, **metrics})
    per_variant = pd.DataFrame(rows)

    ids = df[ORIG_ID_COL].astype(str).to_numpy()
    orig = pd.Series(prediction[variants == "original"], index=ids[variants == "original"])
    pert = pd.Series(prediction[variants != "original"], index=ids[variants != "original"])
    pert = pert[~pert.index.duplicated()]
    common = orig.index.intersection(pert.index)
    o, p = orig.loc[common].to_numpy(), pert.loc[common].to_numpy()
    if output == "binary":
        agree = float(np.mean(o == p)) if len(common) else float("nan")
        consistency = {"pairs": float(len(common)), "agreement": agree, "flip_rate": 1 - agree}
    else:
        diff = np.abs(o - p)
        consistency = {"pairs": float(len(common)), "mean_abs_change": float(np.mean(diff)),
                       "agreement": float(np.mean(diff < 0.05)), "flip_rate": float(np.mean(diff >= 0.05))}
    return per_variant, consistency
